# server.py
import io

import numpy as np
import soundfile as sf
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from faster_whisper import WhisperModel

from profanity import is_profane_word

app = FastAPI()

ALLOWED_ORIGINS = [
    "http://localhost",
    "http://localhost:3000",
    "https://pcpartguide.com",
    "https://www.pcpartguide.com",
]

PCPARTGUIDE_REGEX = r"https://([a-z0-9-]+\.)*pcpartguide\.com$"

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_origin_regex=PCPARTGUIDE_REGEX,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
model = WhisperModel("tiny.en", compute_type="int8")  # good CPU perf

def generate_beep(duration_s, sr, freq=1000.0):
    """Generate a loud enough pure tone to fully mask speech."""
    t = np.linspace(0, duration_s, int(sr * duration_s), endpoint=False)
    # Use a slightly hotter amplitude than before so the underlying audio is
    # completely replaced when we overwrite the segment.
    return (0.6 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def merge_spans(spans):
    if not spans:
        return []
    spans = sorted(spans)
    merged = [spans[0]]
    for start, end in spans[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged

@app.post("/censor")
async def censor_audio(file: UploadFile = File(...)):
    # Load audio
    data, sr = sf.read(io.BytesIO(await file.read()), dtype="float32", always_2d=False)
    if data.ndim > 1:
        data = np.mean(data, axis=1)  # mono

    # Transcribe with word timestamps
    segments, info = model.transcribe(data, word_timestamps=True, vad_filter=True)
    words = []
    for seg in segments:
        for w in seg.words or []:
            words.append({"text": w.word.strip().lower(), "start": w.start, "end": w.end})

    # Find profane words
    censor_spans = []
    for w in words:
        if is_profane_word(w["text"]):
            # Pad start/end slightly to ensure the whole word is covered even if
            # Whisper under/overshoots the timestamps by a handful of frames.
            censor_spans.append((
                max(0.0, (w["start"] or 0.0) - 0.05),
                (w["end"] or 0.0) + 0.1,
            ))

    censor_spans = merge_spans(censor_spans)

    # Apply beeps
    y = data.copy()
    n_samples = len(y)
    for (t0, t1) in censor_spans:
        i0, i1 = int(t0 * sr), int(t1 * sr)
        i0 = max(0, i0)
        i1 = min(n_samples, max(i0 + int(0.15 * sr), i1))
        if i0 >= i1:
            continue

        seg_len = i1 - i0
        beep = generate_beep(seg_len / sr, sr)
        if len(beep) < seg_len:
            beep = np.pad(beep, (0, seg_len - len(beep)), mode="edge")
        elif len(beep) > seg_len:
            beep = beep[:seg_len]
        y[i0:i1] = beep

    # Encode to WAV for return
    out_buf = io.BytesIO()
    sf.write(out_buf, y, sr, format="WAV")
    out_buf.seek(0)
    return {"sample_rate": sr, "censored_wav": out_buf.getvalue().hex()}  # or return as FileResponse
