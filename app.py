# server.py
import io

import numpy as np
import soundfile as sf
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from faster_whisper import WhisperModel

from profanity import detect_profane_spans

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

_model: WhisperModel | None = None


def get_model() -> WhisperModel:
    """Instantiate the Whisper model on first use."""

    global _model
    if _model is None:
        _model = WhisperModel("tiny.en", compute_type="int8")  # good CPU perf
    return _model

def generate_beep(duration_s, sr, freq=1000.0):
    """Generate a loud enough pure tone to fully mask speech."""
    t = np.linspace(0, duration_s, int(sr * duration_s), endpoint=False)
    # Use a slightly hotter amplitude than before so the underlying audio is
    # completely replaced when we overwrite the segment.
    return (0.6 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


@app.post("/censor")
async def censor_audio(file: UploadFile = File(...)):
    # Load audio
    data, sr = sf.read(io.BytesIO(await file.read()), dtype="float32", always_2d=False)
    if data.ndim > 1:
        data = np.mean(data, axis=1)  # mono

    # Transcribe with word timestamps
    model = get_model()
    segments, info = model.transcribe(data, word_timestamps=True, vad_filter=True)
    words = []
    for seg in segments:
        for w in seg.words or []:
            words.append({"text": w.word.strip().lower(), "start": w.start, "end": w.end})

    # Find profane words
    censor_spans = detect_profane_spans(words)

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
