# server.py
import io

import numpy as np
import soundfile as sf
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from faster_whisper import WhisperModel

app = FastAPI()

ALLOWED_ORIGINS = [
    "http://localhost",
    "http://localhost:3000",
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

PROFANE = {"foo", "bar"}  # replace with your lexicon

def generate_beep(duration_s, sr, freq=1000.0):
    t = np.linspace(0, duration_s, int(sr*duration_s), endpoint=False)
    return (0.2*np.sin(2*np.pi*freq*t)).astype(np.float32)

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
        token = ''.join(ch for ch in w["text"] if ch.isalpha())
        if token in PROFANE:
            censor_spans.append((w["start"], w["end"]))

    # Apply beeps
    y = data.copy()
    for (t0, t1) in censor_spans:
        i0, i1 = int(t0*sr), int(t1*sr)
        dur = max(i1 - i0, int(0.15*sr))  # min 150ms beep
        beep = generate_beep(dur/sr, sr)
        seg = y[i0:i0+len(beep)]
        # pad if needed
        if len(seg) < len(beep):
            pad = np.zeros(len(beep)-len(seg), dtype=np.float32)
            seg = np.concatenate([seg, pad])
        y[i0:i0+len(beep)] = np.clip(seg + beep, -1.0, 1.0)

    # Encode to WAV for return
    out_buf = io.BytesIO()
    sf.write(out_buf, y, sr, format="WAV")
    out_buf.seek(0)
    return {"sample_rate": sr, "censored_wav": out_buf.getvalue().hex()}  # or return as FileResponse
