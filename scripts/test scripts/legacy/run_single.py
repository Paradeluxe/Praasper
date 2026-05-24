#!/usr/bin/env python3
"""Run latest Praasper on a single file."""
import sys
sys.path.insert(0, '/mnt/e/praasper')

# Monkey-patch FunASR audio loader to use soundfile instead of ffmpeg
import soundfile as sf
import librosa
import torch
def _load_audio_soundfile(file, sr=16000):
    data, file_sr = sf.read(file, dtype='float32')
    if len(data.shape) > 1:
        data = data.mean(axis=1)
    if file_sr != sr:
        data = librosa.resample(data, orig_sr=file_sr, target_sr=sr)
    return data
from funasr.utils import load_utils
load_utils._load_audio_ffmpeg = _load_audio_soundfile

from praasper import init_model

AUDIO_PATH = '/mnt/e/Corpus/ma/audio/01-1.wav'

model = init_model()
model.annote(input_path=AUDIO_PATH, verbose=False, skip_existing=False, params=None)
print("DONE")
