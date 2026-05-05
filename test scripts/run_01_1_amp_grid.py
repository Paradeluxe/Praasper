#!/usr/bin/env python3
import os, sys
sys.path.insert(0, '/mnt/e/praasper')

import soundfile as sf
import librosa
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

AUDIO_DIR = '/mnt/e/Corpus/ma/audio'
OUTPUT_DIR = '/mnt/e/Corpus/ma/output/audio'
os.makedirs(OUTPUT_DIR, exist_ok=True)

fk = '01-1'
wav = f'{AUDIO_DIR}/{fk}.wav'
out_path = f'{OUTPUT_DIR}/{fk}.TextGrid'

# Remove old output if exists to force rerun
if os.path.exists(out_path):
    os.remove(out_path)
    print(f'{fk}: removed old output')

model = init_model()
print(f'{fk}: running with amp grid [1.1, 1.2, 1.3] ...', flush=True)
model.annote(input_path=wav, verbose=False, skip_existing=False, params=None)
print(f'{fk}: done')
