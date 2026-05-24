#!/usr/bin/env python3
"""
Run Praasper with DEFAULT SOP (auto_vad grid search per file).
params=None triggers auto_vad internally — this IS Praasper's true default behavior.

Usage:
    /mnt/e/praasper_venv_wsl/bin/python run_default_sop.py
    /mnt/e/praasper_venv_wsl/bin/python run_default_sop.py --skip-existing
"""

import os
import sys
import gc
import argparse
from pathlib import Path

# ── Monkey-patch FunASR ffmpeg loader → soundfile ──────────────────────────────
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

# ── Import Praasper ────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent))
from praasper import init_model

# ── Paths ──────────────────────────────────────────────────────────────────────
AUDIO_DIR   = Path("/mnt/e/Corpus/ma/audio")
OUTPUT_DIR  = Path("/mnt/e/Corpus/ma/output/audio")


def main():
    parser = argparse.ArgumentParser(description="Run Praasper default SOP (auto_vad)")
    parser.add_argument("--audio-dir", type=Path, default=AUDIO_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    files = sorted([f.stem for f in args.audio_dir.glob("*.wav")])
    print(f"Found {len(files)} audio files")
    if not files:
        print("No .wav files found. Exiting.")
        sys.exit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    model = init_model()
    print("Default SOP: params=None → auto_vad grid search per file")
    print("Grid: amp=1.1, eps_ratio in [0.02, 0.025, 0.03, 0.035, 0.04, 0.05]")

    processed = 0
    skipped = 0
    for i, fk in enumerate(files):
        wav_path = args.audio_dir / f"{fk}.wav"
        out_path = args.output_dir / f"{fk}.TextGrid"

        if args.skip_existing and out_path.exists():
            skipped += 1
            print(f"[{i+1}/{len(files)}] {fk}  SKIP (exists)")
            continue

        print(f"[{i+1}/{len(files)}] {fk}  ", end="", flush=True)
        try:
            # params=None = DEFAULT SOP: triggers auto_vad internally
            model.annote(
                input_path=str(wav_path),
                verbose=False,
                skip_existing=False,
                params=None,
            )
            processed += 1
            print("OK")
        except Exception as e:
            print(f"FAIL: {e}")

        gc.collect()

    print(f"\nDone. Processed={processed}  Skipped={skipped}  Total={len(files)}")
    model.release_resources()


if __name__ == "__main__":
    main()
