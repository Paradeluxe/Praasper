#!/usr/bin/env python3
"""
Run default Praasper (NO grid search) on 208 audio files.
Pure defaults: amp=1.2, cutoff0=0, cutoff1=5400, numValid=5000, eps_ratio=0.03

Usage:
    /mnt/e/praasper_venv_wsl/bin/python run_default_praasper.py
    /mnt/e/praasper_venv_wsl/bin/python run_default_praasper.py --skip-existing
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
    parser = argparse.ArgumentParser(description="Run default Praasper on all 208 files")
    parser.add_argument("--audio-dir", type=Path, default=AUDIO_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--skip-existing", action="store_true", help="Skip if output TextGrid exists")
    args = parser.parse_args()

    files = sorted([f.stem for f in args.audio_dir.glob("*.wav")])
    print(f"Found {len(files)} audio files in {args.audio_dir}")
    if not files:
        print("No .wav files found. Exiting.")
        sys.exit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Use pure default params (no grid search)
    model = init_model()
    default_params = init_model.default_params
    print(f"Default params: {default_params}")

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
            model.annote(
                input_path=str(wav_path),
                verbose=False,
                skip_existing=False,
                params=default_params,  # <-- pure default, NO auto_vad grid
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
