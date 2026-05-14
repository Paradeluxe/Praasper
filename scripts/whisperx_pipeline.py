#!/usr/bin/env python3
"""
WhisperX Pipeline — openai/whisper-large-v3 (vanilla PyTorch backend)
Model: openai/whisper-large-v3
Output: /mnt/e/Corpus/ma/output/whisperx/{stem}.TextGrid
"""

import sys
import io
from pathlib import Path
from tqdm import tqdm
import textgrid

import whisper

# ── Paths ─────────────────────────────────────────────────────────────────────
AUDIO_DIR   = Path("/mnt/e/Corpus/ma/audio")
OUT_DIR     = Path("/mnt/e/Corpus/ma/output/whisperx")
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_SIZE  = "turbo"
DEVICE      = "cuda"

# ── TextGrid writer ───────────────────────────────────────────────────────────
def intervals_to_textgrid(intervals, audio_duration, out_path):
    tg = textgrid.TextGrid()
    tier = textgrid.IntervalTier(name="words", maxTime=audio_duration)
    prev_end = 0.0
    # Merge overlapping intervals into single intervals
    MIN_DUR = 0.001  # textgrid rejects zero-duration intervals
    merged = []
    for seg in intervals:
        start, end, word = seg["start"], seg["end"], seg["text"]
        if end - start < MIN_DUR:
            end = start + MIN_DUR
        if merged and start <= merged[-1]["end"]:
            # overlap — merge into previous
            merged[-1]["end"] = max(merged[-1]["end"], end)
            merged[-1]["text"] += word
        else:
            merged.append({"start": start, "end": end, "text": word})

    for seg in merged:
        start, end, word = seg["start"], seg["end"], seg["text"]
        if end > audio_duration:
            end = audio_duration
        if start > prev_end + MIN_DUR:
            tier.addInterval(textgrid.Interval(prev_end, start, ""))
        tier.addInterval(textgrid.Interval(start, end, word))
        prev_end = end
    if prev_end < audio_duration:
        tier.addInterval(textgrid.Interval(prev_end, audio_duration, ""))
    tg.append(tier)
    tg.maxTime = audio_duration
    tg.write(str(out_path))


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)

    audio_files = sorted(AUDIO_DIR.glob("*.wav"))
    print(f"[whisperx] {len(audio_files)} files | model={MODEL_SIZE} | device={DEVICE}")

    model = whisper.load_model(MODEL_SIZE, device=DEVICE)

    for af in tqdm(audio_files, desc="whisperx"):
        stem  = af.stem
        out_p = OUT_DIR / f"{stem}.TextGrid"
        if out_p.exists():
            tqdm.write(f"  skip {stem} (exists)")
            continue
        try:
            result = model.transcribe(
                str(af),
                language="zh",
                word_timestamps=True,
            )

            intervals = []
            audio_duration = 0.0
            for seg in result.get("segments", []):
                audio_duration = max(audio_duration, seg.get("end", 0))
                for w in seg.get("words", []):
                    intervals.append({
                        "start": w["start"],
                        "end":   w["end"],
                        "text":  w["word"].strip(),
                    })

            intervals_to_textgrid(intervals, audio_duration, out_p)
            tqdm.write(f"  done {stem}")
        except Exception as e:
            tqdm.write(f"  ERROR {stem}: {e}")

    del model
    print("[whisperx] DONE")


if __name__ == "__main__":
    main()
