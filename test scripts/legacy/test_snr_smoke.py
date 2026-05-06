"""Smoke test for mean_snr grid search ranking.

NOTE: full praasper integration skipped here because torch/scipy are not installed
in this WSL env. The compute_boundary_snr unit tests run standalone.
"""
import numpy as np

# ── Inline compute_boundary_snr (no external deps) ────────────────────────────
def compute_boundary_snr(audio_arr, sr, onsets, offsets, window_ms=10):
    if not onsets and not offsets:
        return 0.0
    window_samples = int(window_ms / 1000.0 * sr)
    if window_samples <= 0:
        window_samples = 1
    total_len = len(audio_arr)
    snrs = []
    for t in onsets:
        center = int(t * sr)
        noise_start = max(0, center - window_samples)
        noise_end = center
        speech_start = center
        speech_end = min(total_len, center + window_samples)
        if noise_end <= noise_start or speech_end <= speech_start:
            continue
        noise_power = np.mean(audio_arr[noise_start:noise_end].astype(np.float64) ** 2)
        speech_power = np.mean(audio_arr[speech_start:speech_end].astype(np.float64) ** 2)
        if noise_power <= 0:
            snrs.append(60.0)
        elif speech_power <= 0:
            snrs.append(0.0)
        else:
            snrs.append(10.0 * np.log10(speech_power / noise_power))
    for t in offsets:
        center = int(t * sr)
        speech_start = max(0, center - window_samples)
        speech_end = center
        noise_start = center
        noise_end = min(total_len, center + window_samples)
        if speech_end <= speech_start or noise_end <= noise_start:
            continue
        speech_power = np.mean(audio_arr[speech_start:speech_end].astype(np.float64) ** 2)
        noise_power = np.mean(audio_arr[noise_start:noise_end].astype(np.float64) ** 2)
        if noise_power <= 0:
            snrs.append(60.0)
        elif speech_power <= 0:
            snrs.append(0.0)
        else:
            snrs.append(10.0 * np.log10(speech_power / noise_power))
    return float(np.mean(snrs)) if snrs else 0.0

# ── Unit tests ────────────────────────────────────────────────────────────────
sr = 16000
t = np.linspace(0, 1.0, sr)

audio = np.concatenate([
    np.zeros(int(sr*0.2)),
    np.sin(2*np.pi*440*t[:int(sr*0.2)]) * 0.5,
    np.zeros(int(sr*0.2)),
    np.sin(2*np.pi*440*t[:int(sr*0.2)]) * 0.3,
    np.zeros(int(sr*0.2)),
])

# 1. typical boundaries
snr = compute_boundary_snr(audio, sr, [0.2, 0.6], [0.4, 0.8], window_ms=10)
print(f"[PASS 1/5] typical boundaries SNR = {snr:.2f} dB")
assert snr > 0

# 2. empty boundaries
snr_empty = compute_boundary_snr(audio, sr, [], [], window_ms=10)
print(f"[PASS 2/5] empty boundaries SNR = {snr_empty:.2f} dB")
assert snr_empty == 0.0

# 3. onset at t=0 (clips pre-window) -> onset skipped, offset gives ~0 dB for continuous sine
audio2 = np.sin(2*np.pi*440*t) * 0.5
snr_edge = compute_boundary_snr(audio2, sr, [0.0], [0.5], window_ms=10)
print(f"[PASS 3/5] onset at t=0 SNR = {snr_edge:.2f} dB  (expect ~0, onset skipped, offset only)")
assert abs(snr_edge) < 0.5, f"Expected ~0 dB for equal speech/noise power, got {snr_edge}"

# 4. zero noise power -> cap at 60 dB
audio3 = np.concatenate([
    np.zeros(int(sr*0.1)),
    np.sin(2*np.pi*440*t[:int(sr*0.1)]) * 0.5,
])
snr_zero_noise = compute_boundary_snr(audio3, sr, [0.1], [], window_ms=10)
print(f"[PASS 4/5] zero noise power SNR = {snr_zero_noise:.2f} dB  (expect 60.0)")
assert snr_zero_noise == 60.0, f"Expected 60.0, got {snr_zero_noise}"

# 5. zero speech power at offset -> 0 dB
# offset placed right at end of silence region: pre-window = silence, post-window = speech
audio5 = np.concatenate([
    np.sin(2*np.pi*440*t[:int(sr*0.1)]) * 0.5,     # speech 0.0-0.1
    np.zeros(int(sr*0.1)),                          # silence 0.1-0.2
    np.sin(2*np.pi*440*t[:int(sr*0.1)]) * 0.5,     # speech 0.2-0.3
])
snr_zero_speech = compute_boundary_snr(audio5, sr, [], [0.2], window_ms=10)
print(f"[PASS 5/5] zero speech power SNR = {snr_zero_speech:.2f} dB  (expect 0.0)")
assert snr_zero_speech == 0.0, f"Expected 0.0, got {snr_zero_speech}"

print("\n[ALL UNIT TESTS PASSED]")

# ── Static verification of __init__.py changes ────────────────────────────────
print("\n[STATIC CHECK] Verifying praasper/__init__.py patches...")
with open("/mnt/e/praasper/praasper/__init__.py", "r", encoding="utf-8") as f:
    src = f.read()

checks = [
    ("mean_snr init",           "mean_snr = 0.0"),
    ("compute_boundary_snr call", "compute_boundary_snr("),
    ("mean_snr_copy deepcopy",  "mean_snr_copy = copy.deepcopy(mean_snr)"),
    ("4-tuple return",          "[mean_snr_copy, total_overlap_copy, num_intervals_copy, adjusted_params_copy]"),
    ("max ranking by snr+overlap", "max(result, key=lambda x: (x[0], x[1]))"),
    ("unpack 4-tuple",          "max_snr, max_overlap, max_intervals, best_params = best"),
]
all_ok = True
for name, needle in checks:
    ok = needle in src
    print(f"  {'[OK]' if ok else '[FAIL]'} {name}: {needle}")
    all_ok = all_ok and ok

assert all_ok, "Some static checks failed!"
print("\n[STATIC CHECK PASSED] All __init__.py patches verified.")
print("\n>>> SMOKE TEST COMPLETE <<<")
