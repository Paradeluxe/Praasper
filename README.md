# Praasper

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](./LICENSE)
[![PyPI version](https://img.shields.io/pypi/v/praasper.svg)](https://pypi.org/project/praasper/)
[![PyPI Downloads](https://img.shields.io/pypi/dm/praasper.svg?label=downloads)](https://pypi.org/project/praasper/)
![Python](https://img.shields.io/badge/python-3.10_|_3.11_|_3.12-blue.svg)

**[Setup](#setup)** | **[Usage](#how-to-use)** | **[Mechanism](#mechanism)**

***Praasper*** is a speech processing framework designed to help researchers transcribe audio files into word-level timestamps — from a single word to a complete sentence — with high accuracy in both transcription and timestamps.

![mechanism](promote/mechanism.png)

In ***Praasper***, the pipeline has four stages. **First**, long recordings are split at natural pauses via pause-aware chunking. **Second**, **VAD** (*Praditor*) performs coarse DBSCAN clustering followed by fine sliding-window boundary detection — automatically calibrated per file via a two-stage grid search (amp × eps_ratio → numValid refinement). **Third**, **ASR** (*Fun-ASR-Nano*) transcribes each VAD-bounded segment with word-level timestamps. **Fourth**, timestamps are aligned to VAD intervals by temporal overlap and exported as a Praat TextGrid file.
# How to use

Here is one of the **simplest** examples:

```python
import praasper

model = praasper.init_model()
model.annote("data_folder")
```

### `init_model()` parameters

|     Param    | Required |     Type    |            Default            | Description                                                                                                                                |
| :----------: | :------: | :---------: | :---------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------- |
| `infer_mode` | optional |    `str`    |           `"local"`           | ASR backend: `"local"` for on-device FunASR-Nano, `"api"` for DashScope cloud API.                                                         |
|   `device`   | optional |    `str`    |            `"auto"`           | Hardware for local inference: `"auto"`, `"cuda"`, or `"cpu"`. Ignored in API mode.                                                         |
|     `ASR`    | optional |    `str`    | FunAudioLLM/Fun-ASR-Nano-2512 | Advanced: override the default local ASR model. See [FunASR model zoo](https://github.com/modelscope/funasr?tab=readme-ov-file#model-zoo). |
|   `api_key`  |   API    | `str\|None` |             `None`            | Required when `infer_mode="api"`. Can also be set via `DASHSCOPE_API_KEY` env var.                                                          |
| `cache_dir`  | optional | `str\|None` |             `None`            | Directory for caching ASR models. When set, `HF_HOME` / `MODELSCOPE_CACHE` / `TRANSFORMERS_CACHE` are redirected here.                      |
|   `effort`   | optional |    `str`    |          `"medium"`           | Grid search depth: `"low"` (3 combos), `"medium"` (22 combos), or `"high"` (100 combos). Can be overridden per-run via `annote()`.          |

### `annote()` parameters

|     Param      | Required |     Type    |   Default   | Description                                                       |
| :------------: | :------: | :---------: | :---------: | :---------------------------------------------------------------- |
| `input_path`   |   yes    |    `str`    |      —      | Path to a `.wav` file or a folder of `.wav` files.               |
|   `seg_dur`   | optional |   `float`   |    15.     | Maximum segment duration in seconds.                              |
|  `min_pause`  | optional |   `float`   |    0.2     | Minimum pause between utterances in seconds.                      |
| `skip_existing` | optional |   `bool`    |   `False`   | Skip files that already have an output `.TextGrid`.               |
|   `verbose`   | optional |   `bool`    |   `False`   | Print verbose progress messages.                                  |
|   `effort`    | optional | `str\|None` | (inherit)  | Override `init_model` effort. `None` = use init_model default.   |
|   `params`    | optional | `dict\|str\|None` | `None` | Custom VAD params: a dict, a `.txt` file path, or `None` (auto). |

Here are code examples showing how to use these parameters:

```python
import praasper

# Local inference (default)
model = praasper.init_model()

# Local inference on GPU
model = praasper.init_model(device="cuda")

# DashScope cloud API
model = praasper.init_model(infer_mode="api", api_key="sk-...")

# Custom local ASR model (advanced)
model = praasper.init_model(ASR="iic/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch")

# Low-effort grid search (3 combos, fastest)
model = praasper.init_model(effort="low")

# High-effort grid search (100 combos, most thorough VAD calibration)
model = praasper.init_model(effort="high")

model.annote(
    input_path="data_folder",
    min_pause=.8,
    seg_dur=15.
)
```

## Custom VAD Parameters

By default, ***Praasper*** automatically calibrates VAD parameters for each recording via a grid search — no manual tuning is required. For advanced users who need custom parameters, a dict or `.txt` file can be passed directly.

Parameters use the internal ***Praditor*** format: a dict with `onset` and `offset` sections, each containing `amp`, `cutoff0`, `cutoff1`, `numValid`, and `eps_ratio`. Praasper keeps onset and offset identical internally.

Here is a code example showing how to override the default parameters for a specific audio file. The VAD parameters are passed as a dict with `onset` and `offset` sections:

```python
import praasper

model = praasper.init_model()

# Define custom VAD parameters
custom_params = {
    "onset":  {"amp": "1.05", "cutoff0": "60", "cutoff1": "10800", "numValid": "475", "eps_ratio": "0.05"},
    "offset": {"amp": "1.05", "cutoff0": "60", "cutoff1": "10800", "numValid": "475", "eps_ratio": "0.05"},
}

model.annote(
    input_path="data_folder",
    params=custom_params,
)
```

Alternatively, save the parameters to a `.txt` file and pass the file path instead:

```python
model.annote(
    input_path="data_folder",
    params="/path/to/custom_params.txt",
)
```

In both cases, Praasper will use your custom VAD parameters instead of running the auto grid search. To export the current parameters to a `.txt` file for later reuse:

```python
model.export_params("/path/to/custom_params.txt")
```

## ASR: local vs. cloud

Praasper supports two ASR backends, chosen via the `infer_mode` parameter:

| `infer_mode`        | Backend                                                     | Best for                                |
| :------------------ | :---------------------------------------------------------- | :-------------------------------------- |
| `"local"` (default) | **FunASR-Nano** — lightweight, runs on laptop CPU/GPU       | Offline work, no API costs, privacy     |
| `"api"`             | **DashScope** (AliCloud) — cloud ASR with stronger accuracy | High-accuracy needs, server deployments |

### Local mode (`infer_mode="local"`)

```python
model = praasper.init_model()                    # auto-detect GPU/CPU
model = praasper.init_model(device="cuda")       # force GPU
model = praasper.init_model(device="cpu")        # force CPU
```

The default model is `FunAudioLLM/Fun-ASR-Nano-2512`, which supports Chinese, English, and Japanese with word-level timestamps. Power users can swap in a different FunASR model via the `ASR` parameter.

### API mode (`infer_mode="api"`)

```python
model = praasper.init_model(infer_mode="api", api_key="sk-...")
# or set DASHSCOPE_API_KEY environment variable:
# model = praasper.init_model(infer_mode="api")
```

> **Note:** DashScope requires audio files to be hosted at a public URL (HTTP/HTTPS/OSS). Local file paths are not supported in API mode.

# Mechanism

***Praasper*** processes audio in four stages:

**1. Pause-aware chunking.** Long recordings are split into segments (default 15 s) at natural pauses detected by the VAD, placing boundaries at silence-gap midpoints. If no gap is found, the threshold relaxes until a boundary can be placed. This preserves utterance integrity across chunks.

**2. Voice Activity Detection (VAD).** Praasper uses ***Praditor***, a DBSCAN-based detector. The first stage clusters the amplitude envelope to separate speech from silence into broad candidate segments. The second stage applies a sliding-window detector with locally estimated noise thresholds to place onset and offset boundaries at millisecond precision. By default, Praasper auto-calibrates per recording via `effort`:

| Effort    | Stage 1                              | Stage 2                | Total combos |
|-----------|--------------------------------------|------------------------|:------------:|
| `"low"`   | `amp`=[1.2] × `eps_ratio`=[0.02,0.03,0.04]            | skipped                | 3            |
| `"normal"`| `amp`=[1.1,1.2,1.3] × `eps_ratio`=[0.02–0.05, 6 steps] | `numValid`=[500–5000]  | 22           |
| `"high"`  | 8 `amp` × 6 `eps_ratio` × `cutoff0`=[0,200]            | `numValid`=[500–5000]  | 100          |

Stage 1 maximises onset boundary SNR; Stage 2 refines `numValid` (DBSCAN min points) around the winner. Manual tuning is available but not required.

**3. Automatic Speech Recognition (ASR).** Each VAD-bounded segment is transcribed by **Fun-ASR-Nano**, a lightweight model producing word-level timestamps. It supports Chinese (Mandarin and 7 dialects: Wu, Cantonese, Min, Hakka, Gan, Xiang, Jin), English, and Japanese. For higher accuracy, DashScope cloud ASR is available via `infer_mode="api"`.

**4. Overlap matching and export.** ASR word timestamps are matched to VAD intervals by maximum temporal overlap. Unmatched words are assigned to the nearest interval by distance. Words in the same interval are concatenated; empty intervals are discarded. Adjacent overlapping intervals are merged. Processing runs in parallel (4 workers). Output is a Praat-compatible TextGrid file.

Advanced users can override the auto-calibration by supplying custom VAD parameters via a Python dict or `.txt` file (see [Custom VAD Parameters](#custom-vad-parameters)).

# Setup

## pip installation

```bash
pip install -U praasper
```

> If you have a successful installation and don't care about GPU acceleration, you can stop right here.

## GPU Acceleration (Windows/Linux)

**Praasper** uses `Fun-ASR-Nano` from [FunASR](https://github.com/modelscope/funasr) as the default local ASR engine.
Cloud ASR is also available via DashScope (`infer_mode="api"`).

> FunASR needs a GPU build of `torch` for CUDA acceleration. Praasper auto-detects your hardware
> at startup — if it finds a GPU driver but no CUDA torch, it prints the exact `pip` command you need.

- **macOS**: CPU only.
- **Windows/Linux**: CUDA → CPU fallback.

Check your driver's CUDA version:

```bash
nvidia-smi
```

Example output (driver supports CUDA 13.0):

```
| NVIDIA-SMI 576.80    Driver Version: 576.80    CUDA Version: 13.0 |
```

Install a `torch` build ≤ your driver version (e.g. driver 13.0 → `cu130`, `cu129`, `cu128` all work).
See [all available CUDA builds](https://pytorch.org/get-started/previous-versions/) for a full list.

```bash
pip install --force-reinstall torch torchaudio --index-url https://download.pytorch.org/whl/cu130
```

Or with `uv`:

```bash
uv pip install --force-reinstall torch torchaudio --index-url https://download.pytorch.org/whl/cu130
```

> If you're unsure which version to use, just import `praasper` — it will tell you.

## ffmpeg requirement (new in 0.7.4.post1)

Praasper uses FunASR-Nano for local ASR. FunASR needs an `ffmpeg` binary to
decode audio on systems that lack the `torchcodec` backend for `torchaudio.load`.
Praasper resolves `ffmpeg` automatically at `init_model()` time, in this order:

1. **System `ffmpeg` on `PATH`** — Fastest path. Install once via your OS
   package manager (`apt`, `choco`, `brew`, etc.) — see [ffmpeg.org/download.html](https://ffmpeg.org/download.html)
   for binaries and instructions for every platform.
2. **Bundled binary** — Praasper automatically detects that no system `ffmpeg`
   is present and downloads a bundled `ffmpeg` (~90MB) into the active venv
   on first `init_model()`. This happens transparently — you don't need to
   install anything manually.

If both sources fail (e.g. `static-ffmpeg` was uninstalled or blocked),
Praasper raises a `RuntimeError` at `init_model()` time with the exact
fix to install.

| Your situation | What you need to do |
|---|---|
| I'm comfortable installing system software | Install `ffmpeg` once — see [ffmpeg.org/download.html](https://ffmpeg.org/download.html) |
| I don't want to install anything manually | None — Praasper auto-detects missing ffmpeg and downloads the bundled binary on first `init_model()` |
| `static-ffmpeg` was uninstalled / blocked | `pip install static-ffmpeg`, then call `init_model()` again |


## License

Praasper is **dual-licensed** under AGPL v3 + a commercial license ([LICENSE](./LICENSE)):

- **AGPL v3** (default): free, open source. Academic / personal / non-profit /
  small orgs can use it directly. Only requirement: if you offer Praasper as a
  network service, you must make the source available.
- **Commercial License**: if you cannot accept AGPL copyleft obligations (e.g.
  commercial products, SaaS, large org internal use), purchase a commercial
  license to waive AGPL terms.

See [COMMERCIAL-LICENSE.md](./COMMERCIAL-LICENSE.md) for details.
