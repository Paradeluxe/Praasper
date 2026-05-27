# Praasper

[![PyPI version](https://img.shields.io/pypi/v/praasper.svg)](https://pypi.org/project/praasper/)
[![PyPI Downloads](https://img.shields.io/pypi/dm/praasper.svg?label=downloads)](https://pypi.org/project/praasper/)
![Python](https://img.shields.io/badge/python-3.10_|_3.11_|_3.12-blue.svg)
![GitHub License](https://img.shields.io/github/license/Paradeluxe/Praasper)

**[Setup](#setup)** | **[Usage](#how-to-use)** | **[Mechanism](#mechanism)**

***Praasper*** is a speech processing framework designed to help researchers transcribe audio files into word-level timestamps — from a single word to a complete sentence — with high accuracy in both transcription and timestamps.

![mechanism](promote/mechanism.png)

In ***Praasper***, the pipeline has four stages. **First**, long recordings are split at natural pauses via pause-aware chunking. **Second**, a two-stage **VAD** (*Praditor*) performs coarse DBSCAN clustering followed by fine sliding-window boundary detection — automatically calibrated per file via a two-stage grid search (amp × eps_ratio → numValid refinement). **Third**, **ASR** (*Fun-ASR-Nano*) transcribes each VAD-bounded segment with word-level timestamps. **Fourth**, timestamps are aligned to VAD intervals by temporal overlap and exported as a Praat TextGrid file.

# How to use

Here is one of the **simplest** examples:

```python
import praasper

model = praasper.init_model()
model.annote("data_folder")
```

Here are the parameters you can pass to `init_model` and `annote`:

|     Param    |            Default            | Description                                                                                                                                |
| :----------: | :---------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------- |
| `infer_mode` |           `"local"`           | ASR backend: `"local"` for on-device FunASR-Nano, `"api"` for DashScope cloud API.                                                         |
|   `device`   |            `"auto"`           | Hardware for local inference: `"auto"`, `"cuda"`, or `"cpu"`. Ignored in API mode.                                                         |
|     `ASR`    | FunAudioLLM/Fun-ASR-Nano-2512 | Advanced: override the default local ASR model. See [FunASR model zoo](https://github.com/modelscope/funasr?tab=readme-ov-file#model-zoo). |
|   `api_key`  |             `None`            | DashScope API key. Required when `infer_mode="api"`. Can also be set via `DASHSCOPE_API_KEY` env var.                                      |
| `cache_dir`  |             `None`            | Directory for caching ASR models. When set, `HF_HOME` / `MODELSCOPE_CACHE` / `TRANSFORMERS_CACHE` are redirected here.                      |
|   `effort`   |          `\"normal\"`          | Grid search level: `\"normal\"` (22 combos) or `\"high\"` (100 combos). `\"high\"` adds `cutoff0` sweep and 0.05-step `amp`.                    |
| `input_path` |               —               | Path to the folder where audio files are stored.                                                                                           |
|   `seg_dur`  |              15.              | Segment large audio into pieces, in seconds.                                                                                               |
|  `min_pause` |              0.2              | Minimum pause duration between two utterances, in seconds.                                                                                 |
| `skip_existing` |          `False`           | Skip files that already have an output `.TextGrid`.                                                                                        |
|   `verbose`  |            `False`            | Print verbose progress messages during processing.                                                                                         |
|   `effort`   |          `\"normal\"`          | Grid search effort passed to `annote()`: overrides `init_model` effort for a single run. `\"normal\"` or `\"high\"`.                           |

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

# High-effort grid search (100 combos, more thorough VAD calibration)
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

The default model is `FunAudioLLM/Fun-ASR-Nano-2512`, which supports Mandarin, Cantonese, English, Japanese, and Korean with word-level timestamps. Power users can swap in a different FunASR model via the `ASR` parameter.

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

**2. Voice Activity Detection (VAD).** Praasper uses ***Praditor***, a DBSCAN-based two-stage detector. The first stage clusters the amplitude envelope to separate speech from silence into broad candidate segments. The second stage applies a sliding-window detector with locally estimated noise thresholds to place onset and offset boundaries at millisecond precision. By default, Praasper auto-calibrates per recording: a two-stage grid search selects the best parameters: Stage 1 sweeps `amp` (1.1–1.3) × `eps_ratio` (0.02–0.05) to maximise onset boundary SNR, then Stage 2 refines `numValid` (DBSCAN min points) around the winner. With `effort=\"high\"`, the grid expands to 8 `amp` values (0.05 step) and a `cutoff0` sweep [0, 200 Hz], totalling 100 combos. Manual tuning is available but not required.

**3. Automatic Speech Recognition (ASR).** Each VAD-bounded segment is transcribed by **Fun-ASR-Nano**, a lightweight model producing word-level timestamps (Mandarin, English, Japanese, and Chinese dialects). For higher accuracy, DashScope cloud ASR is available via `infer_mode="api"`.

**4. Overlap matching and export.** ASR word timestamps are matched to VAD intervals by maximum temporal overlap. Unmatched words are assigned to the nearest interval by distance. Words in the same interval are concatenated; empty intervals are discarded. Adjacent overlapping intervals are merged. Processing runs in parallel (4 workers). Output is a Praat-compatible TextGrid file.

Advanced users can override the auto-calibration by supplying custom VAD parameters via a Python dict or `.txt` file (see [Custom VAD Parameters](#custom-vad-parameters)).

# Setup

## pip installation

```bash
pip install -U praasper
```

> If you have a successful installation and don't care about GPU acceleration, you can stop right here.

## GPU Acceleration (Windows/Linux)

Currently, ***Praasper*** utilizes `Fun-ASR-Nano` from **[FunASR](https://github.com/modelscope/funasr)** as the default local ASR engine. Cloud ASR is also available via DashScope (`infer_mode="api"`).

> `FunASR` automatically detects the best currently available device to use. But you still need to first install the GPU-support version of `torch` in order to enable CUDA acceleration.

- For **macOS** users, only `CPU` is supported as the processing device.
- For **Windows/Linux** users, the priority order should be: `CUDA` -> `CPU`.

If you have no experience in installing `CUDA`, follow the steps below:

**First**, go to command line and check the latest CUDA version your system supports:

```bash
nvidia-smi
```

Results should pop up like this (It means that this device supports CUDA up to version 12.9).

```bash
| NVIDIA-SMI 576.80                 Driver Version: 576.80         CUDA Version: 12.9     |
```

**Next**, go to **[NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit)** and download the latest version, or whichever version that fits your system/need.

**Lastly**, install `torch` that fits your CUDA version. Find the correct `pip` command **[in this link](https://pytorch.org/get-started/locally/)**.

Here is an example for CUDA 12.9:

```bash
pip install --force-reinstall torch torchaudio --index-url https://download.pytorch.org/whl/cu129
```

## (Advanced) uv installation

`uv` is also highly recommended for a much **faster** installation. First, make sure `uv` is installed in your default environment:

```bash
pip install uv
```

Then, create a virtual environment (e.g., `.venv`):

```bash
uv venv .venv
```

You should see a new `.venv` folder appear in your project directory. (You may also want to restart the terminal.)

Lastly, install `praasper` (by prefixing `pip` with `uv`):

```bash
uv pip install -U praasper
```

For `CUDA` support, here is an example for downloading `torch` that fits CUDA 12.9:

```bash
uv pip install --force-reinstall torch torchaudio --index-url https://download.pytorch.org/whl/cu129
```

