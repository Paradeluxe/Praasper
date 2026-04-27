# Praasper
[![PyPI Downloads](https://img.shields.io/pypi/dm/praasper.svg?label=PyPI%20downloads)](
https://pypi.org/project/praasper/)
![Python](https://img.shields.io/badge/python->=3.10-blue.svg)
![GitHub License](https://img.shields.io/github/license/Paradeluxe/Praasper)

[**Setup**](#setup) | [**Usage**](#how-to-use) | [**Mechanism**](#mechanism)

***Praasper*** is a speech processing framework designed to help researchers transcribe audio files into word-level timestamps — from a single word to a complete sentence — with high accuracy in both transcription and timestamps.

![mechanism](promote/mechanism.png)

In ***Praasper***, the pipeline has three steps. First, **VAD** (*Praditor*) segments audio into chunks and extracts speech intervals with millisecond-level precision. Second, **ASR** (*Fun-ASR-Nano*) transcribes each chunk with word-level timestamps. Third, timestamps are aligned to VAD intervals and exported as a TextGrid file where each interval contains one word and its start/end time.

# How to use

Here is one of the **simplest** examples:

```python
import praasper

model = praasper.init_model()
model.annote("data_folder")
```

Here are the parameters you can pass to `init_model` and `annote`:

| Param | Default | Description |
| :---: | :---: | :--- |
| `infer_mode` | `"local"` | ASR backend: `"local"` for on-device FunASR-Nano, `"api"` for DashScope cloud API. |
| `device` | `"auto"` | Hardware for local inference: `"auto"`, `"cuda"`, or `"cpu"`. Ignored in API mode. |
| `ASR` | FunAudioLLM/Fun-ASR-Nano-2512 | Advanced: override the default local ASR model. See [FunASR model zoo](https://github.com/modelscope/funasr?tab=readme-ov-file#model-zoo). |
| `api_key` | `None` | DashScope API key. Required when `infer_mode="api"`. Can also be set via `DASHSCOPE_API_KEY` env var. |
| `input_path` | — | Path to the folder where audio files are stored. |
| `seg_dur` | 10. | Segment large audio into pieces, in seconds. |
| `min_pause` | 0.2 | Minimum pause duration between two utterances, in seconds. |


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

model.annote(
    input_path="data_folder",
    min_pause=.8,
    seg_dur=15.
)
```

## Fine-tune *Praditor*

***Praasper*** is embedded with a default set of parameters for ***Praditor***. But the default parameters may not always be optimal. In that case, you are recommended to use a custom set of parameters for ***Praditor***.

1. Use the latest version of [***Praditor* (v1.3.1)**](https://github.com/Paradeluxe/Praditor/releases). It supports VAD.
2. Annotate the audio file. Fine-tune the parameters until the results fit your standard.
3. Click `Save` under the `Current` mode (top-right corner).

***Praditor*** will then save a `.txt` param file to the same folder as the input audio file, which ***Praasper*** will use to override the default params.

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

| `infer_mode` | Backend | Best for |
|:---|:---|:---|
| `"local"` (default) | **FunASR-Nano** — lightweight, runs on laptop CPU/GPU | Offline work, no API costs, privacy |
| `"api"` | **DashScope** (AliCloud) — cloud ASR with stronger accuracy | High-accuracy needs, server deployments |

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

***Praditor*** is applied to perform **Voice Activity Detection (VAD)** — it segments large audio files into smaller pieces and extracts speech intervals with **millisecond-level precision**. It was originally developed as a Speech Onset Detection (SOT) algorithm for language researchers.

The default ASR engine is **Fun-ASR-Nano** (local mode), a lightweight model that runs on laptop CPU/GPU and supports multiple languages including Mandarin, Cantonese, English, Japanese, and Korean — all with word-level timestamps. For higher accuracy, Praasper can also use **DashScope** cloud ASR via `infer_mode="api"`.

The VAD intervals and ASR timestamps are then aligned via overlap matching: each word from the ASR output is assigned to the VAD interval it overlaps with most. The result is exported as a TextGrid file where each interval contains one word with its start time, end time, and text.

# Setup

## pip installation

```bash
pip install -U praasper
```
> If you have a successful installation and don't care about GPU acceleration, you can stop right here.

## GPU Acceleration (Windows/Linux)

Currently, ***Praasper*** utilizes `Fun-ASR-Nano` from [**FunASR**](https://github.com/modelscope/funasr) as the default local ASR engine. Cloud ASR is also available via DashScope (`infer_mode="api"`).

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

**Next**, go to [**NVIDIA CUDA Toolkit**](https://developer.nvidia.com/cuda-toolkit) and download the latest version, or whichever version that fits your system/need.

**Lastly**, install `torch` that fits your CUDA version. Find the correct `pip` command [**in this link**](https://pytorch.org/get-started/locally/).

Here is an example for CUDA 12.9:

```bash
pip install --reinstall torch torchaudio --index-url https://download.pytorch.org/whl/cu129
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
uv pip install --reinstall torch torchaudio --index-url https://download.pytorch.org/whl/cu129
```
