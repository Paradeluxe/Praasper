# Praasper
[![PyPI Downloads](https://img.shields.io/pypi/dm/praasper.svg?label=PyPI%20downloads)](
https://pypi.org/project/praasper/)
![Python](https://img.shields.io/badge/python->=3.10-blue.svg)
![GitHub License](https://img.shields.io/github/license/Paradeluxe/Praasper)


**Praasper** is an Automatic Speech Recognition (ASR) application designed help researchers transribe audio files to both **word-** and **phoneme-level** text.

![mechanism](promote/mechanism.png)

In **Praasper**, we adopt a rather simple and straightforward pipeline to extract phoneme-level information from audio files. The pipeline includes [Whisper](https://github.com/openai/whisper) and [Praditor](https://github.com/Paradeluxe/Praditor). 

Now **Praasper** support **Mandarin**. In the near future we plan to add support for **Cantonese** and **English**.



# How to use

The default model is `large-v3-turbo`.

>I personally recommend to use the SOTA model as time isn't a really big problem for offline processing.

```python
import praasper

model = praasper.init_model(model_name="large-v3-turbo")  
model.annote(input_path="data")  # The folder where you store .wav

# If you want to know what other models are available:

# import whisper
# print(whisper.available_models())
```


# Mechanism

**Whisper** is used to transcribe the audio file to **word-level text**. At this point, speech onsets and offsets exhibit time deviations in seconds.

**Praditor** is applied to perform **Voice Activity Detection (VAD)** algorithm to trim the currently existing word/character-level timestamps to **millisecond level**. It is a Speech Onset Detection (SOT) algorithm we developed for langauge researchers.

To extract phoneme boundaries, we designed an **edge detection algorithm**. 
- The audio file is first resampled to **16 kHz** as to remove noise in the high-frequency domain. 
- A kernel,`[-1, 0, 1]`, is then applied to the frequency domain to enhance the edge(s) between phonetic segments.
- The most prominent **n** peaks are then selected so as to match the wanted number of phonemes.

# Setup
## pip installation

```bash
pip install -U praasper
```
> If you have a succesful installation and don't care if there is GPU accelaration, you can stop it right here.


## GPU Acceleration (Windows/Linux)
`Whisper` can automaticly detects the best currently available device to use. But you still need to first install GPU-support version `torch` in order to enable CUDA acceleration.

- For **macOS** users, `Whisper` only detects CPU as the processing device.
- For **Windows/Linux** users, the priority order should be: `CUDA` -> `CPU`.

If you have no experience in installing `CUDA`, follow the steps below:



**First**, go to command line and check the latest CUDA version your system supports:

```bash
nvidia-smi
```

Results should pop up like this:
```bash
| NVIDIA-SMI 576.80                 Driver Version: 576.80         CUDA Version: 12.9     |
```
It means that this device supports CUDA up to version 12.9.

**Next**, go to [**NVIDIA CUDA Toolkit**](https://developer.nvidia.com/cuda-toolkit) and download the latest version, or whichever version that fits your system/need.

**Next**, uninstall the default CPU-only torch:

```bash
pip uninstall torch
```

**Lastly**, install `torch` that fits your CUDA version. Find the correct `pip` command [**in this link**](https://pytorch.org/get-started/locally/).

Here is an example for CUDA 12.9:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu129
```


## (Advanced) uv installation
`uv` is also highly recommended for way **FASTER** installation. First, make sure `uv` is installed to your default environment:

```bash
pip install uv
```

Then, create a virtual environment (e.g., .venv):

```bash
uv venv .venv
```

You should see a new `.venv` folder pops up in your project folder now. (You might also want to restart the terminal.)

Lastly, install `praasper` (by adding `uv` before `pip`):


```bash
uv pip install praasper
```
For `CUDA` support,

```bash
uv pip install --reinstall torch --index-url https://download.pytorch.org/whl/cu129
# Or whichever version that matches your CUDA version
```