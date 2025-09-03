# Praasper
![Python](https://img.shields.io/badge/python-3.10-blue.svg)
![GitHub License](https://img.shields.io/github/license/Paradeluxe/Praasper)


**Praasper** is an Automatic Speech Recognition (ASR) application designed help researchers transribe audio files to both **word-** and **phoneme-level** text.

![mechanism](promote/mechanism.png)



# Mechanism
In **Praasper**, we adopt a rather simple and straightforward pipeline to extract phoneme-level information from audio files.

**Whisper** ([repo](https://github.com/openai/whisper)) is used to transcribe the audio file to **word-level text**. At this point, speech onsets and offsets exhibit time deviations in seconds.

```Python
model = whisper.load_model("large-v3-turbo", device="cuda")
result = model.transcribe(wav, word_timestamps=True)
```

[**Praditor**](https://github.com/Paradeluxe/Praditor) is applied to perform **Voice Activity Detection (VAD)** algorithm to trim the currently existing word/character-level timestamps (at millisecond level). It is a Speech Onset Detection (SOT) algorithm we developed for langauge researchers.

To extract phoneme boundaries, we designed an **edge detection algorithm**. 
- The audio file is first resampled to **16 kHz** as to remove noise in the high-frequency domain. 
- A kernel,`[-1, 0, 1]`, is then applied to the frequency domain to enhance the edge(s) between phonetic segments.
- The most prominent **n** peaks are then selected so as to match the wanted number of phonemes.

# Support

| Precision | Completed  | Developing  |
| :---: | :---: | :---: |
| Word  | Mandarin  |  Cantonese, English |
|  Phoneme |  Mandarin |  Cantonese, English |

# Setup
We provide multiple ways to setup different versions of **Praasper**.