<h1 align="center">Praasper</h1>


**Praasper** is an Automatic Speech Recognition (ASR) application designed help researchers transribe audio files to both word- and phoneme-level text.


| Precision | Completed  | Developing  |
| :---: | :---: | :---: |
| Word  | Mandarin  |  Cantonese, English |
|  Phoneme |  Mandarin |  Cantonese, English |

# Mechanism
In **Praasper**, we adopt a rather simple and straightforward pipeline to extract phoneme-level information from audio files.

1. [**Whisper**](https://github.com/openai/whisper) is used to transcribe the audio file to **word-level text**. At this point, speech onsets and offsets exhibit time deviations in seconds.

```Python
model = whisper.load_model("large-v3-turbo", device="cuda")
result = model.transcribe(wav, word_timestamps=True)
```

2. [**Praditor**](https://github.com/Paradeluxe/Praditor) is applied to perform **Voice Activity Detection (VAD)** algorithm to trim the currently existing word/character-level timestamps (at millisecond level). It is a Speech Onset Detection (SOT) algorithm we developed for langauge researchers.

3. To extract phoneme boundaries, we designed an **Edge detection algorithm**. The audio file is first down-sampled to 16 kHz as to remove noise in the high-frequency domain. A kernel,`[-1, 0, 1]`, is then applied to the frequency domain to increase the contrast between phonemes (rather than within-phonemes).


# Setup
We provide multiple ways to setup different versions of **Praasper**.