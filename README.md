<h1 align="center">Praasper</h1>
<p align="center">

  <!-- <a href="https://github.com/m-bain/whisperX/stargazers">
    <img src="https://img.shields.io/github/stars/m-bain/whisperX.svg?colorA=orange&colorB=orange&logo=github"
         alt="GitHub stars">
  </a> -->
  <!-- <a href="https://github.com/m-bain/whisperX/issues">
        <img src="https://img.shields.io/github/issues/m-bain/whisperx.svg"
             alt="GitHub issues">
  </a> -->
  <!-- <a href="https://github.com/m-bain/whisperX/blob/master/LICENSE">
        <img src="https://img.shields.io/github/license/m-bain/whisperX.svg"
             alt="GitHub license">
  </a>
    -->
</p>


**Praasper** is an Automatic Speech Recognition (ASR) application designed help researchers transribe audio files to both word- and phoneme-level text.

![mechanism](promote/mechanism.png)



# Mechanism
In **Praasper**, we adopt a rather simple and straightforward pipeline to extract phoneme-level information from audio files.

[**Whisper**](https://github.com/openai/whisper) is used to transcribe the audio file to **word-level text**. At this point, speech onsets and offsets exhibit time deviations in seconds.

```Python
model = whisper.load_model("large-v3-turbo", device="cuda")
result = model.transcribe(wav, word_timestamps=True)
```

[**Praditor**](https://github.com/Paradeluxe/Praditor) is applied to perform **Voice Activity Detection (VAD)** algorithm to trim the currently existing word/character-level timestamps (at millisecond level). It is a Speech Onset Detection (SOT) algorithm we developed for langauge researchers.

To extract phoneme boundaries, we designed an **Edge detection algorithm**. 
- The audio file is first down-sampled to 16 kHz as to remove noise in the high-frequency domain. 
- A kernel,`[-1, 0, 1]`, is then applied to the frequency domain to remove low-contrast frequency bands.

# Support

| Precision | Completed  | Developing  |
| :---: | :---: | :---: |
| Word  | Mandarin  |  Cantonese, English |
|  Phoneme |  Mandarin |  Cantonese, English |

# Setup
We provide multiple ways to setup different versions of **Praasper**.