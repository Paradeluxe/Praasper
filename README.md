<h1 align="center">Praasper</h1>


**Praasper** is an Automatic Speech Recognition (ASR) application that can help researchers transribe audio files to both word- and phoneme-level text.

| Precision | Completed  | Developing  |
| - | - | - |
| word-level  | Mandarin  |  Cantonese, English |
|  phoneme-level |  Mandarin |  Cantonese, English |


In **Praasper**, we adopt a rather simple and straightforward pipeline to extract phoneme-level information from audio files.

1. [**Whisper**](https://github.com/openai/whisper) is used to transcribe the audio file to word-level text.

```Python
model = whisper.load_model("large-v3-turbo", device="cuda")
result = model.transcribe(wav, word_timestamps=True)
```

2. [**Praditor**](https://github.com/Paradeluxe/Praditor) is used to apply Voice Activity Detection (VAD) algorithm to trim the currently existing word/character-level timestamps.
