<h1 align="center">Praasper</h1>


Praasper is an Automatic Speech Recognition (ASR) applications that can help researchers transribe audio files to both word- and phoneme-level text.

In Praasper, we adopt a rather simple-and-straightforward pipeline to extract phoneme-level information from audio files.

1. First, we use [Whisper](https://github.com/openai/whisper) to transcribe the audio file to word-level text.

```Python
model = whisper.load_model("large-v3-turbo", device="cuda")
result = model.transcribe(wav, word_timestamps=True)
```
