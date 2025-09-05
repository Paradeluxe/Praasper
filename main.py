from praasper.process import *


data_path = os.path.abspath("data")
# input_dir = os.path.abspath("input")
# output_dir = os.path.abspath("output")

fnames = [os.path.splitext(f)[0] for f in os.listdir(data_path) if f.endswith('.wav')]


for fname in fnames:
    wav_path = os.path.join(data_path, fname + ".wav")
    tg_path = wav_path.replace(".wav", "_whisper.TextGrid")
    vad_path = wav_path.replace(".wav", "_VAD.TextGrid")

    transcribe_wav_file(wav_path, vad=vad_path)
    word_timestamp(wav_path, tg_path)