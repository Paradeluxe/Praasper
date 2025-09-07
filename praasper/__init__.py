try:
    from .process import *
    from .tool import *
except ImportError:
    from process import *
    from tool import *

import os
import whisper

class init_model:

    def __init__(self, model_name: str="large-v3-turbo"):

        self.name = model_name

        available_models = whisper.available_models()
        if self.name in available_models:
            print(f"[{show_elapsed_time()}] Choose Whisper model: {self.name}")
        else:
            raise ValueError(f"[{show_elapsed_time()}] Model {self.name} is not in the available Whisper models. Available models are: {available_models}")


    def annote(self, input_path: str):

        fnames = [os.path.splitext(f)[0] for f in os.listdir(input_path) if f.endswith('.wav')]

        for fname in fnames:
            wav_path = os.path.join(input_path, fname + ".wav")
            tg_path = wav_path.replace(".wav", "_whisper.TextGrid")
            vad_path = wav_path.replace(".wav", "_VAD.TextGrid")


            get_vad(wav_path)
            transcribe_wav_file(wav_path, vad=vad_path, model_name=self.name)
            word_timestamp(wav_path, tg_path)


if __name__ == "__main__":
    model = init_model(model_name="large-v3-turbo")
    model.annote(input_path=os.path.abspath("data"))