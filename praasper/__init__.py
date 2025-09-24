try:
    from .utils import *
    from .process import *
    from .word_boundary import *
except ImportError:
    from utils import *
    from process import *
    from word_boundary import *

import os
import whisper
import torch

class init_model:

    def __init__(self, model_name: str="large-v3-turbo"):

        self.name = model_name

        available_models = whisper.available_models()
        if self.name in available_models:
            print(f"[{show_elapsed_time()}] Loading Whisper model: {self.name}")
        else:
            raise ValueError(f"[{show_elapsed_time()}] Model {self.name} is not in the available Whisper models. Available models are: {available_models}")
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.whisper_model = whisper.load_model(self.name, device=device)
        print(f"[{show_elapsed_time()}] Model loaded successfully. Current device in use: {self.whisper_model.device if hasattr(self.whisper_model, 'device') else 'Unknown'}")

    def annote(self, input_path: str, language=None, verbose: bool=False):

        fnames = [os.path.splitext(f)[0] for f in os.listdir(input_path) if f.endswith('.wav')]
        print(f"[{show_elapsed_time()}] {len(fnames)} valid audio files detected in {input_path}")

        for idx, fname in enumerate(fnames):
            wav_path = os.path.join(input_path, fname + ".wav")
            tg_path = wav_path.replace(".wav", "_whisper.TextGrid")
            vad_path = wav_path.replace(".wav", "_VAD.TextGrid")

            print(f"[{show_elapsed_time()}] Processing {os.path.basename(wav_path)} ({idx+1}/{len(fnames)})")

            get_vad(wav_path)
            language = transcribe_wav_file(wav_path, vad=vad_path, whisper_model=self.whisper_model, language=language)
            # word_timestamp(wav_path, tg_path, language=language)
            plot_audio_power_curve(wav_path, tar_sr=16000, verbose=verbose)
        
        print(f"[{show_elapsed_time()}] Processing completed.")


if __name__ == "__main__":
    model = init_model(model_name="large-v3-turbo")
    model.annote(input_path=os.path.abspath("input"), language=None, verbose=False)