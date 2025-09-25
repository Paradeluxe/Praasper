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

    def annote(self, input_path: str, sr=None, language=None, verbose: bool=False):

        fnames = [os.path.splitext(f)[0] for f in os.listdir(input_path) if f.endswith('.wav')]
        print(f"[{show_elapsed_time()}] {len(fnames)} valid audio files detected in {input_path}")

        for idx, fname in enumerate(fnames):
            wav_path = os.path.join(input_path, fname + ".wav")
            tg_path = wav_path.replace(".wav", "_whisper.TextGrid")
            vad_path = wav_path.replace(".wav", "_VAD.TextGrid")

            output_path = os.path.join(os.path.dirname(os.path.dirname(tg_path)), "output")
            os.makedirs(output_path, exist_ok=True)
            final_path = os.path.join(output_path, os.path.basename(wav_path).replace(".wav", ".TextGrid"))

            final_tg = None
            print(f"--------------- Processing {os.path.basename(wav_path)} ({idx+1}/{len(fnames)}) ---------------")
            try:
                get_vad(wav_path, verbose=verbose)
            except Exception as e:
                print(f"[{show_elapsed_time()}] Error processing {os.path.basename(wav_path)}: {e}")
                continue
            try:
                language, final_tg = transcribe_wav_file(wav_path, vad=vad_path, whisper_model=self.whisper_model, language=language)
            except Exception as e:
                print(f"[{show_elapsed_time()}] Error transcribing {os.path.basename(wav_path)}: {e}")
                continue
            try:
                final_tg = find_word_boundary(wav_path, tar_sr=sr, verbose=verbose)
            except Exception as e:
                print(f"[{show_elapsed_time()}] Error finding word boundary {os.path.basename(wav_path)}: {e}")
            if final_tg:
                final_tg.write(final_path)
        
        print(f"--------------- Processing completed ---------------")


if __name__ == "__main__":
    model = init_model(model_name="large-v3-turbo")
    model.annote(input_path=os.path.abspath("data"), sr=12000, language=None, verbose=False)