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
import shutil

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

    def annote(self, input_path: str, sr=None, seg_dur=10., language=None, verbose: bool=False):

        fnames = [os.path.splitext(f)[0] for f in os.listdir(input_path) if f.endswith('.wav')]
        print(f"[{show_elapsed_time()}] {len(fnames)} valid audio files detected in {input_path}")

        for idx, fname in enumerate(fnames):
            wav_path = os.path.join(input_path, fname + ".wav")

            dir_name = os.path.dirname(os.path.dirname(wav_path))

            tmp_path = os.path.join(dir_name, "tmp")
            if os.path.exists(tmp_path):
                shutil.rmtree(tmp_path)
                print(f"[{show_elapsed_time()}] Temporary directory {tmp_path} removed.")

            os.makedirs(tmp_path, exist_ok=False)

            output_path = os.path.join(dir_name, "output")
            os.makedirs(output_path, exist_ok=True)
            
            final_path = os.path.join(output_path, os.path.basename(wav_path).replace(".wav", ".TextGrid"))

            audio_obj = ReadSound(wav_path)

            final_tg = TextGrid()
            final_tg.tiers.append(IntervalTier(name="words", minTime=0., maxTime=audio_obj.duration_seconds))

            print(f"--------------- Processing {os.path.basename(wav_path)} ({idx+1}/{len(fnames)}) ---------------")
            count = 0
            for start, end in segment_audio(audio_obj, segment_duration=seg_dur):
                count += 1
                print(f"[{show_elapsed_time()}] Processing segment: {start/1000:.3f} - {end/1000:.3f} ({count})")
                audio_clip = audio_obj[start:end]
                clip_path = os.path.join(tmp_path, os.path.basename(wav_path).replace(".wav", f"_{count}.wav"))
                audio_clip.save(clip_path)


                try:
                    vad_tg = get_vad(clip_path, verbose=verbose)
                except Exception as e:
                    print(f"[{show_elapsed_time()}] Error processing {os.path.basename(wav_path)}: {e}")
                    os.remove(clip_path)
                    continue
                try:
                    language, tg = transcribe_wav_file(clip_path, vad=vad_tg, whisper_model=self.whisper_model, language=language)
                except Exception as e:
                    print(f"[{show_elapsed_time()}] Error transcribing {os.path.basename(wav_path)}: {e}")
                    os.remove(clip_path)
                    continue
                # print(tg.tiers[0].intervals)
                try:
                # print(whisper_tg.tiers[0].intervals)
                    tg = find_word_boundary(clip_path, tg, tar_sr=sr, verbose=verbose)
                except Exception as e:
                    print(f"[{show_elapsed_time()}] Error finding word boundary {os.path.basename(wav_path)}: {e}")
                    os.remove(clip_path)

                for interval in tg.tiers[0].intervals:
                    final_tg.tiers[0].add(interval.minTime + start/1000, interval.maxTime + start/1000, interval.mark)
                if os.path.exists(clip_path):
                    os.remove(clip_path)
                
            final_tg.write(final_path)
            # if final_tg:
                # final_tg.write(final_path)
        
        shutil.rmtree(tmp_path)
        print(f"--------------- Processing completed ---------------")


if __name__ == "__main__":
    model = init_model(model_name="large-v3-turbo")
    model.annote(input_path=os.path.abspath("data"), sr=12000, seg_dur=3.5, language=None, verbose=False)