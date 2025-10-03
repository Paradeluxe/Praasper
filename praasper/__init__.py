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

    def annote(
        self,
        input_path: str,
        sr=None,
        seg_dur=10.,
        merge_words: bool=False,
        language=None,
        verbose: bool=False
    ):

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


                vad_tg = get_vad(clip_path, wav_path, verbose=verbose)
                # print(vad_tg.tiers[0].intervals)

                language, tg = transcribe_wav_file(clip_path, vad=vad_tg, whisper_model=self.whisper_model, language=language)
                # print(tg.tiers[0].intervals)

                if language in ["zh"]:
                    print(f"[{show_elapsed_time()}] ({os.path.basename(clip_path)}) (beta) Start word boundary detection...")
                    tg = find_word_boundary(clip_path, tg, tar_sr=sr, verbose=verbose)
                else:
                    print(f"[{show_elapsed_time()}] ({os.path.basename(clip_path)}) Language {language} is currently not supported for word boundary detection.")
                # print(tg.tiers[0].intervals)


                for interval in tg.tiers[0].intervals:
                    try:
                        final_tg.tiers[0].add(interval.minTime + start/1000, interval.maxTime + start/1000, interval.mark)
                    except ValueError:  # 浮点数精度问题
                        # print(f"精度问题 {final_tg.tiers[0].intervals[-1].maxTime} {interval.minTime + start/1000}")
                        final_tg.tiers[0].add(final_tg.tiers[0].intervals[-1].maxTime, interval.maxTime + start/1000, interval.mark)
                    
                if merge_words:
                    # 合并相邻的 interval
                    tier = final_tg.tiers[0]
                    i = 0
                    while i < len(tier.intervals) - 1:
                        current_interval = tier.intervals[i]
                        next_interval = tier.intervals[i + 1]
                        if abs(current_interval.maxTime - next_interval.minTime) < 1e-6:  # 考虑浮点数精度问题
                            # 合并相邻的 interval
                            new_interval = Interval(
                                minTime=current_interval.minTime,
                                maxTime=next_interval.maxTime,
                                mark=current_interval.mark + next_interval.mark
                            )
                            # 移除原有的两个 interval
                            tier.intervals.pop(i)
                            tier.intervals.pop(i)
                            # 插入合并后的 interval
                            tier.intervals.insert(i, new_interval)
                        else:
                            i += 1

                
                        
                if os.path.exists(clip_path):
                    os.remove(clip_path)
                
            final_tg.write(final_path)
        
        shutil.rmtree(tmp_path)
        print(f"--------------- Processing completed ---------------")


if __name__ == "__main__":
    model = init_model(model_name="large-v3-turbo")
    model.annote(
        input_path=os.path.abspath("big_data"),
        sr=12000,
        seg_dur=15.,
        merge_words=True,
        language=None,
        verbose=False
    )