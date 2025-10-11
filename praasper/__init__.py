try:
    from .utils import *
    from .process import *
    from .select_word import *
    from .post_process import *

except ImportError:
    from utils import *
    from process import *
    from select_word import *
    from post_process import *

import os
import shutil

class init_model:

    def __init__(self, model_name: str="large-v3-turbo"):

        self.name = model_name

        # 注意：在这个版本中，我们使用的是 SenseVoiceSmall 模型而不是 Whisper
        # 所以不需要检查 Whisper 模型的可用性
        print(f"[{show_elapsed_time()}] Initializing model with {self.name}")
        
        # 由于使用的是 SenseVoiceSmall 模型，不需要加载 Whisper 模型
        # self.whisper_model = None
        print(f"[{show_elapsed_time()}] Model initialized successfully")

    def annote(
        self,
        input_path: str,
        sr=None,
        seg_dur=10.,
        min_speech=0.2,
        language=None,
        verbose: bool=False
    ):
        # whisper_model = whisper.load_model("large-v3-turbo", device="cuda:0")

        fnames = [os.path.splitext(f)[0] for f in os.listdir(input_path) if f.endswith('.wav')]
        print(f"[{show_elapsed_time()}] {len(fnames)} valid audio files detected in {input_path}")

        if not fnames:
            return

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
            segments = segment_audio(audio_obj, segment_duration=seg_dur)
            for start, end in segments:
                count += 1

                print(f"[{show_elapsed_time()}] Processing segment: {start/1000:.3f} - {end/1000:.3f} ({count})")
                audio_clip = audio_obj[start:end]
                clip_path = os.path.join(tmp_path, os.path.basename(wav_path).replace(".wav", f"_{count}.wav"))
                audio_clip.save(clip_path)


                try:
                    vad_tg = get_vad(clip_path, wav_path, verbose=verbose)
                except Exception as e:
                    print(f"[{show_elapsed_time()}] ({os.path.basename(clip_path)}) VAD Error: {e}")
                    continue
                
                intervals = vad_tg.tiers[0].intervals
                valid_intervals = [interval for interval in intervals if interval.mark not in ["", None] and interval.maxTime - interval.minTime > min_speech]
                # print(valid_intervals)
                
                for idx, valid_interval in enumerate(valid_intervals):
                    s, e = valid_interval.minTime, valid_interval.maxTime
                    print(f"[{show_elapsed_time()}] ({os.path.basename(clip_path)}) VAD segment: {s:.3f} - {e:.3f}")

                    interval_path = os.path.join(tmp_path, os.path.basename(clip_path).replace(".wav", f"_{idx}.wav"))
                    audio_clip[s*1000:e*1000].save(interval_path)
                    text = get_text_from_audio(interval_path)

                    text = purify_text(text)
                    if not text:
                        continue
                    if not is_single_language(text):
                        text = post_process(text, language)

                    final_tg.tiers[0].add(s+start/1000, e+start/1000, text)



            final_tg.write(final_path)
                
                        
            # if os.path.exists(clip_path):
            #     os.remove(clip_path)
                
                # exit()
                
        
        shutil.rmtree(tmp_path)
        print(f"--------------- Processing completed ---------------")


if __name__ == "__main__":
    model = init_model(model_name="large-v3-turbo")
    model.annote(
        input_path=os.path.abspath("input"),
        sr=12000,
        seg_dur=20.,
        language="zh",
        verbose=False
    )

    # [(0.0, 0.65994), (18.63519, 18.91781)]