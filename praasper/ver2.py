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
        whisper_model = whisper.load_model("large-v3-turbo", device="cuda:0")

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
            for start, end in segment_audio(audio_obj, segment_duration=seg_dur):
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
                
                min_noise_dur = 2.0
                intervals = vad_tg.tiers[0].intervals


                min_power = float('inf')  # 初始化最小功率为正无穷大

                for idx, interval in enumerate(intervals):
                    if interval.mark not in ["", None]:
                        continue
                    dur = interval.maxTime - interval.minTime
                    power = audio_clip[interval.minTime*1000:interval.maxTime*1000].power()
                    if power < min_power:  # 如果当前功率小于最小功率
                        min_power = power  # 更新最小功率
                        target_noise_idx = idx  # 更新目标噪声区间的索引
                target_noise_interval = intervals[target_noise_idx]
                dur = target_noise_interval.maxTime - target_noise_interval.minTime
                target_noise = audio_clip[(target_noise_interval.minTime+dur * 0.1)*1000:(target_noise_interval.maxTime-dur * 0.1)*1000]

                _count = int(min_noise_dur / target_noise.duration_seconds)
                for i in range(_count):
                    if i // 2 != 0:
                        target_noise += target_noise.reverse()
                    else:
                        target_noise += target_noise


                cand_audios = []
                good_intervals = []
                last_start_time = 0.
                for idx, interval in enumerate(intervals):
                    if interval.mark in ["", None]:
                        continue

                    # pre_constraint = (intervals[idx-1].maxTime - intervals[idx-1].minTime)
                    # post_constraint = (intervals[idx+1].maxTime - intervals[idx+1].minTime)
                    dur = target_noise.duration_seconds

                    start_clip = interval.minTime
                    end_clip = interval.maxTime

                    audio_interval = audio_clip[(start_clip - dur * 0.2)*1000:(end_clip + dur * 0.2)*1000]

                    # 处理 audio_pre
                    audio_pre = audio_clip[(start_clip - pre_constraint * 0.8)*1000:(start_clip - pre_constraint * 0.2)*1000]
                    _count = int(min_noise_dur / audio_pre.duration_seconds)
                    extra_pre = audio_pre
                    for i in range(_count):
                        if i // 2 != 0:
                            extra_pre = audio_pre + extra_pre
                        else:
                            extra_pre = audio_pre.reverse() + extra_pre
                    audio_pre = extra_pre
                    
                    # 处理 audio_post
                    audio_post = audio_clip[(end_clip + post_constraint * 0.2)*1000:(end_clip + post_constraint*0.8)*1000]
                    _count = int(min_noise_dur / audio_post.duration_seconds)
                    extra_post = audio_post
                    for i in range(_count):
                        if i // 2 != 0:
                            extra_post += audio_post
                        else:
                            extra_post += audio_post.reverse()
                    audio_post = extra_post
                    # print(audio_pre.duration_seconds, audio_interval.duration_seconds, audio_post.duration_seconds)

                    good_intervals.append([last_start_time + audio_pre.duration_seconds, last_start_time + audio_pre.duration_seconds + audio_interval.duration_seconds])
                    # 拼接 audio_interval 和 audio_interval
                    audio_interval = audio_pre + audio_interval + audio_post
                    last_start_time += audio_interval.duration_seconds
                    cand_audios.append(audio_interval)
                print(good_intervals)
                print("-")

                batch_size = 4
                for i in range(0, len(cand_audios), batch_size):
                    batch_audios = cand_audios[i:i+batch_size]

                    batch_audio = batch_audios[0]
                    for audio in batch_audios[1:]:
                        batch_audio += audio

                    interval_path = os.path.join(tmp_path, os.path.basename(clip_path).replace(".wav", f"_batch_{i}.wav"))
                    batch_audio.save(interval_path)


                    result = whisper_model.transcribe(interval_path, temperature=[0.0], fp16=torch.cuda.is_available())#, word_timestamps=True)

                    good_indices = []
                    for idx_intval, intval in enumerate(good_intervals):
                        if any([has_time_overlap(segment["start"], segment["end"], intval[0], intval[1]) for segment in result["segments"]]):
                            good_indices.append(idx_intval)
                            

                
                    print("-")
                    intervals = [interval for interval in intervals if interval.mark == "+"]
                    intervals = [intervals[idx] for idx in good_indices]
                    print(intervals)
                    print(result["segments"])
                    print("-")
                    for idx_seg, segment in enumerate(result["segments"]):
                        text = purify_text(segment["text"])
                        final_tg.tiers[0].addInterval(Interval(intervals[idx_seg].minTime + start/1000, intervals[idx_seg].maxTime + start/1000, text))

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
                
                # exit()
                
            final_tg.write(final_path)
        
        # shutil.rmtree(tmp_path)
        print(f"--------------- Processing completed ---------------")


if __name__ == "__main__":
    model = init_model(model_name="large-v3-turbo")
    model.annote(
        input_path=os.path.abspath("input_short"),
        sr=12000,
        seg_dur=20.,
        merge_words=True,
        language=None,
        verbose=False
    )

    # [(0.0, 0.65994), (18.63519, 18.91781)]