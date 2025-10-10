try:
    from .utils import *
    from .process import *
    from .word_boundary import *
    from .select_word import *
except ImportError:
    from utils import *
    from process import *
    from word_boundary import *
    from select_word import *

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
            segments = segment_audio(audio_obj, segment_duration=seg_dur)
            print(segments)
            for start, end in segments:
                count += 1

                print(f"[{show_elapsed_time()}] Processing segment: {start/1000:.3f} - {end/1000:.3f} ({count})")
                audio_clip = audio_obj[start:end]
                clip_path = os.path.join(tmp_path, os.path.basename(wav_path).replace(".wav", f"_{count}.wav"))
                audio_clip.save(clip_path)

                # try:
                vad_tg = get_vad(clip_path, wav_path, verbose=verbose)
                # except Exception as e:
                    # print(f"[{show_elapsed_time()}] ({os.path.basename(clip_path)}) VAD Error: {e}")
                    # continue
                
                intervals = vad_tg.tiers[0].intervals
                valid_intervals = [interval for interval in intervals if interval.mark not in ["", None]]
                print(valid_intervals)
                min_noise_dur = 1.5
                
                target_noise = audio_clip.min_power_segment(.5)
                target_noise.save(os.path.join(tmp_path, os.path.basename(wav_path).replace(".wav", f"_{count}_noise.wav")))

                print(f"Target noise duration: {target_noise.duration_seconds:.3f}")

                _count = int(min_noise_dur / target_noise.duration_seconds)
                for i in range(_count):
                    if i // 2 != 0:
                        target_noise += target_noise
                    else:
                        target_noise += target_noise.reverse()


                cand_audios = []
                last_start_time = 0.
                for idx, interval in enumerate(intervals):
                    if interval.mark in ["", None]:
                        continue

                    start_clip = interval.minTime
                    end_clip = interval.maxTime

                    audio_interval = audio_clip[(start_clip - 0.1)*1000:(end_clip + 0.1)*1000]

                    audio_interval = target_noise + audio_interval + target_noise
                    


                    last_start_time += audio_interval.duration_seconds
                    cand_audios.append(audio_interval)

                batch_size = 3
                for i in range(0, len(cand_audios), batch_size):
                    good_intervals = []
                    print("-")
                    print(i)
                    batch_audios = cand_audios[i:i+batch_size]
                    print(batch_audios)

                    good_intervals.append([0.0 + target_noise.duration_seconds + 0.1, 0.0 + batch_audios[0].duration_seconds - target_noise.duration_seconds - 0.1])
                    startup_time = 0.0
                    batch_audio = batch_audios[0]
                    for audio in batch_audios[1:]:
                        startup_time += audio.duration_seconds
                        print(audio.duration_seconds)
                        batch_audio += audio
                        good_intervals.append([startup_time + target_noise.duration_seconds + 0.1, startup_time + audio.duration_seconds - target_noise.duration_seconds - 0.1])
                    print("", batch_audio.duration_seconds)
                    print(good_intervals)

                    interval_path = os.path.join(tmp_path, os.path.basename(clip_path).replace(".wav", f"_batch_{i}.wav"))
                    batch_audio.save(interval_path)


                    result = whisper_model.transcribe(interval_path, temperature=[0.0], fp16=torch.cuda.is_available())#, word_timestamps=True)
                    lang = result["language"]
                    print(result)
                    print(good_intervals)

                    aff_dict = {}
                    for idx_intval, intval in enumerate(good_intervals):
                        print(intval)
                        # 以result["segments"]为索引，vad区间所属最大的segment（单个）
                        time_overlaps = [has_time_overlap(segment["start"], segment["end"], intval[0], intval[1]) for segment in result["segments"]]
                        print(time_overlaps)
                        if any(time_overlap != 0 for time_overlap in time_overlaps):
                            # 找到最大的 time_overlaps 值对应的索引
                            max_overlap_idx = time_overlaps.index(max(time_overlaps))
                            aff_dict.setdefault(result["segments"][max_overlap_idx]["text"], []).append(i+idx_intval)
                                
                    print(aff_dict)
                    # continue
                    for idx_seg, seg in enumerate(aff_dict):
                        if len(aff_dict[seg]) > 1:
                            seg_results = []
                            for idx_intval in aff_dict[seg]:
                                seg_audio = batch_audios[idx_intval]
                                seg_path = os.path.join(tmp_path, os.path.basename(clip_path).replace(".wav", f"_batch_{i}_interval_{idx_intval}.wav"))
                                seg_audio.save(seg_path)
                                
                                # initial_prompt = f"如果所识别内容和“{result["segments"][idx_seg]["text"]}”相似，请直接输出“{result["segments"][idx_seg]["text"]}”；否则，请输出空字符串"

                                seg_result = whisper_model.transcribe(seg_path, temperature=[0.0], fp16=torch.cuda.is_available(), language=lang)
                                # print(seg_result)
                                seg_results.append(purify_text(seg_result["text"]))
                            
                            text = purify_text(result["segments"][idx_seg]["text"])
                            answer = ""
                            retry_count = 0
                            while answer not in seg_results:
                                text1, text2 = seg_results
                                answer = which_is_closer(text1, text2, text)
                                retry_count += 1
                                if retry_count > 5:
                                    break
                            if answer in [text1, text2]:
                                aff_dict[seg] = [aff_dict[seg][[text1, text2].index(answer)]]
                                # break
                            else:
                                aff_dict[seg] = []
                    print(aff_dict)
                
                    for text, indices_time in aff_dict.items():
                        if not indices_time:
                            continue
                        interval = valid_intervals[indices_time[0]]
                        print(interval)
                        s, e = interval.minTime, interval.maxTime
                        s += start/1000
                        e += start/1000

                        final_tg.tiers[0].add(s, e, text)

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
        input_path=os.path.abspath("input_single"),
        sr=12000,
        seg_dur=15.,
        merge_words=True,
        language=None,
        verbose=False
    )

    # [(0.0, 0.65994), (18.63519, 18.91781)]