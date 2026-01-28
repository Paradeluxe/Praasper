import os
import shutil

try:
    from .utils import *
    from .process import *
    from .select_word import *
    from .post_process import *

except ImportError:
    from praasper.utils import *
    from praasper.process import *
    from praasper.select_word import *
    from praasper.post_process import *

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'  # 设置镜像源


class init_model:

    def __init__(
        self,
        ASR: str="iic/SenseVoiceSmall",
        infer_mode: str = "funasr",
        LLM: str="Qwen/Qwen2.5-1.5B-Instruct",
        device: str= "cpu"
    ):

        # 验证 infer_mode 参数值
        allowed_modes = ["direct", "funasr"]
        if infer_mode not in allowed_modes:
            raise ValueError(f"infer_mode must be one of {allowed_modes}, got {infer_mode}")
        
        self.ASR = ASR
        self.infer_mode = infer_mode
        self.LLM = LLM
        self.device = device
        print(f"[{show_elapsed_time()}] Initializing model with {self.ASR}")

        init_LLM(self.LLM)

        self.model = SelectWord(
            model=self.ASR,
            infer_mode=self.infer_mode,
            device=self.device
        )
        print(f"[{show_elapsed_time()}] Using device: {self.model.device}")
        

    def annote(
        self,
        input_path: str,
        seg_dur=10.,
        min_speech=0.2,
        min_pause=0.2,
        language=None,
        verbose: bool=False,
        skip_existing: bool=False,
        enable_post_process: bool=True
    ):
        if os.path.isdir(input_path):
            fnames = [os.path.splitext(f)[0] for f in os.listdir(input_path) if f.endswith('.wav')]
            print(f"[{show_elapsed_time()}] {len(fnames)} valid audio files detected in {input_path}")

        else:
            fnames = [os.path.splitext(os.path.basename(input_path))[0]]
            input_path = os.path.dirname(input_path)
            print(f"[{show_elapsed_time()}] {fnames[0]} is detected in {input_path}")


        if not fnames:
            return

        for idx, fname in enumerate(fnames):
            wav_path = os.path.join(input_path, fname + ".wav")

            dir_name = os.path.dirname(os.path.dirname(wav_path))
            tmp_path = os.path.join(dir_name, "tmp")
            # 获取输入文件夹的名称
            input_folder_name = os.path.basename(input_path)
            # 在output目录下创建与输入文件夹同名的子目录
            output_path = os.path.join(dir_name, "output", input_folder_name)
            final_path = os.path.join(output_path, os.path.basename(wav_path).replace(".wav", ".TextGrid"))

            # 检查结果文件是否已存在，如果存在且skip_existing为True则跳过处理
            if skip_existing and os.path.exists(final_path):
                print(f"[{show_elapsed_time()}] Skipping {os.path.basename(wav_path)} (result exists)")
                continue

            try:
                # 尝试加载音频文件
                audio_obj = ReadSound(wav_path)
            except Exception as e:
                print(f"[{show_elapsed_time()}] Error loading audio file {wav_path}: {str(e)}")
                continue

            # 仅在音频加载成功后创建临时目录（很正确）
            if os.path.exists(tmp_path):
                shutil.rmtree(tmp_path)
                print(f"[{show_elapsed_time()}] Temporary directory {tmp_path} removed.")
            os.makedirs(tmp_path, exist_ok=False)

            os.makedirs(output_path, exist_ok=True)


            final_tg = TextGrid()
            final_tg.tiers.append(IntervalTier(name="words", minTime=0., maxTime=audio_obj.duration_seconds))
            final_tg.tiers[0].strict = False

            print(f"--------------- Processing {os.path.basename(wav_path)} ({idx+1}/{len(fnames)}) ---------------")
            count = 0
            segments = segment_audio(audio_obj, segment_duration=seg_dur, params="folder", min_pause=min_pause)
            for start, end in segments:
                count += 1

                print(f"[{show_elapsed_time()}] Processing segment: {start/1000:.3f} - {end/1000:.3f} ({count})")
                audio_clip = audio_obj[start:end]
                clip_path = os.path.join(tmp_path, os.path.basename(wav_path).replace(".wav", f"_{count}.wav"))
                audio_clip.save(clip_path)


                try:
                    vad_tg = get_vad(clip_path, params="folder", min_pause=min_pause, verbose=verbose)
                except Exception as e:
                    print(f"[{show_elapsed_time()}] ({os.path.basename(clip_path)}) VAD Error: {e}")
                    continue
                
                intervals = vad_tg.tiers[0].intervals
                valid_intervals = [interval for interval in intervals if interval.mark not in ["", None] and interval.maxTime - interval.minTime > min_speech]
                # print(valid_intervals)
                
                for idx, valid_interval in enumerate(valid_intervals):
                    s, e = valid_interval.minTime, valid_interval.maxTime

                    interval_path = os.path.join(tmp_path, os.path.basename(clip_path).replace(".wav", f"_{idx}.wav"))
                    audio_clip[s*1000:e*1000].save(interval_path)
                    text = self.model.transcribe(interval_path)

                    text = purify_text(text)
                    if not text:
                        continue
                    
                    if not is_single_language(text) and language is not None and enable_post_process:
                        text_proc = post_process(text, language)
                        print(f"[{show_elapsed_time()}] ({os.path.basename(clip_path)}) Activate post-process: ({text}) -> ({text_proc})")
                        text = text_proc

                    s_point = s + start/1000
                    e_point = e + start/1000

                    if e_point >= audio_obj.duration_seconds:
                        e_point = audio_obj.duration_seconds
                    final_tg.tiers[0].add(s_point, e_point, text)

                    # print(audio_obj.duration_seconds)
                    print(f"[{show_elapsed_time()}] ({os.path.basename(clip_path)}) Detect speech: {s+start/1000:.3f} - {e+start/1000:.3f} ({text})")


            #############################
            # 因为在建立textgrid的时候使用了strict=False的mode，有可能存在某个tier是重复的
            # 需要检查并调整
            #############################


            # 检查并合并重叠的interval
            tier = final_tg.tiers[0]
            i = 1
            while i < len(tier.intervals):
                prev = tier.intervals[i - 1]
                curr = tier.intervals[i]
                # 如果当前interval与上一个interval重叠
                if curr.minTime < prev.maxTime:
                    # 合并：取最早开始和最晚结束
                    new_min = prev.minTime
                    new_max = max(prev.maxTime, curr.maxTime)
                    # 替换上一个interval
                    prev.maxTime = new_max
                    prev.mark = "(-)"
                    # 删除当前interval
                    del tier.intervals[i]
                    # 从头重新检查
                    i = 1
                else:
                    i += 1
            
            # 从头开始遍历每一个interval，如果其mark是"(-)"，则去audio_obj截取出这一段，保存到临时文件夹，并且用ASR跑一遍
            tier = final_tg.tiers[0]
            for interval in tier.intervals:
                if interval.mark == "(-)":
                    s_ms = interval.minTime * 1000
                    e_ms = interval.maxTime * 1000
                    clip = audio_obj[s_ms:e_ms]
                    tmp_wav = os.path.join(tmp_path, f"{os.path.splitext(os.path.basename(wav_path))[0]}_redo_{s_ms}_{e_ms}.wav")
                    clip.save(tmp_wav)
                    text = self.model.transcribe(tmp_wav)
                    text = purify_text(text)
                    if not text:
                        continue
                    
                    if not is_single_language(text) and language is not None and enable_post_process:
                        text_proc = post_process(text, language)
                        print(f"[{show_elapsed_time()}] ({os.path.basename(clip_path)}) Activate post-process: ({text}) -> ({text_proc})")
                        text = text_proc
                        
                    interval.mark = text
                    print(f"[{show_elapsed_time()}] ({os.path.basename(wav_path)}) Redo speech: {interval.minTime:.3f} - {interval.maxTime:.3f} ({text})")

            # ----------------------------
            final_tg.write(final_path)
                
        
        shutil.rmtree(tmp_path)
        print(f"--------------- Processing completed ---------------")


if __name__ == "__main__":
    model = init_model(
        "iic/SenseVoiceSmall",
        "Qwen/Qwen3-4B-Instruct-2507"
    )
    model.annote(
        input_path = r"E:\Corpus\ma\audio\50-2.wav",# os.path.abspath("input_single"),
        # seg_dur=20.,
        # min_pause=.8,
        language="zh",
        # verbose=False
    )
