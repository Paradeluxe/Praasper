import os
import shutil
import itertools
import numpy as np

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

# default_params = {'onset': {'amp': '1.47', 'cutoff0': '60', 'cutoff1': '10800', 'numValid': '475', 'eps_ratio': '0.093'}, 'offset': {'amp': '1.47', 'cutoff0': '60', 'cutoff1': '10800', 'numValid': '475', 'eps_ratio': '0.093'}}

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


 
            # auto search best params
            self.auto_vad(
                wav_path=wav_path,
                min_pause=min_pause,
            )

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


    def auto_vad(self, wav_path, min_pause=0.2, verbose=False):
        """
        自动选取最优的VAD参数，根据随机选取的10秒音频。

        """


        print(f"[{show_elapsed_time()}] ({os.path.basename(wav_path)}) VAD processing started...")


        dir_name = os.path.dirname(os.path.dirname(wav_path))
        tmp_path = os.path.join(dir_name, "tmp")

        audio_obj = ReadSound(wav_path)
        print(f"[{show_elapsed_time()}] ({os.path.basename(wav_path)}) Full audio duration: {audio_obj.duration_seconds:.3f}")
        # 随机选取连续十秒音频，如果音频不够十秒则选取整个音频
        import random
        if audio_obj.duration_seconds > 10:
            # 计算最大起始时间
            max_start = audio_obj.duration_seconds - 10
            # 随机生成起始时间
            start_time = random.uniform(0, max_start)
            end_time = start_time + 10
        else:
            # 音频不足十秒，选取整个音频
            start_time = 0
            end_time = audio_obj.duration_seconds
        
        # 截取选定的音频段
        selected_audio = audio_obj[start_time*1000:end_time*1000]
        print(f"[{show_elapsed_time()}] ({os.path.basename(wav_path)}) Selected audio: {start_time:.3f} - {end_time:.3f}")
        
        # 确保临时目录存在
        os.makedirs(tmp_path, exist_ok=True)
        # 将selected_audio保存到tmp文件夹
        base_name = os.path.splitext(os.path.basename(wav_path))[0]
        selected_audio_path = os.path.join(tmp_path, f"{base_name}_selected_{start_time:.0f}_{end_time:.0f}.wav")
        selected_audio.save(selected_audio_path)
        print(f"[{show_elapsed_time()}] ({os.path.basename(wav_path)}) Selected audio saved to: {selected_audio_path}")



        standard_transcript = self.model.transcribe(selected_audio_path)
        standard_transcript = purify_text(standard_transcript)


        params = default_params
        params["offset"] = params["onset"]  # VAD模式特供

        from itertools import product

        def generate_param_grid(params):
            """生成参数组合列表"""
            keys = params.keys()
            values = params.values()
            for combination in product(*values):
                yield dict(zip(keys, combination))


        # df_res = pd.DataFrame(columns=["params", "transcript", "similarity"])
        res = {}

        # 使用示例
        param_grid = {
            'amp': [round(n, 2) for n in np.arange(1.1, 2.00, 0.1)],
            'numValid': range(1000, 4000, 1000)
        }


        for params_replace in generate_param_grid(param_grid):
            
            adjusted_params = params.copy()

            for p in params_replace:
                adjusted_params["onset"][p] = str(params_replace[p])
            
            adjusted_params["offset"] = adjusted_params["onset"]

            # print(adjusted_params)
            
            
            onsets = autoPraditorWithTimeRange(adjusted_params, selected_audio, "onset", verbose=False)
            offsets = autoPraditorWithTimeRange(adjusted_params, selected_audio, "offset", verbose=False)

            onsets = sorted(onsets)
            offsets = sorted(offsets)


            if verbose:   
                print(f"[{show_elapsed_time()}] ({os.path.basename(wav_path)}) VAD onsets: {onsets}")
                print(f"[{show_elapsed_time()}] ({os.path.basename(wav_path)}) VAD offsets: {offsets}")
                


            if onsets or offsets:
                if not onsets:
                    onsets = [0.0]
                if not offsets:
                    offsets = [selected_audio.duration_seconds]
            else:
                return tg
            
            if onsets[0] >= offsets[0]:
                onsets = [0.0] + onsets
            
            if offsets[-1] <= onsets[-1]:
                offsets.append(selected_audio.duration_seconds)

            # Select the one offset that is closest to onset and earlier than onset
            valid_onsets = []
            valid_offsets = []

            if len(onsets) <= len(offsets):
                for i, onset in enumerate(onsets):
                    # print(onset)
                    if i == 0:
                        valid_offsets.append(offsets[-1])  # 最后一个offset
                        valid_onsets.append(onsets[0])  # 第一个offset
                    else:
                        try:
                            valid_offsets.append(max([offset for offset in offsets if onsets[i-1] < offset < onset]))  # 不会影响到最后一个offset，应为需要夹在onset中间
                            valid_onsets.append(onset)  # 下一个onset

                        except ValueError:
                            pass
            else:  # len(onsets) > len(offsets)
                reversed_offsets = list(reversed(offsets))
                for i, reversed_offset in enumerate(reversed_offsets):
                    if i == 0:
                        valid_onsets.append(onsets[0])
                        valid_offsets.append(reversed_offset)
                    else:
                        try:
                            valid_onsets.append(min([onset for onset in onsets if reversed_offset < onset < reversed_offsets[i-1]]))
                            valid_offsets.append(reversed_offset)

                        except ValueError:
                            pass
                valid_offsets = list(reversed(valid_offsets))
            


            onsets = sorted(valid_onsets)
            offsets = sorted(valid_offsets)


            # 根据min pause 调整onset和offset
            bad_onsets = []
            bad_offsets = []

            for i in range(len(onsets)-1):
                if onsets[i+1] - offsets[i] < min_pause:
                    bad_onsets.append(onsets[i+1])
                    bad_offsets.append(offsets[i])

            onsets = [x for x in onsets if x not in bad_onsets]
            offsets = [x for x in offsets if x not in bad_offsets]

            # print(params_replace)
            # print(onsets, offsets)
            # print()


            # 确保临时目录存在
            os.makedirs(tmp_path, exist_ok=True)
            this_transcript = ""
            for i, (onset, offset) in enumerate(zip(onsets, offsets)):
                audio_obj_clipped = selected_audio[onset*1000:offset*1000]
                # 为裁剪的音频文件生成唯一的文件名
                base_name = os.path.splitext(os.path.basename(wav_path))[0]
                clip_path = os.path.join(tmp_path, f"{base_name}_clip_{i}.wav")
                audio_obj_clipped.save(clip_path)

                # 转录并获取结果
                transcript = self.model.transcribe(clip_path)
                transcript = purify_text(transcript)
                
                this_transcript += transcript
            
            similarity = calculate_ipa_similarity(this_transcript, standard_transcript)

            if similarity > 0.9:
                try:
                    res[params_replace["amp"]][params_replace["numValid"]] = list(zip(onsets, offsets))
                except KeyError:
                    res[params_replace["amp"]] = {params_replace["numValid"]: list(zip(onsets, offsets))}
            # print(res[params_replace["amp"]], )
            if len(res[params_replace["amp"]]) == len(param_grid["numValid"]):
                # 获取当前amp下所有numValid值对应的列表
                num_valid_lists = list(res[params_replace["amp"]].values())
                # 获取对应的numValid值
                # num_valid_values = list(res[params_replace["amp"]].keys())
                
                time_diffs = []
                # 生成所有numValid列表的两两组合
                for i, j in itertools.combinations(range(len(num_valid_lists)), 2):
                    list1 = num_valid_lists[i]
                    list2 = num_valid_lists[j]
                    # nv1 = num_valid_values[i]
                    # nv2 = num_valid_values[j]
                    # 计算时间差异
                    time_diff = calculate_time_diff(list1, list2)
                    # print(f"[{show_elapsed_time()}] Amp: {params_replace['amp']}, NumValid {nv1} vs {nv2}: Time difference = {time_diff:.4f}")
                    time_diffs.append(time_diff)

                # 计算当前 amp 下所有 numValid 两两组合的平均时间差异
                average_time_diff = np.mean(time_diffs) if time_diffs else np.inf
                print(f"[{show_elapsed_time()}] Amp: {params_replace['amp']}, Average time difference = {average_time_diff:.4f}")

                if average_time_diff == 0.:
                    print(f"Best parameters are {params_replace}")
                    break



        # print(res)

        # 清理临时文件
        import shutil
        if os.path.exists(tmp_path):
            try:
                shutil.rmtree(tmp_path)
                print(f"[{show_elapsed_time()}] Temporary directory {tmp_path} removed")
            except Exception as e:
                print(f"[{show_elapsed_time()}] Error removing temporary directory: {e}")


        return params_replace