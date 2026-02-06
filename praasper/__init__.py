import os
import shutil
import itertools
import numpy as np
import gc
import torch
from itertools import product
import copy
import random
import jellyfish
import concurrent.futures
from tqdm import tqdm

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
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["DISABLE_TQDM"] = "1"

# 清空资源
def clear_resources():
    # 清理 CUDA 缓存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    # 强制垃圾回收
    gc.collect()

class init_model:

    def __init__(
        self,
        ASR: str="iic/SenseVoiceSmall",
        infer_mode: str = "funasr",
        # LLM: str="Qwen/Qwen2.5-1.5B-Instruct",
        device: str= "cpu"
    ):

        # 验证 infer_mode 参数值
        allowed_modes = ["direct", "funasr"]
        if infer_mode not in allowed_modes:
            raise ValueError(f"infer_mode must be one of {allowed_modes}, got {infer_mode}")
        
        # if infer_mode == "direct" and ASR != "FunAudioLLM/Fun-ASR-Nano-2512":
        #     self.tokenizer = init_tokenizer(ASR)
        
        self.ASR = ASR
        self.infer_mode = infer_mode
        # self.LLM = LLM
        self.device = device


        print(f"[{show_elapsed_time()}] Trying device ({self.device})...")
        # 检测硬件
        if self.device == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
                print(f"[{show_elapsed_time()}] CUDA detected, using GPU.")
            else:
                self.device = "cpu"
                print(f"[{show_elapsed_time()}] CUDA not available, using CPU.")
        else:
            print(f"[{show_elapsed_time()}] Using device: {self.device}")


        self.model = SelectWord(
            model=self.ASR,
            infer_mode=self.infer_mode,
            device=self.device
        )

        # init_LLM(self.LLM)


        # self.g2p = G2PModel()

        self.params = {'onset': {'amp': '1.47', 'cutoff0': '60', 'cutoff1': '10800', 'numValid': '475', 'eps_ratio': '0.093'}, 'offset': {'amp': '1.47', 'cutoff0': '60', 'cutoff1': '10800', 'numValid': '475', 'eps_ratio': '0.093'}}


    def annote(
        self,
        input_path: str,
        seg_dur=10.,
        min_pause=0.2,
        skip_existing: bool=False,
        verbose: bool=False,
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


            print(f"--------------- Locate optimal parameters for {os.path.basename(wav_path)} ---------------")
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
            segments = segment_audio(audio_obj, segment_duration=seg_dur, params=self.params, min_pause=min_pause)

            def process_segments(segment):
            # for start, end in segments:
                start, end = segment
                # count += 1

                # print(f"[{show_elapsed_time()}] Processing segment: {start/1000:.3f} - {end/1000:.3f} ({count})")
                audio_clip = audio_obj[start:end]
                clip_path = os.path.join(tmp_path, os.path.basename(wav_path).replace(".wav", f"_{start}_{end}.wav"))
                audio_clip.save(clip_path)

                audio_clip_result = self.model.transcribe(clip_path)
                audio_clip_single_words = audio_clip_result[0][0]["ctc_timestamps"]
                # print(audio_clip_single_words)


                # try:
                vad_tg = get_vad(clip_path, params=self.params, min_pause=min_pause, verbose=verbose)
                # except Exception as e:
                #     print(f"[{show_elapsed_time()}] ({os.path.basename(clip_path)}) VAD Error: {e}")
                #     return [None, None, None]
                
                intervals = vad_tg.tiers[0].intervals
                valid_intervals = [interval for interval in intervals if interval.mark not in ["", None]]# and interval.maxTime - interval.minTime > min_speech]
                # print(valid_intervals)

                # 整体转录模式
                for word in audio_clip_single_words:
                    s_w = word["start_time"]
                    e_w = word["end_time"]
                    t_w = word['token']

                    best_interval = None
                    max_overlap = -1
                    min_distance = float('inf')
                    closest_interval = None

                    for interval in valid_intervals:
                        s_i, e_i = interval.minTime, interval.maxTime
                        
                        # 计算重叠时间
                        overlap_start = max(s_w, s_i)
                        overlap_end = min(e_w, e_i)
                        overlap = max(0, overlap_end - overlap_start)
                        
                        # 如果有重叠且比当前最大重叠大
                        if overlap > max_overlap:
                            max_overlap = overlap
                            best_interval = interval
                        
                        # 计算距离（无重叠时使用）
                        if overlap == 0:
                            # 计算单词与间隔的距离
                            if e_w <= s_i:
                                # 单词在间隔前
                                distance = s_i - e_w
                            else:
                                # 单词在间隔后
                                distance = s_w - e_i
                            
                            if distance < min_distance:
                                min_distance = distance
                                closest_interval = interval
                    
                    # 选择最佳间隔
                    if best_interval is not None:
                        selected_interval = best_interval
                    else:
                        selected_interval = closest_interval
                    
                    # 将单词添加到选定的间隔
                    if selected_interval is not None:
                        if selected_interval.mark == "+":
                            selected_interval.mark = ""
                        
                        selected_interval.mark += f"{t_w}"
                        # selected_interval.words.append((t_w, s_w, e_w))
                
                timestamps = []
                for idx, valid_interval in enumerate(valid_intervals):
                    s, e = valid_interval.minTime, valid_interval.maxTime

                    # 获取间隔的文本（由前面的单词拼接而成）
                    text = valid_interval.mark

                    text = purify_text(text)
                    if not text or text == "+":
                        continue
                

                    s_point = s + start/1000
                    e_point = e + start/1000

                    if e_point >= audio_obj.duration_seconds:
                        e_point = audio_obj.duration_seconds
                    
                    timestamps.append([s_point, e_point, text])

                return timestamps

            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:  # 使用线程池并发处理
                result = list(tqdm(executor.map(process_segments, segments), total=len(segments), desc="Processing segments"))

            #############################
            # 因为在建立textgrid的时候使用了strict=False的mode，有可能存在某个tier是重复的
            # 需要检查并调整
            #############################
            results = result[0]
            for r in result[1:]:
                results.extend(r)
            for s_point, e_point, text in results:
                if s_point is None or e_point is None or text is None:
                    continue
                final_tg.tiers[0].add(s_point, e_point, text)

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
                    prev.mark += curr.mark
                    # 删除当前interval
                    del tier.intervals[i]
                    # 从头重新检查
                    i = 1
                else:
                    i += 1
 
            # ----------------------------
            final_tg.write(final_path)
                
        if os.path.exists(tmp_path):
            shutil.rmtree(tmp_path)
        print(f"--------------- Processing completed ---------------")


    def auto_vad(self, wav_path, min_speech=0.2, min_pause=0.2, verbose=False):
        """
        自动选取最优的VAD参数，根据随机选取的10秒音频。

        """


        # print(f"[{show_elapsed_time()}] ({os.path.basename(wav_path)}) VAD processing started...")


        dir_name = os.path.dirname(os.path.dirname(wav_path))
        tmp_path = os.path.join(dir_name, "tmp")

        audio_obj = ReadSound(wav_path)
        print(f"[{show_elapsed_time()}] ({os.path.basename(wav_path)}) Full audio duration: {audio_obj.duration_seconds:.3f}")

        if audio_obj.duration_seconds > 10:
            # 计算最大起始时间
            max_start = audio_obj.duration_seconds - 10
            # 随机生成起始时间
            start_time = random.uniform(0, max_start)
            end_time = start_time + 10
        else:
            # 音频不足时间，选取整个音频
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


        standard_result = self.model.transcribe(selected_audio_path)
        standard_transcript = standard_result[0][0]["text_tn"]  # text_tn 没有标点符号


        # params = default_params
        # params["offset"] = params["onset"]  # VAD模式特供


        def generate_param_grid(params):
            """生成参数组合列表"""
            keys = params.keys()
            values = params.values()
            for combination in product(*values):
                yield dict(zip(keys, combination))


        # df_res = pd.DataFrame(columns=["params", "transcript", "similarity"])
        # res = {}
        result = []

        # 使用示例
        param_grid = {
            'amp': np.arange(1.1, 1.5, 0.2),  #！找最多interval的amp
            "cutoff0": [0, 200],#range(0, 400, 200),
            'cutoff1': [min(audio_obj.frame_rate//2, 10800)],
            "numValid": [2000],#[int(min_speech/2*audio_obj.frame_rate//2)],

            'eps_ratio': np.arange(0.01, 0.15, 0.04)
        }
        def grid_search_optimal_params(params_replace):
        # for params_replace in generate_param_grid(param_grid):
            # res_key = (params_replace["amp"], params_replace["cutoff0"])
            adjusted_params = default_params.copy()
            adjusted_params["offset"] = adjusted_params["onset"]
            # print(adjusted_params)
            for p in params_replace:
                adjusted_params["onset"][p] = str(params_replace[p])
            # print(params_replace)
            adjusted_params["offset"] = adjusted_params["onset"]

            # print(adjusted_params)
            # print(f"[{show_elapsed_time()}] Testing {adjusted_params["onset"]}")
            # exit()
            
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

                os.makedirs(tmp_path, exist_ok=True)  # 确保临时目录存在

                audio_obj_clipped = None
                for i, (onset, offset) in enumerate(zip(onsets, offsets)):
                    if audio_obj_clipped is None:
                        audio_obj_clipped = selected_audio[onset*1000:offset*1000]
                    else:
                        audio_obj_clipped += selected_audio[onset*1000:offset*1000]
                    
                    if i != len(onsets)-1:
                        # 创建500ms空白音频
                        silence_duration = 0.2  # 500ms
                        silence_samples = int(silence_duration * selected_audio.frame_rate)
                        silence_arr = np.zeros(silence_samples, dtype=np.float32)
                        silence_audio = ReadSound(fpath=None, arr=silence_arr, duration_seconds=silence_duration, frame_rate=selected_audio.frame_rate)
                        audio_obj_clipped += silence_audio 
                # 为裁剪的音频文件生成唯一的文件名
                base_name = os.path.splitext(os.path.basename(wav_path))[0]
                clip_path = os.path.join(tmp_path, f"{base_name}_clip_{i}.wav")
                audio_obj_clipped.save(clip_path)

            
                # 转录并获取结果
                clip_result = self.model.transcribe(clip_path)
                clip_transcript = clip_result[0][0]["text_tn"]

                similarity = jellyfish.jaro_winkler_similarity(clip_transcript, standard_transcript)
                similarity = min(similarity, 0.9)
                num_intervals = len(onsets)
            else:
                similarity = 0.
                num_intervals = 0  # 可能有潜在bug，即所有num_intervals都等于0（日后待修）

            # 复制两个变量
            num_intervals_copy = num_intervals
            adjusted_params_copy = copy.deepcopy(adjusted_params)

            # 然后可以像原来一样使用复制后的变量
            return [num_intervals_copy, similarity, adjusted_params_copy]
            # result.append([num_intervals_copy, similarity, adjusted_params_copy])
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:  # 使用线程池并发处理
            result = list(tqdm(executor.map(grid_search_optimal_params, generate_param_grid(param_grid)),
                              total=len(list(generate_param_grid(param_grid))),
                              desc="Grid searching VAD params"))
        
        max_result = max(result, key=lambda x: (x[1], x[0]))
        max_item, best_params = max_result[0], max_result[2]
        
        print(f"[{show_elapsed_time()}] ({os.path.basename(wav_path)}) VAD best numIntervals: {max_item}, params: {best_params['onset']}")

        self.params = best_params

    def release_resources(self):
        """释放模型资源"""
        if hasattr(self, 'model') and self.model is not None:
            # 如果是直接模式，尝试释放模型
            if hasattr(self.model, 'to'):
                self.model.to('cpu')
            # 清空模型引用
            del self.model
        # 清空资源
        clear_resources()
        print("Resources released successfully")

# 导出clear_resources函数
__all__ = ['init_model', 'clear_resources']