import os
from textgrid import TextGrid, IntervalTier
import librosa
import numpy as np
from scipy.signal import convolve2d, find_peaks, butter, filtfilt
import torch

try:
    import matplotlib.pyplot as plt
except ImportError:
    pass

try:
    from .utils import *
    from .VAD.core_auto import *
# from .praditor.tool_auto import * 
except ImportError:
    from utils import *
    from VAD.core_auto import *
from scipy.signal import argrelextrema
import os
import unicodedata


default_params = {'onset': {'amp': '1.47', 'cutoff0': '60', 'cutoff1': '10800', 'numValid': '475', 'eps_ratio': '0.093'}, 'offset': {'amp': '1.47', 'cutoff0': '60', 'cutoff1': '10800', 'numValid': '475', 'eps_ratio': '0.093'}}




def purify_text(text):
    """
    清理文本中的无效字符，保留所有语言的文字字符
    
    :param text: 输入的文本
    :return: 清理后的文本
    """

    text = text.strip()
    # 只删除标点符号，保留所有语言的文字字符
    text = ''.join('' if unicodedata.category(c).startswith('P') else c for c in text)
    return text



def detect_energy_valleys(wav_path, tg_path):
    try:
        y, sr = librosa.load(wav_path, mono=False, sr=16000)
        y = y[0]
        y = np.gradient(np.gradient(y))
    except ValueError:
        y, sr = librosa.load(wav_path, mono=True, sr=16000)
        y = np.gradient(np.gradient(y))

    

    # 使用librosa计算音频的均方根能量(rms)
    rms = librosa.feature.rms(y=y, frame_length=128, hop_length=32, center=True)[0]
    
    # 计算对应的时间轴
    time = librosa.times_like(rms, sr=sr, hop_length=32)

    
    # 找到波谷的索引
    valley_indices = find_peaks(-rms, width=(5, None))[0]


    tg_vad = TextGrid()
    tg_vad.read(tg_path)
    intervals = [interval for interval in tg_vad.tiers[0] if interval.mark != ""]


    # 定义筛选规则：
    # 1. 波谷中的波谷/拐点
    # 2. 不允许左高右低
    for idx, interval in enumerate(intervals):
        if idx == len(intervals) - 1:
            break
        
        current_interval = intervals[idx]
        next_interval = intervals[idx+1]

        if current_interval.maxTime != next_interval.minTime:
            continue

        cand_valleys = [t for t in time[valley_indices] if current_interval.minTime + 0.01 < t < next_interval.maxTime - 0.01]

        # 获取 cand_valleys 对应的 rms 值
        cand_valleys_rms = [rms[np.where(time == t)[0][0]] for t in cand_valleys]

        # 筛选出左相邻小于右相邻的波谷
        valid_valleys = []
        valid_valleys_rms = []
        if len(cand_valleys) >= 3:

            for idx_in_time, t in enumerate(cand_valleys):
                if idx_in_time == len(cand_valleys) - 1:
                    continue
                else:

                    if cand_valleys_rms[idx_in_time] < cand_valleys_rms[idx_in_time+1]:
                        valid_valleys.append(t)
                        valid_valleys_rms.append(rms[idx_in_time])
            
            if not valid_valleys:
                valid_valleys = cand_valleys
                valid_valleys_rms = cand_valleys_rms

        else:
            valid_valleys = cand_valleys
            valid_valleys_rms = cand_valleys_rms




        min_valley_time = valid_valleys[np.argmin(valid_valleys_rms)]
        
        current_interval.maxTime = min_valley_time
        next_interval.minTime = current_interval.maxTime
             
        # print(valid_valleys, valid_valleys_rms)
    tg_vad.write(tg_path.replace(".TextGrid", "_recali.TextGrid"))


def calc_power_valley(y, sr):
    """
    用librosa库计算一段信号的功率曲线，并找到最低的波谷所在的时间戳

    :param y: 音频信号
    :param sr: 采样率
    :return: 功率曲线, 最低波谷对应的时间戳
    """

    # 使用librosa计算音频的均方根能量(rms)
    rms = librosa.feature.rms(y=y, frame_length=512, hop_length=256, center=False)[0]
    
    # 计算对应的时间轴
    time = librosa.times_like(rms, sr=sr, hop_length=256)
    # 找到所有波谷的索引
    valley_indices = argrelextrema(rms, np.less)[0]
        
    # 获取所有波谷对应的时间戳
    valley_time = time[valley_indices]
    
    
    return rms, valley_time



def bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    if low == 0:
        b, a = butter(order, high, btype='low', output="ba")
        filtered_data = filtfilt(b, a, data)
    else:
        try:
            b, a = butter(order, [low, high], btype='bandpass', output="ba")
            filtered_data = filtfilt(b, a, data)
        except ValueError:  # 如果设置的最高频率大于了可接受的范围
            b, a = butter(order, low, btype='high', output="ba")
            filtered_data = filtfilt(b, a, data)
    return filtered_data


def find_valid_peaks(y, sr, db_thre=20, peak_prominence=10, nfft=2048, win_length=1024):
    # 计算频谱图
    spectrogram = librosa.stft(y, n_fft=nfft, win_length=win_length, center=True)
    spectrogram_db = librosa.amplitude_to_db(abs(spectrogram), ref=1.0)  # 使用librosa.amplitude_to_db已将y值转换为对数刻度，top_db=None确保不限制最大分贝值
    
    kernel = np.array([[-1, 0, 1]])
    convolved_spectrogram = convolve2d(spectrogram_db, kernel, mode='same', boundary='symm')
    convolved_spectrogram = np.where(np.abs(convolved_spectrogram) < db_thre, 0, convolved_spectrogram)

    # 按频率轴求和，保持维度以方便后续绘图
    convolved_spectrogram = np.sum(np.abs(convolved_spectrogram), axis=0, keepdims=False)
    # 在保持输出信号长度不变的情况下，对卷积后的频谱图求一阶导
    # convolved_spectrogram = np.gradient(convolved_spectrogram)
    time_axis = np.linspace(0, len(convolved_spectrogram) * librosa.core.get_duration(y=y, sr=sr) / len(convolved_spectrogram), len(convolved_spectrogram))

    # 找到所有的波峰和波谷
    peaks, _ = find_peaks(convolved_spectrogram, prominence=(peak_prominence, None))
    # valleys, _ = find_peaks(-convolved_spectrogram, prominence=(10, None))

    # valid_valleys = valleys[np.abs(convolved_spectrogram[valleys]) > 0]

    # 提取有效波峰和波谷对应的时间和值
    # peak_times = time_axis[valid_peaks]

    return peaks, time_axis, convolved_spectrogram


def compare_centroids(y, time_stamp, sr, length=0.05):
    """
    比较前后的音频片段的频谱能量质心
    """

    target_sample = int(time_stamp * sr)
    start_prev = target_sample - int(length * sr)
    end_prev = target_sample
    start_next = target_sample
    end_next = target_sample + int(length * sr)

    # 提取前后0.01s的音频片段
    y_prev = y[start_prev:end_prev]
    y_next = y[start_next:end_next]
    
    # 计算前0.01s的频谱能量质心
    centroid_prev = librosa.feature.spectral_centroid(y=y_prev, sr=sr)
    centroid_prev_mean = np.mean(centroid_prev)
    
    # 计算后0.01s的频谱能量质心
    centroid_next = librosa.feature.spectral_centroid(y=y_next, sr=sr)
    centroid_next_mean = np.mean(centroid_next)


    # 计算前0.01s的音频能量总量
    energy_prev = np.sum(y_prev ** 2)
    
    # 计算后0.01s的音频能量总量
    energy_next = np.sum(y_next ** 2)

    return centroid_prev_mean, centroid_next_mean, energy_prev, energy_next




def segment_audio(audio_obj, segment_duration=10, min_pause=0.2, params="self", verbose=False):
    wav_path = audio_obj.fpath

    print(f"[{show_elapsed_time()}] ({os.path.basename(wav_path)}) Start segmentation (<= {segment_duration}s)...")


    audio_obj = ReadSound(wav_path)

    # 获取 wav 文件所在的文件夹路径
    wav_folder = os.path.dirname(wav_path)
    all_txt_path = os.path.join(wav_folder, "params.txt")
    self_txt_path = wav_path.replace(".wav", ".txt")

    

    if params == "all":
        if os.path.exists(all_txt_path):
            with open(all_txt_path, "r") as f:
                params = eval(f.read())
        else:
            params = default_params
    
    elif params == "self":
        if os.path.exists(self_txt_path):
            with open(self_txt_path, "r") as f:
                params = eval(f.read())
        else:
            params = default_params
    elif params == "default":
        params = default_params

    else:  # 具体参数
        params = params
    

    segments = []

    y = audio_obj.arr
    sr = audio_obj.frame_rate

    audio_len = len(y) /sr

    start = 0.0 * 1000
    end = segment_duration * 1000
    while end <= audio_len * 1000:
        segment = audio_obj[start:end]
        # print(type(segment) == type(audio_obj))
        onsets = autoPraditorWithTimeRange(params, segment, "onset", verbose=False)
        offsets = autoPraditorWithTimeRange(params, segment, "offset", verbose=False)
        # print()
        # print(start, end)
        # print(onsets, offsets, audio_len * 1000)
        if not onsets or not offsets:
            segments[-1][1] = end
            # continue
        else:
            # 从最后一个onset开始往前遍历
            for i in range(len(onsets)-1, 0, -1):
                current_onset = onsets[i]
                # 找到当前onset之前的最后一个offset
                prev_offset = [xset for xset in offsets if xset < current_onset][-1]
                # prev_offset = offsets[i-1]
                if current_onset - prev_offset > min_pause:
                    # 若差值大于min_pause，则取得他们的均值
                    target_offset = (current_onset + prev_offset) / 2
                    end = start + target_offset * 1000
                    break
            else:
                # 若所有onset和对应offset差值都不大于min_pause，则取最后一个onset和第一个offset的均值
                target_offset = (onsets[-1] + offsets[0]) / 2
                end = start + target_offset * 1000


            # end = start + (target_offset + onsets[-1]) / 2 * 1000

            segments.append([start, end])

        start = end
        end = start + segment_duration * 1000

        if end > audio_len * 1000:
            if audio_len * 1000 - start > 10:
                segments[-1][1] = audio_len * 1000
                break
            else:
                segments.append([start, audio_len * 1000])
                break
    if not segments:
        segments.append([0.0, audio_len * 1000])
    
    # print(segments)
    # exit()
    return segments
    



def get_vad(wav_path, ori_wav_path, min_pause=0.2, params="self", if_save=False, verbose=False):
    print(f"[{show_elapsed_time()}] ({os.path.basename(wav_path)}) VAD processing started...")


    audio_obj = ReadSound(wav_path)

    # 获取 wav 文件所在的文件夹路径
    wav_folder = os.path.dirname(wav_path)
    all_txt_path = os.path.join(wav_folder, "params.txt")
    self_txt_path = ori_wav_path.replace(".wav", "_vad.txt")
    if not os.path.exists(self_txt_path):
        self_txt_path = ori_wav_path.replace(".wav", ".txt")

    
    # print(f"[{show_elapsed_time()}] ({os.path.basename(wav_path)}) VAD params: {params}")
    # print(default_params)
    if params == "all":
        if os.path.exists(all_txt_path):
            with open(all_txt_path, "r") as f:
                params = eval(f.read())
        else:
            params = default_params
    
    elif params == "self":
        if os.path.exists(self_txt_path):
            with open(self_txt_path, "r") as f:
                params = eval(f.read())
        else:
            params = default_params
    elif params == "default":
        params = default_params

    else:  # 具体参数
        params = params
    # print(params)


    onsets = autoPraditorWithTimeRange(params, audio_obj, "onset", verbose=False)
    offsets = autoPraditorWithTimeRange(params, audio_obj, "offset", verbose=False)

    if verbose:   
        print(f"[{show_elapsed_time()}] ({os.path.basename(wav_path)}) VAD onsets: {onsets}")
        print(f"[{show_elapsed_time()}] ({os.path.basename(wav_path)}) VAD offsets: {offsets}")




    # Select the one offset that is closest to onset and earlier than onset
    valid_onsets = []
    valid_offsets = []
    for i, onset in enumerate(onsets):
        # print(onset)
        if i == 0:
            valid_offsets.append(offsets[-1])
            valid_onsets.append(onset)
        else:
            try:
                valid_offsets.append(max([offset for offset in offsets if onsets[i-1] < offset < onset]))
                valid_onsets.append(onset)

            except ValueError:
                pass
    


    onsets = sorted(valid_onsets)
    offsets = sorted(valid_offsets)

    tg = TextGrid()
    interval_tier = IntervalTier(name="interval", minTime=0., maxTime=audio_obj.duration_seconds)



    bad_onsets = []
    bad_offsets = []

    for i in range(len(onsets)-1):
        if onsets[i+1] - offsets[i] < min_pause:
            bad_onsets.append(onsets[i+1])
            bad_offsets.append(offsets[i])

    onsets = [x for x in onsets if x not in bad_onsets]
    offsets = [x for x in offsets if x not in bad_offsets]
    
    for onset, offset in zip(onsets, offsets):
        interval_tier.add(onset, offset, "+")


    tg.append(interval_tier)
    tg.write(wav_path.replace(".wav", "_vad.TextGrid"))  # 将TextGrid对象写入文件

    tg = TextGrid()
    tg.read(wav_path.replace(".wav", "_vad.TextGrid"))

    if not if_save:
        os.remove(wav_path.replace(".wav", "_vad.TextGrid"))
    else:
        print(f"[{show_elapsed_time()}] ({os.path.basename(wav_path)}) VAD results saved")
    
    
    return tg

def transcribe_wav_file(wav_path, vad, whisper_model, language, if_save=False):
    """
    使用 Whisper 模型转录 .wav 文件
    
    :param file_path: .wav 文件的路径
    :param path_vad: VAD TextGrid 文件的路径
    :return: 转录结果
    """

    # 转录音频文件
    initial_prompt = """
    请保留所有语气词，包括但不限于嗯、啊、呃、唉、呵、呼、哼、咳、呜、哇、呀、喔、哦、哎、嘛。
    """


    if language is None:
        result = whisper_model.transcribe(wav_path, initial_prompt=initial_prompt, fp16=torch.cuda.is_available(), word_timestamps=True)
    else:
        result = whisper_model.transcribe(wav_path, initial_prompt=initial_prompt, fp16=torch.cuda.is_available(), word_timestamps=True, language=language)

    # language = result["language"]
    language = is_single_language(result["text"])
    print(result["text"])

    print(f"[{show_elapsed_time()}] ({os.path.basename(wav_path)}) Transcribing into {language}...")
    # print(result)

    # 加载 path_vad 对应的 TextGrid 文件
    vad_tg = vad
    # 提取所有 mark 为空字符串的 interval 的起止时间
    vad_intervals = []
    sil_intervals = []
    for tier in vad_tg:
        for interval in tier:
            if interval.mark == "" or interval.mark is None:
                sil_intervals.append((interval.minTime, interval.maxTime))
            else:
                vad_intervals.append((interval.minTime, interval.maxTime))


    tg = TextGrid()
    tier = IntervalTier(name='word', minTime=0.0, maxTime=vad_tg.tiers[0].maxTime)

    # print(empty_mark_intervals)
    for segment in result["segments"]:
        intervals = []
        for idx, word in enumerate(segment["words"]):
            start_time = segment["words"][idx]["start"]
            end_time = segment["words"][idx]["end"]
            # print( segment["words"][idx])
            # if  segment["words"][idx]["text"].strip() == "":
            #     continue
            # print(start_time, end_time, segment["words"][idx]["word"])
            
            text = ''.join(c for c in segment["words"][idx]["word"] if c.isalpha() or c.isspace() or c.isnumeric() or not unicodedata.category(c).startswith('P'))  # 去掉标点符号
            text = text.strip()
            if not text:
                continue


            for sil_interval in sil_intervals:
                # silence in sound
                if start_time < sil_interval[0] < sil_interval[1] < end_time:
                    if end_time - sil_interval[1] > sil_interval[0] - start_time:
                        start_time = sil_interval[1]
                    else:
                        end_time = sil_interval[0]
                    
        
            for vad_interval in vad_intervals:
                if vad_interval[0] <= start_time <= vad_interval[1] <= end_time:
                    end_time = vad_interval[1]
                    
                    try:
                        if segment["words"][idx+1]["start"] == segment["words"][idx]["end"]:
                            segment["words"][idx+1]["start"] = end_time
                            
                    except IndexError:
                        pass
                
                elif start_time <= vad_interval[0] <= end_time <= vad_interval[1]:
                    start_time = vad_interval[0]
            

            # print(start_time, end_time, text)
            if intervals and round(float(start_time), 4) == round(float(intervals[-1][0]), 4):
                intervals[-1][1] = max(end_time, intervals[-1][1])
                intervals[-1][2] += text
                
            elif intervals and start_time == end_time:
                intervals[-1][2] += text
            else:
                intervals.append([start_time, end_time, text])
            # print()
        
        # for idx, interval in enumerate(intervals):
        #     if idx == len(intervals) - 1:
        #         continue

        #     if intervals[idx][0] != intervals[idx-1][1]:
        #         for start_time, end_time in intervals:
        #             if start_time <= intervals[idx-1][1] <= end_time:
        #                 intervals[idx-1][1] = end_time
        #                 break
        for start_time, end_time in vad_intervals:
            within_indices = [idx for idx, interval in enumerate(intervals) if start_time <= interval[0] <= interval[1] <= end_time]
            if not within_indices:
                continue
            intervals[within_indices[0]][0] = start_time
            intervals[within_indices[-1]][1] = end_time
            


        for start_time, end_time, text in intervals:
            # print(start_time, end_time, text)
            tier.add(start_time, end_time, text)

    # 检查tier里是否有mark=”+“的interval，若有则删除
    tier.intervals = [interval for interval in tier.intervals if interval.mark != "+"]
    # print(tier.intervals)
    # 遍历 word_tier 中的每一个 interval
    # print()
    new_intervals = []
    for interval in tier.intervals:
        if len(interval.mark) > 1:
            # 计算每个新 interval 的时长
            duration = (interval.maxTime - interval.minTime) / len(interval.mark)
            start_time = interval.minTime
            # 将每个中文字拆分为一个新的 interval
            for char in interval.mark:
                new_interval = Interval(start_time, start_time + duration, char)
                new_intervals.append(new_interval)
                start_time += duration
        else:
            new_intervals.append(interval)
    # 替换原有的 intervals
    tier.intervals = new_intervals


    for idx, interval in enumerate(tier.intervals):
        if idx == len(tier.intervals) -1:
            break


        current_interval = tier.intervals[idx]
        next_interval = tier.intervals[idx + 1]

        if next_interval.minTime < current_interval.maxTime and next_interval.mark != "":
            current_interval.maxTime = next_interval.minTime

    tg.append(tier)

    if if_save:
        tg.write(wav_path.replace(".wav", "_whisper.TextGrid"))
        print(f"[{show_elapsed_time()}] ({os.path.basename(wav_path)}) Whisper word-level transcription saved")
    return language, tg


def phon_timestamps(wav_path, tg_path, language, if_save=False):

    if language.lower() not in ['zh', 'en', 'yue']:
        print(f"[{show_elapsed_time()}] Language {language} not currently supported.")

        wav_folder = os.path.dirname(os.path.dirname(wav_path))
        output_path = os.path.join(wav_folder, "output")
        os.makedirs(output_path, exist_ok=True)
        new_tg_path = os.path.join(output_path, os.path.basename(wav_path).replace(".wav", ".TextGrid"))
        tg.write(new_tg_path)
        return


    print(f"[{show_elapsed_time()}] ({os.path.basename(wav_path)}) Trimming word-level annotation...")
    detect_energy_valleys(wav_path, tg_path)

    return
    y, sr = librosa.load(wav_path, sr=16000)  # 加载音频文件
    # y = bandpass_filter(y, 200, 8000, sr)
    
    tg = TextGrid.fromFile(tg_path)  # 加载 TextGrid 文件

    word_tier = [tier for tier in tg if tier.name == 'word'][0]

    # 计算 tg 的 segment 中 mark 不为空的 interval 的平均时长
    non_empty_intervals = [interval.maxTime - interval.minTime for tier in tg for interval in tier if interval.mark != ""]
    average_word_duration = np.mean(non_empty_intervals) if non_empty_intervals else 0
    
    word_intervals = [interval for interval in word_tier.intervals if interval.mark != ""]
    for i in range(len(word_intervals) - 1):
        current_interval = word_intervals[i]
        next_interval = word_intervals[i + 1]
        # 检查两个 interval 是否相粘着（前一个的结束时间等于后一个的开始时间）

        if current_interval.maxTime == next_interval.minTime:
            target_boundary = current_interval.maxTime - current_interval.minTime

            start_sample = int(current_interval.minTime * sr)
            end_sample = int(next_interval.maxTime * sr)
            y_vad = y[start_sample:end_sample]

            peaks, time_axis, convolved_spectrogram = find_valid_peaks(y_vad, sr)

            peak_times = time_axis[peaks]

            # 筛选出不在 current_interval.minTime 到 current_interval.minTime + 0.05s 之间的波峰
            valid_peak_times = [t for t in peak_times if t >= 0.05 and (target_boundary -  average_word_duration/2 <= t <= target_boundary + average_word_duration * 3/4)]

            if valid_peak_times:
                # 找到距离 target_boundary 最近且最大的波峰
                # 获取波峰对应的数值
                peak_values_nearby = [convolved_spectrogram[int((t / librosa.core.get_duration(y=y_vad, sr=sr)) * len(convolved_spectrogram))] for t in valid_peak_times]
                # 找到最大波峰对应的时间
                closest_peak_time = valid_peak_times[np.argmax(peak_values_nearby)]
            else:
                closest_peak_time = target_boundary
            
            # 找到之后，开始写入
            target_boundary = closest_peak_time + current_interval.minTime

            current_interval.maxTime = target_boundary
            next_interval.minTime = target_boundary
            print(target_boundary)
            print("SR:", sr)
            # 以target_boundary为0点，分别计算-0.01s到0s以及0s到0.01s的频谱能量质心
            
            centroid_prev_mean, centroid_next_mean, energy_prev, energy_next = compare_centroids(y, target_boundary, sr)

            print(current_interval.mark, next_interval.mark, f"{centroid_prev_mean:.2f} Hz", f"{centroid_next_mean:.2f} Hz", f"{energy_prev:.2f}", f"{energy_next:.2f}")

            if centroid_prev_mean > centroid_next_mean and energy_prev < energy_next:
                next_cvt = extract_cvt(next_interval.mark, lang=language)[0]
                print(next_interval.mark, next_cvt)
                if next_cvt[0] in ["d", "l"]:
                    y_alt = bandpass_filter(y, 4000, 6000, sr)
                elif not next_cvt[0]:
                    y_alt = bandpass_filter(y, 0, 2000, sr)
                else:
                    y_alt = y


                # 从 target_boundary 往前，在 peak_times 里找第一个左质心小于右质心的峰值
                # 将峰值时间按从大到小排序，从靠近 target_boundary 的峰值开始检查
                sorted_peak_times = sorted([t for t in peak_times if target_boundary - current_interval.minTime < t < next_interval.maxTime - current_interval.minTime]) + \
                                    sorted([t for t in peak_times if 0.05 < t < target_boundary - current_interval.minTime - 0.000001], reverse=True) # 要做两个方向
                print(sorted_peak_times)
# !

                for peak_time in sorted_peak_times:
                    # 计算该峰值对应的实际时间点
                    actual_peak_time = peak_time + current_interval.minTime
                    
                    new_centroid_prev_mean, new_centroid_next_mean, new_energy_prev, new_energy_next = compare_centroids(y_alt, actual_peak_time, sr)
                    print("\t", actual_peak_time, f"{new_centroid_prev_mean:.2f} Hz", f"{new_centroid_next_mean:.2f} Hz", f"{new_energy_prev:.2f}", f"{new_energy_next:.2f}")
                    if next_cvt[0] in ["d", "l"]:
                        if new_energy_prev < new_energy_next:
                            target_boundary = actual_peak_time
                            # print("调整后", target_boundary, f"{centroid_prev_mean} Hz", f"{centroid_next_mean} Hz")
                            current_interval.maxTime = target_boundary
                            next_interval.minTime = target_boundary
                            break
                    
                    else:
                        if new_centroid_prev_mean < new_centroid_next_mean: # and new_energy_prev < new_energy_next:
                            target_boundary = actual_peak_time
                            # print("调整后", target_boundary, f"{centroid_prev_mean} Hz", f"{centroid_next_mean} Hz")
                            current_interval.maxTime = target_boundary
                            next_interval.minTime = target_boundary
                            break
                    
            print()

    # y, sr = librosa.load(wav_path, sr=18000)


    tg.write(wav_path.replace(".wav", "_recali.TextGrid"))

    phon_tier = IntervalTier(name="phoneme", minTime=0, maxTime=word_tier.maxTime)

    for interval in word_intervals:

        con, vow, tone = extract_cvt(interval.mark, lang=language)[0]
        # !先尝试分开声韵母

        # print(cvt)
        start_sample = int(interval.minTime * sr)
        end_sample = int(interval.maxTime * sr)
        # print(interval.mark, interval.minTime, interval.maxTime)

        y_vad = y[start_sample:end_sample]

        expected_num = 2 if con else 1 #len(vow) + 1 if con else len(vow)
        # print(expected_num)

        peaks, time_axis, convolved_spectrogram = find_valid_peaks(y_vad, sr)

        # 按峰值大小对峰值索引进行排序
        # 筛选出左能量质心高于右能量质心的 peak
        valid_peaks = []
        for peak in peaks:
            peak_time = time_axis[peak] + interval.minTime
            centroid_prev_mean, centroid_next_mean, energy_prev, energy_next = compare_centroids(y, peak_time, sr)
            # print(f"频谱能量质心（-0.01s到0s）: {centroid_prev_mean} Hz | 频谱能量质心（0s到0.01s）: {centroid_next_mean} Hz | 频谱能量（-0.01s到0s）: {energy_prev} | 频谱能量（0s到0.01s）: {energy_next}")
            if centroid_prev_mean > centroid_next_mean: # and energy_prev > energy_next:
                valid_peaks.append(peak)
        
        # 获取前 expected_num-1 个最大的符合条件的 peak
        sorted_peaks = sorted(valid_peaks, key=lambda x: convolved_spectrogram[x], reverse=True)[:expected_num-1]


        # 获取波峰对应的时间戳
        peak_times = time_axis[sorted_peaks]

        peak_timestamps = [interval.minTime] + [pt + interval.minTime for pt in peak_times] + [interval.maxTime]

        # peak_timestamps.sort()
        peak_timestamps = sorted(peak_timestamps, reverse=True)
        # print(peak_timestamps)

        for t, time_stamp in enumerate(peak_timestamps):
            if t == 0:
                phon_tier.add(peak_timestamps[t+1], peak_timestamps[t], "".join(vow))
            else:
                try:
                    phon_tier.add(peak_timestamps[t+1], peak_timestamps[t], con)
                except IndexError:
                    pass
    # print(tg.maxTime)
    tg.append(phon_tier)


    # 保存修改后的 TextGrid 文件
    # 检查 output 文件夹是否存在，如果不存在则创建
    wav_folder = os.path.dirname(os.path.dirname(wav_path))
    output_path = os.path.join(wav_folder, "output")
    os.makedirs(output_path, exist_ok=True)
    new_tg_path = os.path.join(output_path, os.path.basename(wav_path).replace(".wav", ".TextGrid"))
    tg.write(new_tg_path)
    print(f"[{show_elapsed_time()}] ({os.path.basename(wav_path)}) Phoneme-level segmentation saved")

    return tg


if __name__ == "__main__":
    segments = segment_audio("data/test_audio.wav", segment_duration=3.5)
    # print(segments)
