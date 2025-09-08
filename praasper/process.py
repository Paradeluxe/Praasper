import os
import whisper
from textgrid import TextGrid, IntervalTier
import librosa
import numpy as np
from scipy.signal import convolve2d, find_peaks
import torch

try:
    from .tool import *
    from .VAD.core_auto import *
# from .praditor.tool_auto import * 
except ImportError:
    from tool import *
    from VAD.core_auto import *


def get_vad(wav_path, params="self"):

    print(f"[{show_elapsed_time()}] ({os.path.basename(wav_path)}) VAD processing started...")


    audio_obj = ReadSound(wav_path)

    # 获取 wav 文件所在的文件夹路径
    wav_folder = os.path.dirname(wav_path)
    all_txt_path = os.path.join(wav_folder, "params.txt")
    self_txt_path = wav_path.replace(".wav", ".txt")

    default_params = {'onset': {'amp': '1.47', 'cutoff0': '60', 'cutoff1': '10800', 'numValid': '475', 'eps_ratio': '0.093'}, 'offset': {'amp': '1.47', 'cutoff0': '60', 'cutoff1': '10800', 'numValid': '475', 'eps_ratio': '0.093'}}
    

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
    



    onsets = autoPraditorWithTimeRange(params, audio_obj, "onset")
    offsets = autoPraditorWithTimeRange(params, audio_obj, "offset")


    tg = TextGrid()
    interval_tier = IntervalTier(name="interval", minTime=0., maxTime=audio_obj.duration_seconds)
    for i in range(len(onsets)):
        try:
            interval_tier.addInterval(Interval(onsets[i], offsets[i], "+"))
        except ValueError:
            continue
    tg.append(interval_tier)
    tg.write(wav_path.replace(".wav", "_VAD.TextGrid"))  # 将TextGrid对象写入文件

    print(f"[{show_elapsed_time()}] ({os.path.basename(wav_path)}) VAD results saved")

# else:


# defs
def transcribe_wav_file(wav_path, vad, whisper_model):
    """
    使用 Whisper 模型转录 .wav 文件
    
    :param file_path: .wav 文件的路径
    :param path_vad: VAD TextGrid 文件的路径
    :return: 转录结果
    """

    # 转录音频文件
    result = whisper_model.transcribe(wav_path, fp16=torch.cuda.is_available(), word_timestamps=True)
    language = result["language"]
    print(f"[{show_elapsed_time()}] ({os.path.basename(wav_path)}) Transcribing into {language}...")
    # print(result)


    # 加载 path_vad 对应的 TextGrid 文件
    try:
        vad_tg = TextGrid.fromFile(vad)
    except FileNotFoundError:
        print(f"错误：未找到文件 {vad}")
        raise

    # 提取所有 mark 为空字符串的 interval 的起止时间
    vad_intervals = []
    empty_mark_intervals = []
    for tier in vad_tg:
        for interval in tier:
            if interval.mark == "":
                empty_mark_intervals.append((interval.minTime, interval.maxTime))
            else:
                vad_intervals.append((interval.minTime, interval.maxTime))



    tg = TextGrid()
    tier = IntervalTier(name='word', minTime=0.0, maxTime=vad_tg.tiers[0].maxTime)
    
    for segment in result["segments"]:
        for idx, word in enumerate(segment["words"]):
            start_time = word["start"]
            end_time = word["end"]
            
            text = word["word"]

            for empty_mark_interval in empty_mark_intervals:
                if empty_mark_interval[0] <= end_time <= empty_mark_interval[1]:
                    end_time = empty_mark_interval[0]
                
                if empty_mark_interval[0] <= start_time <= empty_mark_interval[1]:
                    start_time = empty_mark_interval[1]
                
                if start_time < empty_mark_interval[0] < empty_mark_interval[1] < end_time:
                    pass

            # print(start_time, end_time, text)
            tier.add(start_time, end_time, text)

    for vad_interval in vad_intervals:
        # 找到距离 vad_interval[0] 最近的 interval.minTime
        closest_interval = min(tier.intervals, key=lambda x: abs(x.minTime - vad_interval[0]))

        if closest_interval.minTime - vad_interval[0] != 0:
            closest_interval.minTime = vad_interval[0]

        # 找到距离 vad_interval[1] 最近的 interval.maxTime
        closest_interval = min(tier.intervals, key=lambda x: abs(x.maxTime - vad_interval[1]))

        if closest_interval.maxTime - vad_interval[1] != 0:
            closest_interval.maxTime = vad_interval[1]

    # 检查tier里是否有mark=”+“的interval，若有则删除
    tier.intervals = [interval for interval in tier.intervals if interval.mark != "+"]
    
    tg.append(tier)
    tg.write(wav_path.replace(".wav", "_whisper.TextGrid"))
    print(f"[{show_elapsed_time()}] ({os.path.basename(wav_path)}) Whisper word-level transcription saved")

    return language


def word_timestamp(wav_path, tg_path, language):

    if language.lower() not in ['zh', 'en', 'yue']:
        print(f"[{show_elapsed_time()}] Language {language} not currently supported.")

        wav_folder = os.path.dirname(wav_path)
        output_path = os.path.join(wav_folder, "output")
        os.makedirs(output_path, exist_ok=True)
        new_tg_path = os.path.join(output_path, os.path.basename(wav_path).replace(".wav", ".TextGrid"))
        tg.write(new_tg_path)
        return


    print(f"[{show_elapsed_time()}] ({os.path.basename(wav_path)}) Trimming word-level annotation...")
    # 加载音频文件
    y, sr = librosa.load(wav_path, sr=16000)

    # 创建一个新的IntervalTier
    max_time = librosa.core.get_duration(y=y, sr=sr)

    # 加载 TextGrid 文件
    tg = TextGrid.fromFile(tg_path)
    word_tier = [tier for tier in tg if tier.name == 'word'][0]

    # 计算 tg 的 segment 中 mark 不为空的 interval 的平均时长
    non_empty_intervals = [interval.maxTime - interval.minTime for tier in tg for interval in tier if interval.mark != ""]
    average_word_duration = np.mean(non_empty_intervals) if non_empty_intervals else 0
    # print(f"Speech rate (word dur) is {average_word_duration:.4f} seconds")

    # adjacent_pairs = []
    
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

            # 计算频谱图
            spectrogram = librosa.stft(y_vad, n_fft=2048, win_length=1024, center=True)
            spectrogram_db = librosa.amplitude_to_db(abs(spectrogram), ref=1.0)  # 使用librosa.amplitude_to_db已将y值转换为对数刻度，top_db=None确保不限制最大分贝值
            
            kernel = np.array([[-1, 0, 1]])
            convolved_spectrogram = convolve2d(spectrogram_db, kernel, mode='same', boundary='symm')
            convolved_spectrogram = np.where(np.abs(convolved_spectrogram) < 15, 0, convolved_spectrogram)

            # 按频率轴求和，保持维度以方便后续绘图
            convolved_spectrogram = np.sum(np.abs(convolved_spectrogram), axis=0, keepdims=False)
            # 在保持输出信号长度不变的情况下，对卷积后的频谱图求一阶导
            # convolved_spectrogram = np.gradient(convolved_spectrogram)
            time_axis = np.linspace(0, len(convolved_spectrogram) * librosa.core.get_duration(y=y_vad, sr=sr) / len(convolved_spectrogram), len(convolved_spectrogram))

            # 找到所有的波峰和波谷
            peaks, _ = find_peaks(convolved_spectrogram, prominence=(10, None))
            valleys, _ = find_peaks(-convolved_spectrogram, prominence=(10, None))


            # 只保留波峰和波谷绝对值大于100的点
            valid_peaks = peaks[np.abs(convolved_spectrogram[peaks]) > 0]
            valid_valleys = valleys[np.abs(convolved_spectrogram[valleys]) > 0]

            # 提取有效波峰和波谷对应的时间和值
            peak_times = time_axis[valid_peaks]
            peak_values = convolved_spectrogram[valid_peaks]

            valley_times = time_axis[valid_valleys]
            valley_values = convolved_spectrogram[valid_valleys]    

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
    

    phon_tier = IntervalTier(name="phoneme", minTime=0, maxTime=word_tier.maxTime)

    for interval in word_intervals:
        con, vow, tone = get_pinyin_info(interval.mark)
        expected_num = len(vow) + 1 if con else len(vow)
        phon_series = [con] + vow if con else vow
        # print(expected_num)



        start_sample = int(interval.minTime * sr)
        end_sample = int(interval.maxTime * sr)
        # print(interval.mark, interval.minTime, interval.maxTime)

        y_vad = y[start_sample:end_sample]


        # 计算频谱图
        spectrogram = librosa.stft(y_vad, n_fft=2048, win_length=1024, center=True)
        spectrogram_db = librosa.amplitude_to_db(abs(spectrogram), ref=1.0)  # 使用librosa.amplitude_to_db已将y值转换为对数刻度，top_db=None确保不限制最大分贝值
        
        kernel = np.array([[-1, 0, 1]])
        convolved_spectrogram = convolve2d(spectrogram_db, kernel, mode='same', boundary='symm')
        convolved_spectrogram = np.where(np.abs(convolved_spectrogram) < 20, 0, convolved_spectrogram)

        # 按频率轴求和，保持维度以方便后续绘图
        convolved_spectrogram = np.sum(np.abs(convolved_spectrogram), axis=0, keepdims=False)
        # 在保持输出信号长度不变的情况下，对卷积后的频谱图求一阶导
        # convolved_spectrogram = np.gradient(convolved_spectrogram)
        time_axis = np.linspace(0, len(convolved_spectrogram) * librosa.core.get_duration(y=y_vad, sr=sr) / len(convolved_spectrogram), len(convolved_spectrogram))

        # 找到所有峰值，指定最小峰值高度为 0，后续再筛选最大的前几个
        peaks, _ = find_peaks(convolved_spectrogram)

        if con in ["k", 'b', 't', 'p', 'd']:
            valid_peaks = [p for p in peaks if time_axis[p] <= len(y_vad)/sr - 0.05]

            # if "i" == vow[0]:
            #     vow = vow[0] + vow
            #     expected_num -= 1
            #     phon_series = [con] + vow if con else vow
        
        else:
            # 忽略掉所有头0.05s和后0.05s的peak
            valid_peaks = [p for p in peaks if time_axis[p] >= 0.05 and time_axis[p] <= len(y_vad)/sr - 0.05]
        
        peaks = np.array(valid_peaks)

        # 按峰值大小对峰值索引进行排序
        sorted_peaks = sorted(peaks, key=lambda x: convolved_spectrogram[x], reverse=True)
        # 假设前 5 个峰值最大，可根据实际需求修改数量
        peaks = sorted_peaks[:expected_num-1]
        
        # 获取波峰对应的时间戳
        peak_times = time_axis[peaks]

        peak_timestamps = [interval.minTime] + [pt + interval.minTime for pt in peak_times] + [interval.maxTime]

        peak_timestamps.sort()

        # print(peak_timestamps)
        for t, time_stamp in enumerate(peak_timestamps):
            if t == 0:
                continue
            phon_tier.add(peak_timestamps[t-1], peak_timestamps[t], phon_series[t-1])
    # print(tg.maxTime)
    tg.append(phon_tier)


    # 保存修改后的 TextGrid 文件
    # 检查 output 文件夹是否存在，如果不存在则创建
    wav_folder = os.path.dirname(wav_path)
    output_path = os.path.join(wav_folder, "output")
    os.makedirs(output_path, exist_ok=True)
    new_tg_path = os.path.join(output_path, os.path.basename(wav_path).replace(".wav", ".TextGrid"))
    tg.write(new_tg_path)
    print(f"[{show_elapsed_time()}] ({os.path.basename(wav_path)}) Phoneme-level segmentation saved")



