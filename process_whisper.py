import os
import torch
import whisper
from textgrid import TextGrid, IntervalTier
import librosa
import numpy as np
from scipy.signal import convolve2d, find_peaks


# Check-ups
# 检查CUDA是否可用
# if torch.cuda.is_available():
#     print("CUDA可用，当前使用的设备为:", torch.cuda.get_device_name(0))
# else:
#     print("CUDA不可用，将使用CPU进行计算")



# defs
def transcribe_wav_file(wav, vad):
    """
    使用 Whisper 模型转录 .wav 文件
    
    :param file_path: .wav 文件的路径
    :param path_vad: VAD TextGrid 文件的路径
    :return: 转录结果
    """
    # 加载最佳模型（large-v3）并指定使用设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = whisper.load_model("large-v3-turbo", device=device)
    # 转录音频文件
    result = model.transcribe(wav, word_timestamps=True)
    language = result["language"]

    print(result)


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

            print(start_time, end_time, text)
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


    tg.append(tier)
    tg.write(wav.replace(".wav", ".TextGrid"))


def word_timestamp(wav, tg):

    # 加载音频文件
    y, sr = librosa.load(wav)

    # 创建一个新的IntervalTier
    max_time = librosa.core.get_duration(y=y, sr=sr)

    # 加载 TextGrid 文件
    tg = TextGrid.fromFile(tg_path)
    word_tier = [tier for tier in tg if tier.name == 'word'][0]

    # 计算 tg 的 segment 中 mark 不为空的 interval 的平均时长
    non_empty_intervals = [interval.maxTime - interval.minTime for tier in tg for interval in tier if interval.mark != ""]
    average_word_duration = np.mean(non_empty_intervals) if non_empty_intervals else 0
    print(average_word_duration)

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
            convolved_spectrogram = np.where(np.abs(convolved_spectrogram) < 20, 0, convolved_spectrogram)

            # 按频率轴求和，保持维度以方便后续绘图
            convolved_spectrogram = np.sum(np.abs(convolved_spectrogram), axis=0, keepdims=False)
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
    
    # 保存修改后的 TextGrid 文件
    tg.write(tg_path.replace(".TextGrid", "_processed.TextGrid"))


if __name__ == "__main__":
    data_path = os.path.abspath("data")
    # input_dir = os.path.abspath("input")
    # output_dir = os.path.abspath("output")

    fnames = [os.path.splitext(f)[0] for f in os.listdir(data_path) if f.endswith('.wav')]


    for fname in fnames:
        wav_path = os.path.join(data_path, fname + ".wav")
        tg_path = wav_path.replace(".wav", ".TextGrid")
        vad_path = wav_path.replace(".wav", "_VAD.TextGrid")

        # transcribe_wav_file(wav_path, vad=vad_path)
        word_timestamp(wav_path, tg_path)






