import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import argrelextrema, butter, filtfilt, find_peaks
from textgrid import TextGrid


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


def plot_audio_power_curve(audio_path):
    """
    绘制整段音频的功率曲线
    
    参数:
    audio_path (str): 音频文件的路径
    """
    # 加载音频文件
    y, sr = librosa.load(audio_path, mono=False, sr=48000)

    y = y[0]

    tar_freq = 8000
    y = bandpass_filter(y, 0, tar_freq, sr)

    # 对音频数据进行重采样，将采样率从48000Hz重采样到16000Hz
    y = librosa.resample(y, orig_sr=sr, target_sr=16000)
    sr = 16000  # 更新采样率


    y = np.gradient(np.gradient(y))

    # y = bandpass_filter(y, 2000, 8000, sr)

    # 使用librosa计算音频的均方根能量(rms)
    rms = librosa.feature.rms(y=y, frame_length=128, hop_length=32, center=True)[0]
    
    # 计算对应的时间轴
    time = librosa.times_like(rms, sr=sr, hop_length=32)
    
    # 创建图形
    plt.figure(figsize=(10, 4))
    
    # 绘制功率曲线
    plt.plot(time, rms, alpha=0.3)
    plt.title('Audio Power Curve')
    plt.xlabel('Time (s)')
    plt.ylabel('Power')
    plt.grid(True)

    vertical_line = [.688, .80, .88, 1.16, 1.25, 1.55, 1.75, 1.84, 1.94, 2.22, 2.49]
    for v in vertical_line:
        plt.axvline(x=v, color='r', linestyle='--')

    # 找到波谷的索引
    valley_indices = find_peaks(-rms, width=(5, None))[0]
    # 标出波谷
    plt.scatter(time[valley_indices], rms[valley_indices], color='orange', label='Valley')
    # 绘制波谷连线
    plt.plot(time[valley_indices], rms[valley_indices], color='orange', linestyle='--', label='Valley Connection')

    tg_vad = TextGrid()
    tg_vad.read(r"C:\Users\User\Desktop\Praasper\data\mandarin_sent_whisper.TextGrid")
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
        print()
        print(cand_valleys, cand_valleys_rms)

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
             
        print(valid_valleys, valid_valleys_rms)

        for v in valid_valleys:
            plt.axvline(x=v, color='b', linestyle='--')


    
    plt.legend()
    
    # 显示图形
    plt.tight_layout()
    plt.show()

# 使用示例
if __name__ == "__main__":
    audio_file_path = r"C:\Users\User\Desktop\Praasper\data\mandarin_sent.wav"  # 替换为实际的音频文件路径
    plot_audio_power_curve(audio_file_path)
