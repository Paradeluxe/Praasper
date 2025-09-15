import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import argrelextrema, butter, filtfilt, find_peaks
from textgrid import TextGrid
import pypinyin



import numpy as np
from scipy.ndimage import uniform_filter1d

def remove_low_frequency_drift(signal, window_size, padding_mode='reflect'):
    """
    使用平均池化去除信号中的低频漂移
    
    参数:
    signal -- 输入信号 (1D numpy数组)
    window_size -- 池化窗口大小(奇数)
    padding_mode -- 边界填充模式 ('reflect', 'nearest', 'mirror', 'constant')
    
    返回:
    corrected_signal -- 去除低频漂移后的信号
    drift_component -- 提取出的漂移分量
    """
    # 验证输入
    if window_size % 2 == 0:
        raise ValueError("窗口大小应为奇数，以保证对称性")
    if window_size > len(signal):
        raise ValueError("窗口大小不能超过信号长度")
    
    # 使用Scipy的高效1D均匀滤波器实现滑动平均
    drift_component = uniform_filter1d(signal, size=window_size, mode=padding_mode)
    
    # 从原始信号中减去漂移分量
    corrected_signal = signal - drift_component
    
    return corrected_signal, drift_component




def extract_cvt_zh(character):
    """
    给定一个中文单字，返回其对应的拼音（声母、韵母、声调）
    
    :param character: 单个中文字符
    :return: 包含声母、韵母、声调的字典
    """
    # if len(character) != 1:
        # raise ValueError("只能输入单个中文字符")
    cvts = []
    for char in character:
        # 获取拼音信息
        pinyin_result = pypinyin.pinyin(char, style=pypinyin.TONE3, heteronym=False)[0][0]
        
        # 提取声母、韵母和声调
        initial = pypinyin.pinyin(char, style=pypinyin.INITIALS, heteronym=False)[0][0]
        final_with_tone = pypinyin.pinyin(char, style=pypinyin.FINALS_TONE3, heteronym=False)[0][0]
        
        # 分离韵母和声调
        tone = ''
        final = final_with_tone
        for char in final_with_tone:
            if char.isdigit():
                tone = char
                final = final_with_tone.replace(char, '')
                break
        
        

        final = list(final)
        # 如果n和g相邻，合并成ng
        new_final = []
        i = 0
        while i < len(final):
            if i < len(final) - 1 and final[i] == 'n' and final[i+1] == 'g':
                new_final.append('ng')
                i += 2
            else:
                new_final.append(final[i])
                i += 1
        final = new_final

        cvts.append((
            initial if initial else '',
            final,
            tone if tone else ''
        ))
    
    return cvts



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




def detect_energy_valleys(wav_path, tg_path):
    y, sr = librosa.load(wav_path, mono=False, sr=16000)
    y = y[0]
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
             
        # print(valid_valleys, valid_valleys_rms)
    tg_vad.write(tg_path.replace(".TextGrid", "_recali.TextGrid"))


def plot_audio_power_curve(audio_path):
    """
    绘制整段音频的功率曲线
    
    参数:
    audio_path (str): 音频文件的路径
    """
    # 加载音频文件
    y, sr = librosa.load(audio_path, mono=False, sr=None)
    y = y[0]


    y = np.gradient(y)
    # y = np.gradient(np.gradient(y))

    # y = bandpass_filter(y, 50, sr, sr, order=4)

    # 使用librosa计算音频的均方根能量(rms)
    rms = librosa.feature.rms(y=y, frame_length=128, hop_length=32, center=True)[0]

    # 计算对应的时间轴
    time = librosa.times_like(rms, sr=sr, hop_length=32)
    
    # 创建图形
    plt.figure(figsize=(10, 4))
    
    # 绘制功率曲线
    # plt.plot(time, rms, alpha=0.3)
    plt.title('Audio Power Curve')
    plt.xlabel('Time (s)')
    plt.ylabel('Power')
    plt.grid(True)

    vertical_line = [.688, .80, .88, 1.16, 1.25, 1.55, 1.75, 1.84, 1.94, 2.22, 2.49,
                    3.51, 3.75, 3.94, 4.18, 4.29, 4.46, 4.63, 4.72, 4.958]
    for v in vertical_line:
        plt.axvline(x=v, color='r', linestyle='--')

    # 找到波谷的索引
    valley_indices = find_peaks(-rms, width=(1, None), distance=1)[0]


    # 获取当前函数中的采样率（假设在 plot_audio_power_curve 函数中 sr 变量可用）
    # 根据当前上下文，sr 在 plot_audio_power_curve 函数开头已定义
    # 开始时间和结束时间
    start_time = 0  
    end_time = time[-1]
    
    # 生成插值时间点
    num_samples = int((end_time - start_time) * sr)
    interpolated_time = np.linspace(start_time, end_time, num_samples)
    
    # 进行线性插值
    interpolated_rms = np.interp(interpolated_time, time[valley_indices], rms[valley_indices])

    # interpolated_rms = bandpass_filter(interpolated_rms, 10, sr, sr, order=4)
    
    # 绘制插值结果
    plt.plot(interpolated_time, interpolated_rms, color='green', label='Interpolated RMS', alpha=0.5)


    # 标出波谷
    plt.scatter(time[valley_indices], rms[valley_indices], color='orange', label='Valley')
    plt.show()
    exit()


    tg_vad = TextGrid()
    tg_vad.read(r"C:\Users\User\Desktop\Praasper\data\mandarin_sent_whisper.TextGrid")
    intervals = [interval for interval in tg_vad.tiers[0] if interval.mark != ""]


    # 定义筛选规则：
    # 1. 波谷中的波谷/拐点
    # 2. 不允许左高右低
    # isFirstInterval = True
    # isLastInterval = False
    for idx, interval in enumerate(intervals):
        if idx == len(intervals) - 1:
            break
        
        # if isLastInterval:
        #     isFirstInterval = True
        #     isLastInterval = False
            
        current_interval = intervals[idx]
        next_interval = intervals[idx+1]

        midpoint = (current_interval.minTime + next_interval.maxTime) / 2

        current_con, current_vow, current_tone = extract_cvt_zh(current_interval.mark)[0]
        next_con, next_vow, next_tone = extract_cvt_zh(next_interval.mark)[0]

        print(current_interval.mark, next_interval.mark)
        if current_interval.maxTime != next_interval.minTime:
            # isLastInterval = True
            continue
            

        cand_valleys = [t for t in time[valley_indices] if current_interval.minTime + 0.01 < t < next_interval.maxTime - 0.01]
        # cand_valleys_nocon = [t for t in time[valley_indices_nocon] if current_interval.minTime + 0.01 < t < next_interval.maxTime - 0.01]

        # 获取 cand_valleys 对应的 rms 值
        cand_valleys_rms = [rms[np.where(time == t)[0][0]] for t in cand_valleys]
        # cand_valleys_rms_nocon = [rms_nocon[np.where(time == t)[0][0]] for t in cand_valleys_nocon]
        print(cand_valleys)
        print(cand_valleys_rms)


        # 获取当前函数中的采样率（假设在 plot_audio_power_curve 函数中 sr 变量可用）
        # 根据当前上下文，sr 在 plot_audio_power_curve 函数开头已定义
        # 开始时间和结束时间
        start_time = current_interval.minTime
        end_time = next_interval.maxTime
        
        # 生成插值时间点
        num_samples = int((end_time - start_time) * sr)
        interpolated_time = np.linspace(start_time, end_time, num_samples)
        
        # 进行线性插值
        if len(cand_valleys) > 0:
            interpolated_rms = np.interp(interpolated_time, cand_valleys, cand_valleys_rms)
        else:
            interpolated_rms = np.zeros(num_samples)
        
        # 绘制插值结果
        plt.plot(interpolated_time, interpolated_rms, color='green', label='Interpolated RMS', alpha=0.5)



        isNextConFlag = next_con in ["z", "s", "c", "zh", "ch", "sh", "x"]
        isCurrentConFlag = current_con in ["z", "s", "c", "zh", "ch", "sh", "x"]

        sorted_indices = np.argsort(cand_valleys_rms)
        # sorted_indices_nocon = np.argsort(cand_valleys_rms_nocon)

        print(f"Current: {isCurrentConFlag}; Next: {isNextConFlag}")
        # 找到两个最小值对应的时间中更靠前的一个
        if isNextConFlag and not isCurrentConFlag:
            # min_valley_time = sorted([valid_valleys[sorted_indices[idx_v]] for idx_v in range(2)])[0]
            valid_points = sorted([cand_valleys[sorted_indices[idx_v]] for idx_v in range(2)], key=lambda x: abs(x - midpoint))
            min_valley_time = valid_points[0]
            

        elif isNextConFlag and isCurrentConFlag:
            valid_points = sorted([cand_valleys[sorted_indices[idx_v]] for idx_v in range(3)], key=lambda x: abs(x - midpoint))
            # min_valley_time = sorted([cand_valleys[sorted_indices[idx_v]] for idx_v in range(3)])[1]
            min_valley_time = valid_points[1]
            # min_valley_time = cand_valleys[np.argmin(cand_valleys_rms)]

        elif not isNextConFlag and isCurrentConFlag:
            # min_valley_time = sorted([valid_valleys[sorted_indices[idx_v]] for idx_v in range(2)])[1]
            valid_points = sorted([cand_valleys[sorted_indices[idx_v]] for idx_v in range(3)], key=lambda x: abs(x - midpoint))
            min_valley_time = valid_points[1]
        
        else: 
            min_valley_time = cand_valleys[np.argmin(cand_valleys_rms)]
        
        print(f"最小波谷时间: {min_valley_time}")
        
        current_interval.maxTime = min_valley_time
        next_interval.minTime = current_interval.maxTime


        # for v in valid_valleys:
        plt.axvline(x=min_valley_time, color='b', label="Valid" if idx == 0 else "", alpha=0.3, linewidth=2)
        print()


    
    plt.legend()
    
    # 显示图形
    plt.tight_layout()
    plt.show()

# 使用示例
if __name__ == "__main__":
    audio_file_path = r"C:\Users\User\Desktop\Praasper\data\mandarin_sent.wav"  # 替换为实际的音频文件路径
    plot_audio_power_curve(audio_file_path)
