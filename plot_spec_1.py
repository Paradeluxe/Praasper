import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import argrelextrema, butter, filtfilt, find_peaks
from textgrid import TextGrid
import pypinyin
import parselmouth
import numpy as np
from scipy.ndimage import uniform_filter1d
from pydub import AudioSegment


def read_audio(audio_path):

    audio = AudioSegment.from_wav(audio_path).split_to_mono()[0]
    # y = audio.split_to_mono()[0]
    y = audio.get_array_of_samples()
    # 将Python列表转换为numpy数组
    y = np.array(y, dtype=np.float32)
    print(y.shape)
    sr = audio.frame_rate

    return y, sr



def find_internsity_valley(audio_path, start_time, end_time):
    sound = parselmouth.Sound(audio_path)

    # 计算整个音频的强度对象
    intensity = sound.to_intensity(time_step=0.02)  # 时间步长0.01秒（可调整）

    intensity_points = np.array(intensity.as_array()).flatten()
    time_points = np.array(intensity.xs())
    # 筛选出 current_interval.minTime 和 next_interval.maxTime 之间的时间点和对应的强度值
    mask = (time_points >= start_time) & (time_points <= end_time)
    time_points = time_points[mask]
    intensity_points = intensity_points[mask]
    # 找到强度曲线的波谷索引
    intensity_valley_indices = find_peaks(-intensity_points)[0]

    midpoint = (start_time + end_time) / 2
    # 按照距离 midpoint 的绝对距离对波谷索引排序
    intensity_valley_indices = sorted(intensity_valley_indices, key=lambda idx: abs(time_points[idx] - midpoint))
    # 获取波谷对应的时间点
    valley_times = time_points[intensity_valley_indices]

    min_valley_time = valley_times[0]

    return min_valley_time






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


def find_spec_peak(audio_path, start_time, end_time, if_plot=False:
    """
    绘制音频的频谱质心曲线
    
    参数:
    audio_path (str): 音频文件的路径
    """
    # 加载音频文件
    y, sr = read_audio(audio_path)
    # print(y.shape, sr)
    
    
    # 计算频谱质心
    # 计算每个帧的频谱
    stft = librosa.stft(y)#, win_length=win_length, hop_length=hop_length)
    # 计算每个帧的幅度谱
    magnitude = np.abs(stft)
    # 找到每个帧中频谱幅度最大的索引
    max_magnitude_indices = np.argmax(magnitude, axis=0)
    # 将索引转换为对应的频率
    spectral_peaks = librosa.fft_frequencies(sr=sr)[max_magnitude_indices]
    
    # 计算对应的时间轴
    time = librosa.times_like(spectral_peaks, sr=sr)#, hop_length=hop_length)



    # 筛选出 start_time 到 end_time 之间的数据
    mask = (time >= start_time) & (time <= end_time)
    time = time[mask]
    spectral_peaks = spectral_peaks[mask]
    
    # 定义平均池化的窗口大小
    window_size = int(0.0001 * sr)
    # 创建一个归一化的卷积核
    kernel = np.ones(window_size) / window_size
    # 使用 np.convolve 进行平均池化
    spectral_peaks = np.convolve(spectral_peaks, kernel, mode='same')
    
    if if_plot:
        # 创建图形
        plt.figure(figsize=(10, 4))
        
        # 绘制频谱质心曲线
        plt.plot(time, spectral_peaks, color='r')
        plt.title('Audio Spectral Centroid Curve')
        plt.xlabel('Time (s)')
        plt.ylabel('Spectral Centroid (Hz)')
        plt.axvline(x=time[np.argmax(spectral_peaks)], color='r', linestyle='--')
        plt.grid(True)
        
        # 显示图形
        plt.tight_layout()
        plt.show()
    print(time[np.argmax(spectral_peaks)])
    return time[np.argmax(spectral_peaks)]




def plot_audio_power_curve(audio_path, tg_path, tar_sr=10000):
    """
    绘制整段音频的功率曲线
    
    参数:
    audio_path (str): 音频文件的路径
    """
    # 加载音频文件
    y, sr = read_audio(audio_path)


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
    plt.plot(time, rms, alpha=0.3)
    plt.title('Audio Power Curve')
    plt.xlabel('Time (s)')
    plt.ylabel('Power')
    plt.grid(True)

    vertical_line = [.688, .80, .88, 1.16, 1.25, 1.55, 1.75, 1.84, 1.94, 2.22, 2.49,
                    3.51, 3.75, 3.97, 4.10, 4.29, 4.46, 4.58, 4.72, 4.958]
    for v in vertical_line:
        plt.axvline(x=v, color='r', linestyle='--')

    # 找到波谷的索引
    valley_indices = find_peaks(-rms, width=(1, None), distance=1)[0]



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
    # plt.show()
    # exit()

    # 找到所有的rms[valley_indices]中的valley中的valley
    # 先获取波谷对应的rms值
    valley_rms = rms[valley_indices]
    # 再在这些波谷值中寻找极小值点，即波谷中的波谷
    valley_valleys_indices = find_peaks(-valley_rms, width=(0, None), distance=1)[0]
    # 获取对应的原始时间索引
    valley_valleys_original_indices = valley_indices[valley_valleys_indices]
    # 获取对应的时间和rms值
    valley_valleys_time = time[valley_valleys_original_indices]
    valley_valleys_rms = rms[valley_valleys_original_indices]

    # 绘制波谷中的波谷
    plt.scatter(valley_valleys_time, valley_valleys_rms, color='red', label='Valley of Valleys')



    tg = TextGrid()
    tg.read(tg_path)
    intervals = [interval for interval in tg.tiers[0] if interval.mark != ""]



    for idx, interval in enumerate(intervals):
        if idx == len(intervals) - 1:
            break

            
        current_interval = intervals[idx]
        next_interval = intervals[idx+1]

        midpoint = (current_interval.minTime + next_interval.maxTime) / 2

        current_con, current_vow, current_tone = extract_cvt_zh(current_interval.mark)[0]
        next_con, next_vow, next_tone = extract_cvt_zh(next_interval.mark)[0]

        print(current_interval.mark, next_interval.mark)
        if current_interval.maxTime != next_interval.minTime:
            continue
            

        cand_valleys = [t for t in time[valley_indices] if current_interval.minTime + 0.05 < t < next_interval.maxTime - 0.05]
        cand_valleys_rms = [rms[np.where(time == t)[0][0]] for t in cand_valleys]

        # print(cand_valleys)
        # print(cand_valleys_rms)

        # 将候选波谷时间转换为 numpy 数组以便后续操作
        cand_valleys = np.array(cand_valleys)
        cand_valleys_rms = np.array(cand_valleys_rms)
        
        # 若候选波谷数量少于3个，无法形成波谷，直接使用原数据
        if len(cand_valleys_rms) < 3:
            valid_valleys = cand_valleys
            valid_valleys_rms = cand_valleys_rms
        else:
            # 找到 cand_valleys_rms 中的波谷索引
            valley_indices_in_cand = find_peaks(-cand_valleys_rms)[0]#, width=(1, None), distance=1)[0]
            # 获取对应的波谷时间和 rms 值
            valid_valleys = cand_valleys[valley_indices_in_cand]
            valid_valleys_rms = cand_valleys_rms[valley_indices_in_cand]
        
        if not valid_valleys.any():
            valid_valleys = cand_valleys
            valid_valleys_rms = cand_valleys_rms

        # print(valid_valleys)
        # print(valid_valleys_rms)


        isNextConFlag = next_con in ["z", "zh", "s", "c", "ch", "sh", "x", "j"]
        isCurrentConFlag = current_con in ["z", "zh", "s", "c", "ch", "sh", "x", "j"]

        sorted_indices = np.argsort(valid_valleys_rms)
        # sorted_indices_nocon = np.argsort(cand_valleys_rms_nocon)

        print(f"Current: {isCurrentConFlag}; Next: {isNextConFlag}")
        if next_con not in ["k", "t", "p"] and next_con:

            if isNextConFlag and not isCurrentConFlag:
                # min_valley_time = sorted([valid_valleys[sorted_indices[idx_v]] for idx_v in range(2)])[0]
                mid_peak = find_spec_peak(audio_path, current_interval.minTime, next_interval.maxTime, if_plot=False)
                try:
                    valid_points = [valid_valleys[idx] for idx, time in enumerate(valid_valleys) if time < mid_peak]
                    if valid_points:
                        valid_points = [valid_valleys[np.argmin(valid_valleys_rms[valid_valleys < mid_peak])]]
                    else:
                        valid_points = valid_valleys
                    
                except:
                    valid_points = valid_valleys
                min_valley_time = valid_points[0]
                

            elif isNextConFlag and isCurrentConFlag:
                try:
                    valid_points = sorted([valid_valleys[sorted_indices[idx_v]] for idx_v in range(3)], key=lambda x: abs(x - midpoint))
                except:
                    valid_points = valid_valleys
                
                try:
                    min_valley_time = valid_points[1]
                except IndexError:
                    min_valley_time = valid_valleys[0]

            elif not isNextConFlag and isCurrentConFlag:
                # min_valley_time = sorted([valid_valleys[sorted_indices[idx_v]] for idx_v in range(2)])[1]
                try:
                    valid_points = sorted([valid_valleys[sorted_indices[idx_v]] for idx_v in range(2)], key=lambda x: abs(x - midpoint))
                except IndexError:
                    valid_points = valid_valleys
                # valid_points = sorted(valid_points)
                print(valid_points)

                min_valley_time = valid_points[0]
            
            else:
                min_valley_time = find_internsity_valley(audio_path, current_interval.minTime - 0.01, next_interval.maxTime)

        
        
        elif next_con in ["k", "t", "p"]:
            min_valley_time = valid_valleys[np.argmin(valid_valleys_rms)]
        
        elif not next_con:
            min_valley_time = find_internsity_valley(audio_path, current_interval.minTime, next_interval.maxTime)


        
        print(f"最小波谷时间: {min_valley_time}")
        
        current_interval.maxTime = min_valley_time
        next_interval.minTime = current_interval.maxTime


        # for v in valid_valleys:
        plt.axvline(x=min_valley_time, color='b', label="Valid" if idx == 0 else "", alpha=0.3, linewidth=2)
        print()


    tg.write(tg_path.replace("_whisper.TextGrid", "_whisper_recali.TextGrid"))
    plt.legend()
    
    # 显示图形
    plt.tight_layout()
    plt.show()

# 使用示例
if __name__ == "__main__":
    audio_file_path = r"C:\Users\User\Desktop\Praasper\data\mandarin_sent.wav" 
    # audio_file_path = r"C:\Users\User\Desktop\Praasper\data\test_audio.wav" 

    # peak = find_spec_peak(audio_file_path, 0., 7.74, if_plot=True)
    # exit()
    tg_path = audio_file_path.replace(".wav", "_whisper.TextGrid")
    plot_audio_power_curve(audio_file_path, tg_path, tar_sr=10000)
