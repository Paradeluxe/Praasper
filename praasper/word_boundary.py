try:
    from .utils import *
    from .cvt import *
except ImportError:
    from utils import *
    from cvt import *

import librosa
import numpy as np
from scipy.signal import find_peaks
from textgrid import TextGrid
import parselmouth



def get_expected_num(audio_path):
    tg = TextGrid.fromFile(audio_path.replace(".wav", "_whisper.TextGrid"))
    tier = tg.tiers[0]
    words = []
    for interval in tier.intervals:
        for c in interval.mark.split():
            words.append(c)

    count = 0
    for word in words:
        c, v, t = extract_cvt(word, lang="zh")[0]
        if c in ["z", "zh", "s", "c", "ch", "sh", "x", "j"]:
            count += 1
    return count


def get_spectral_peak_interval(audio_path, whisper_tg, verbose=False):
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
    # mask = (time >= start_time) & (time <= end_time)
    # time = time[mask]
    # spectral_peaks = spectral_peaks[mask]
    
    # 定义平均池化的窗口大小
    window_size = int(0.0001 * sr)
    # 创建一个归一化的卷积核
    kernel = np.ones(window_size) / window_size
    # 使用 np.convolve 进行平均池化
    spectral_peaks = np.convolve(spectral_peaks, kernel, mode='same')


    # 方法一：获取baseline
    # basline = get_baseline_freq_peak(audio_path)  
    # 方法二：根据whisper的标注获取预期的音节数

    tg = whisper_tg
    # print(whisper_tg.tiers[0].intervals)
    tier = tg.tiers[0]
    words = []
    for interval in tier.intervals:
        for c in interval.mark.split():
            words.append(c)

    count = 0
    for word in words:
        c, v, t = extract_cvt(word, lang="zh")[0]
        if c in ["z", "zh", "s", "c", "ch", "sh", "x", "j"]:
            count += 1
    if verbose:
        print(f"[{show_elapsed_time()}] Expected number of syllables: {count}")
    expected_num = count




    ifMeetNum = None
    # 从spectral_peaks的最大值往0遍历，找到最后一个符合expected_num的baseline
    max_peak = np.max(spectral_peaks)
    step = max_peak / 100  # 设置遍历步长

    # for baseline in np.arange(max_peak, 0, -step):
    for baseline in np.arange(0, max_peak, step):
        # 找到所有大于基线的点
        above_baseline_mask = spectral_peaks > baseline
        # 找到所有大于基线的点的连续区间
        continuous_count = 0
        in_interval = False

        cand_timestamps = []
        for idx, is_above in enumerate(above_baseline_mask):
            if is_above and not in_interval:
                # continuous_count += 1
                in_interval = True
                cand_timestamps.append([time[idx]])
            elif not is_above and in_interval:
                cand_timestamps[-1].append(time[idx])
                in_interval = False
            # elif not in_interval:
                # below_frm_num += 1
        # if len(cand_timestamps) >= 1:
        #     continuous_count += 1
        # # 检查从false到true之间的距离是否够大，若够大，才考虑切换
        # for idx in range(1, len(cand_timestamps)):

        #     if cand_timestamps[idx][0] - cand_timestamps[idx-1][1] > 0.005:
        #         continuous_count += 1
        if len(cand_timestamps) == 1 and len(cand_timestamps[0]) == 1:
            cand_timestamps[-1].append(time[-1])
        continuous_count = len(cand_timestamps)

        if verbose:
            print(cand_timestamps, continuous_count, expected_num)
        # 检查最后一个区间是否以大于基线的值结束
        if len(above_baseline_mask) > 0 and above_baseline_mask[-1] and in_interval:
            pass
        # !!! 需要添加：相邻两个peak之间不能太近
        # 若连续区间数量等于预期数量，则使用当前基线
        if ifMeetNum is None:
            if continuous_count == expected_num:
                break
                ifMeetNum = True
    
        elif ifMeetNum:
            if continuous_count != expected_num:
                # print(continuous_count, expected_num, "-")
                ifMeetNum = False
                break
        
        # elif not ifMeetNum:
        #     print(continuous_count, expected_num, "!")
        #     print()
        #     break
    # print(baseline)
    # baseline -= step
    if verbose:
        print(f"Baseline: {baseline:.2f}, Continuous Count: {continuous_count}, Expected Num: {expected_num}, ifMeetNum: {ifMeetNum}")

    # 找到所有大于基线的点
    above_baseline_mask = spectral_peaks > baseline

    # 初始化存储连续区间的列表
    continuous_intervals = []
    start_index = None

    # 遍历掩码，找到连续大于基线的区间
    for idx, is_above in enumerate(above_baseline_mask):
        if is_above and start_index is None:
            start_index = idx
        elif not is_above and start_index is not None:
            continuous_intervals.append((start_index, idx))
            start_index = None

    # 处理最后一个区间
    if start_index is not None:
        continuous_intervals.append((start_index, len(above_baseline_mask)))

    # 存储每个区间最大值的索引
    # max_indices = []
    interval_indices = []
    for idx, (start, end) in enumerate(continuous_intervals):
        # interval_data = spectral_peaks[start:end]
        # max_index_in_interval = np.argmax(interval_data)
        # max_indices.append(start + max_index_in_interval)

        # 找到start和end之间的time上的所有波峰
        interval_spectral_peaks = spectral_peaks[start:end]
        peak_indices, _ = find_peaks(interval_spectral_peaks)
        peak_indices = list(sorted(peak_indices))
        if peak_indices:
            start = start + peak_indices[0]
            end = start + peak_indices[-1]

        dur = (time[end] - time[start]) * 0.  # 也可以去找最近的波峰!!!

        left = time[start] + dur
        right = time[end] - dur
        # print(idx, len(continuous_intervals) - 1)
        if idx == 0:
            interval_indices.append([0.0, left])
            interval_indices.append([right])
        elif idx == len(continuous_intervals) - 1:
            interval_indices[-1].append(left)
            interval_indices.append([right, len(y)/sr])
        else:
            interval_indices[-1].append(left)
            interval_indices.append([right])
        
    if len(interval_indices[-1]) == 1:
            interval_indices[-1].append(len(y)/sr)
    
    if not interval_indices:
        interval_indices.append([0.0, len(y)/sr])
    # print(interval_indices)
    # interval_indices = [i for i in interval_indices if min_pause < i[1] - i[0]]
    # max_indices = np.array(max_indices)
    # print(interval_indices)
    

    if verbose:
        import matplotlib.pyplot as plt

        # 创建图形
        plt.figure(figsize=(10, 4))
        
        # 绘制频谱质心曲线
        plt.plot(time, spectral_peaks, color='r')
        plt.title('Audio Spectral Centroid Curve')
        plt.xlabel('Time (s)')
        plt.ylabel('Spectral Centroid (Hz)')
        for start, end in interval_indices:
            plt.axvline(x=start, color='b', linestyle='--')
            plt.axvline(x=end, color='g', linestyle='--')
        plt.grid(True)
        
        # 显示图形
        plt.tight_layout()
        plt.show()
    # print(time[np.argmax(spectral_peaks)])

    return interval_indices


def find_internsity_valley(audio_path, start_time, end_time, verbose=False):
    sound = parselmouth.Sound(audio_path)

    # 计算整个音频的强度对象
    intensity = sound.to_intensity(time_step=0.02)  # 时间步长0.01秒（可调整）

    intensity_points = np.array(intensity.as_array()).flatten()
    time_points = np.array(intensity.xs())
    # 筛选出 current_interval.minTime 和 next_interval.maxTime 之间的时间点和对应的强度值
    mask = (time_points >= start_time) & (time_points <= end_time)
    time_points = time_points[mask]
    intensity_points = intensity_points[mask]

    # if verbose:
    #     import matplotlib.pyplot as plt

    #     # 创建图形
    #     plt.figure(figsize=(10, 4))
        
    #     # 绘制功率曲线
    #     plt.plot(time_points, intensity_points, alpha=0.3)
    #     plt.title('Audio Power Curve')
    #     plt.xlabel('Time (s)')
    #     plt.ylabel('Power')
    #     plt.grid(True)
    # 找到强度曲线的波谷索引
    intensity_valley_indices = find_peaks(-intensity_points)[0]

    # midpoint = (start_time + end_time) / 2
    # 按照距离 midpoint 的绝对距离对波谷索引排序
    # intensity_valley_indices = sorted(intensity_valley_indices, key=lambda idx: abs(time_points[idx] - midpoint))
    intensity_valley_indices = sorted(intensity_valley_indices, key=lambda idx: intensity_points[idx])
    # 获取波谷对应的时间点
    valley_times = time_points[intensity_valley_indices]
    try:
        min_valley_time = valley_times[0]
    except IndexError:
        min_valley_time = time_points[np.argmin(intensity_points)]
    
    # if verbose:
    #     plt.axvline(x=min_valley_time, color='b', linestyle='--')
    #     plt.show()

    return min_valley_time




def find_word_boundary(wav_path, whisper_tg, tar_sr=10000, min_pause=0.1, verbose=False):
    """
    绘制整段音频的功率曲线
    
    参数:
    audio_path (str): 音频文件的路径
    """
    shifted_peaks_indices = get_spectral_peak_interval(wav_path, whisper_tg, verbose=False)

    # 加载音频文件
    y, sr = read_audio(wav_path, tar_sr=tar_sr)


    y = np.gradient(y)
    # y = np.gradient(np.gradient(y))

    # y = bandpass_filter(y, 50, sr, sr, order=4)

    # 使用librosa计算音频的均方根能量(rms)
    rms = librosa.feature.rms(y=y, frame_length=128, hop_length=32, center=True)[0]

    # 计算对应的时间轴
    time = librosa.times_like(rms, sr=sr, hop_length=32)
    
    if verbose:
        import matplotlib.pyplot as plt

        # 创建图形
        plt.figure(figsize=(10, 4))
        
        # 绘制功率曲线
        plt.plot(time, rms, alpha=0.3)
        plt.title('Audio Power Curve')
        plt.xlabel('Time (s)')
        plt.ylabel('Power')
        plt.grid(True)

        vertical_line = [.688, .80, .88, 1.16, 1.25, 1.55, 1.75, 1.84, 1.94, 2.224, 2.49,
                        3.51, 3.75, 3.97, 4.10, 4.29, 4.46, 4.58, 4.72, 4.958]
        for v in vertical_line:
            plt.axvline(x=v, color='r', linestyle='--')

    # 找到波谷的索引
    valley_indices = find_peaks(-rms, width=(1, None), distance=10)[0]

    if verbose:
        plt.scatter(time[valley_indices], rms[valley_indices], color='orange', label='Valley')



    tg = whisper_tg
    intervals = [interval for interval in tg.tiers[0] if interval.mark != ""]

    for idx, interval in enumerate(intervals):
        if idx == len(intervals) - 1:
            break

            
        current_interval = intervals[idx]
        next_interval = intervals[idx+1]

        current_con, current_vow, current_tone = extract_cvt_zh(current_interval.mark)[0]
        next_con, next_vow, next_tone = extract_cvt_zh(next_interval.mark)[0]

        if current_interval.maxTime != next_interval.minTime:
            continue
            

        cand_valleys = [t for t in time[valley_indices] if current_interval.minTime + 0.05 < t < next_interval.maxTime]
        cand_valleys_rms = [rms[np.where(time == t)[0][0]] for t in cand_valleys]


        # 将候选波谷时间转换为 numpy 数组以便后续操作
        valid_valleys = np.array(cand_valleys)
        valid_valleys_rms = np.array(cand_valleys_rms)
        

        isNextConFlag = next_con in ["z", "zh", "s", "c", "ch", "sh", "x", "j"]
        isCurrentConFlag = current_con in ["z", "zh", "s", "c", "ch", "sh", "x", "j"]

        sorted_indices = np.argsort(valid_valleys_rms)

        best_dur = 0
        left_boundary = None
        right_boundary = None
        print(shifted_peaks_indices)
        for start, end in shifted_peaks_indices:
            if isCurrentConFlag and isNextConFlag:
                if (current_interval.minTime <= start <= next_interval.maxTime) and (current_interval.minTime <= end <= next_interval.maxTime):
                    dur = min(end, next_interval.maxTime) - max(start, current_interval.minTime)

                    if dur > best_dur:
                        best_dur = dur

                        left_boundary = max(start, current_interval.minTime)
                        right_boundary = min(end, next_interval.maxTime)
            
            elif isCurrentConFlag and not isNextConFlag:
                if current_interval.minTime <= start <= next_interval.maxTime:
                
                    dur = next_interval.maxTime - max(start, current_interval.minTime)

                    if dur > best_dur:
                        best_dur = dur

                        left_boundary = max(start, current_interval.minTime)
                        right_boundary = next_interval.maxTime

            
            elif not isCurrentConFlag and isNextConFlag:
                if current_interval.minTime <= end <= next_interval.maxTime:
                    dur = min(end, next_interval.maxTime) - current_interval.minTime

                    if dur > best_dur:
                        best_dur = dur

                        left_boundary = current_interval.minTime
                        right_boundary = min(end, next_interval.maxTime)
            
        if left_boundary is None or right_boundary is None:
            left_boundary = current_interval.minTime
            right_boundary = next_interval.maxTime
        
        try:
            if intervals[idx+1].maxTime != intervals[idx+2].minTime and not isNextConFlag:
                right_boundary -= min_pause
                # print("触发最后一个intervalinterval")
        except IndexError:
            pass

        # valid_valleys = []
        # valley_valleys_rms = []
        mask = (valid_valleys >= left_boundary) & (valid_valleys <= right_boundary)

        while not valid_valleys[mask].any():
            right_boundary += 0.001
            left_boundary -= 0.001
            mask = (valid_valleys >= left_boundary) & (valid_valleys <= right_boundary)
        
        valid_valleys = valid_valleys[mask]
        valid_valleys_rms = valid_valleys_rms[mask]

        sorted_indices = np.argsort(valid_valleys_rms)
        
        if verbose:
            print(current_interval.mark, next_interval.mark)
            print(current_interval.minTime, next_interval.maxTime)
            print(left_boundary, right_boundary)


        if isNextConFlag or isCurrentConFlag:# or next_con in ["h", "d", "t", "k", "p", "f", "g", "n", "m", "b", "l"]:
            if next_con:
            # if isNextConFlag:
            #     valid_points = valid_valleys
            #     min_valley_time = valid_points[-1]
            # elif next_con and not isNextConFlag:
                try:
                    valid_points = [valid_valleys[sorted_indices[idx_v]] for idx_v in range(1)]#, key=lambda x: abs(x - midpoint))
                    if verbose:
                        print(valid_valleys[sorted_indices])
                        print(f"找到最小值{valid_points}")
                    # if valid_points:
                    #     valid_points = [valid_valleys[np.argmin(valid_valleys_rms)]]
                    # else:
                    if not valid_points:
                        valid_points = valid_valleys
                except IndexError:
                    valid_points = valid_valleys
                # if isCurrentConFlag and not isNextConFlag:
                #     min_valley_time = valid_points[0]
                if not isCurrentConFlag and isNextConFlag:
                    # min_valley_time = valid_valleys[-1]
                    min_valley_time = valid_points[0]

                else:
                    min_valley_time = valid_points[0]
            else:
                min_valley_time = find_internsity_valley(wav_path, left_boundary, right_boundary, verbose=verbose)


        else: # 两个都不是
            # if not next_con:
            #     min_valley_time = find_internsity_valley(audio_path, left_boundary, right_boundary, verbose=False)
            # else:
            min_valley_time = find_internsity_valley(wav_path, left_boundary, right_boundary, verbose=verbose)
        if verbose:
            print(min_valley_time)
            print()
        
        
        # elif next_con in ["k", "t", "p"]:
        #     min_valley_time = valid_valleys[np.argmin(valid_valleys_rms)]
        
        # elif not next_con or next_con in ["h"]:
        #     min_valley_time = find_internsity_valley(audio_path, current_interval.minTime, next_interval.maxTime)


        
        # print(f"最小波谷时间: {min_valley_time}")
        
        current_interval.maxTime = min_valley_time
        next_interval.minTime = current_interval.maxTime

        if verbose:
            # for v in valid_valleys:
            plt.axvline(x=min_valley_time, color='b', label="Valid" if idx == 0 else "", alpha=0.3, linewidth=2)
            # print()



    if verbose:
        # plt.legend()
        
        # 显示图形
        plt.tight_layout()
        plt.show()

        # tg.write(tg_path.replace("_whisper.TextGrid", "_whisper_recali.TextGrid"))
    return tg

def get_baseline_freq_peak(audio_path):
    y, sr = read_audio(audio_path)
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
    
    
    vad_path = audio_path.replace(".wav", "_VAD.TextGrid")
    tg = TextGrid()
    tg.read(vad_path)
    
    vad_tiers = [(i.minTime, i.maxTime) for i in tg.tiers[0] if i.mark == ""]
    freq_peaks = []
    for start_time, end_time in vad_tiers:
        mask = (time >= start_time) & (time <= end_time)
        time = time[mask]
        spectral_peaks = spectral_peaks[mask]

        freq_peaks += list(spectral_peaks)
    return np.mean(freq_peaks) * 2.5
    
    


# 使用示例
if __name__ == "__main__":
    # audio_file_path = r"C:\Users\User\Desktop\Praasper\data\mandarin_sent.wav" 
    # audio_file_path = r"C:\Users\User\Desktop\Praasper\data\test_audio.wav" 
    # audio_file_path = r"C:\Users\User\Desktop\Praasper\data\man_clip.wav" 
    audio_file_path = r"C:\Users\User\Desktop\Praasper\data\mandarin_sent_temp.wav" 

    # min_valley_time = find_internsity_valley(audio_file_path, 0, 8.0, verbose=True)
    # peak = find_spec_peak(audio_file_path, verbose=False)
    # get_baseline_freq_peak(audio_file_path)
    # exit()
    find_word_boundary(audio_file_path, tar_sr=12000, verbose=True)
