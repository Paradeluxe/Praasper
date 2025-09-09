import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import convolve2d, find_peaks
from textgrid import TextGrid, PointTier, IntervalTier
import os



def is_in_vad_interval(time_point, tg):
    """
    检查时间点是否在任意一个VAD区间内
    
    参数:
    time_point (float): 待检查的时间点
    tg (TextGrid): TextGrid对象
    
    返回:
    bool: 如果时间点在VAD区间内返回True，否则返回False
    """
    for tier in tg:
        if isinstance(tier, type(tg[0])) and hasattr(tier, 'intervals'):
            for interval in tier.intervals:
                if interval.mark.strip() != "" and interval.minTime <= time_point <= interval.maxTime:
                    return True
    return False

def plot_spectrum(audio_path, vad_path):
    """
    绘制目标音频的频谱图
    
    参数:
    audio_path (str): 音频文件的路径
    """

    # 根据vad_path，用textgrid库读取所有mark非空的interval到一个list
    vad_intervals = []
    try:
        tg = TextGrid.fromFile(vad_path)
        for tier in tg:
            if isinstance(tier, IntervalTier) and tier.name == "word":
                for interval in tier.intervals:
                    if interval.mark.strip() != "":
                        vad_intervals.append(interval)
    except FileNotFoundError:
        print(f"未找到文件: {vad_path}")



    # 加载音频文件
    y, sr = librosa.load(audio_path, sr=8000)

    # 创建一个新的IntervalTier
    max_time = librosa.core.get_duration(y=y, sr=sr)
    # point_tier = PointTier('Valley Points', minTime=0, maxTime=max_time)
    # 创建一个新的TextGrid对象
    new_tg = TextGrid()
    new_tg.minTime = 0
    new_tg.maxTime = max_time
    interval_tier = IntervalTier('PV Intervals', minTime=0, maxTime=max_time)

    for vad_interval in vad_intervals:
        print(vad_interval)
        # 提取当前VAD区间的音频
        start_sample = int(vad_interval.minTime * sr)
        end_sample = int(vad_interval.maxTime * sr)
        y_vad = y[start_sample:end_sample]
    
        # 计算频谱图
        spectrogram = librosa.stft(y_vad, n_fft=2048, win_length=1024, center=True)
        spectrogram_db = librosa.amplitude_to_db(abs(spectrogram), ref=1.0)  # 使用librosa.amplitude_to_db已将y值转换为对数刻度，top_db=None确保不限制最大分贝值
        
        # 创建图形
        plt.figure(figsize=(10, 4))
        
        # 绘制频谱图
        kernel = np.array([[-1, 0, 1]])
        convolved_spectrogram = convolve2d(spectrogram_db, kernel, mode='same', boundary='symm')
        convolved_spectrogram = np.where(np.abs(convolved_spectrogram) < 20, 0, convolved_spectrogram)

    

        # 按频率轴求和，保持维度以方便后续绘图
        convolved_spectrogram = np.sum(np.abs(convolved_spectrogram), axis=0, keepdims=False)
        # convolved_spectrogram = np.gradient(convolved_spectrogram)
        time_axis = np.linspace(0, len(convolved_spectrogram) * librosa.core.get_duration(y=y_vad, sr=sr) / len(convolved_spectrogram), len(convolved_spectrogram))

    
        # 找到所有的波峰和波谷
        peaks, _ = find_peaks(convolved_spectrogram)#, prominence=(10, None))
        valleys, _ = find_peaks(-convolved_spectrogram)#, prominence=(10, None))

        # 提取波峰和波谷对应的时间和值
        peak_times = time_axis[peaks]
        peak_values = convolved_spectrogram[peaks]

        valley_times = time_axis[valleys]
        valley_values = convolved_spectrogram[valleys]


        # 只保留波峰和波谷绝对值大于100的点
        valid_peaks = peaks[np.abs(convolved_spectrogram[peaks]) > 0]
        valid_valleys = valleys[np.abs(convolved_spectrogram[valleys]) > 0]

        # 提取有效波峰和波谷对应的时间和值
        peak_times = time_axis[valid_peaks]
        peak_values = convolved_spectrogram[valid_peaks]

        valley_times = time_axis[valid_valleys]
        valley_values = convolved_spectrogram[valid_valleys]



        plt.plot(time_axis, convolved_spectrogram)

        # 绘制VAD范围内的波谷
        plt.plot(valley_times, valley_values, "go", label="Valley")

        # 绘制VAD范围内的波峰
        plt.plot(peak_times, peak_values, "ro", label="Peak")

        plt.tight_layout()
        # 添加图例
        plt.legend()

        os.makedirs("pic", exist_ok=True)
        plt.savefig(os.path.join("pic", f"spec_{vad_interval.mark.strip()}.png"))
        plt.close()

        
        # valley_times = []

        pv_times = [0] + list(peak_times) + list(valley_times) + [vad_interval.maxTime - vad_interval.minTime]
        pv_times.sort()


        for t, time_stamp in enumerate(pv_times):
            if t == 0:
                continue
            # print(pv_times[t-1] + vad_interval.minTime, pv_times[t] + vad_interval.minTime, 'pv')
            interval_tier.add(pv_times[t-1] + vad_interval.minTime, pv_times[t] + vad_interval.minTime, 'pv')

    

    

    new_tg.append(interval_tier)
    
    # 尝试加载C:\Users\User\Desktop\Praasper\final.TextGrid文件
    final_tg_path = r"C:\Users\User\Desktop\Praasper\data\test_audio.TextGrid"
    try:
        final_tg = TextGrid.fromFile(final_tg_path)
        # 将final.TextGrid的所有tiers添加到新的TextGrid对象中
        for tier in final_tg:
            tier.name = "segment"
            new_tg.append(tier)
    except FileNotFoundError:
        print(f"未找到文件: {final_tg_path}，跳过添加该文件的tiers")
    

    # 构造保存路径，在原音频路径基础上添加_valley_points后缀
    save_path = os.path.splitext(audio_path)[0] + '_valley_points.TextGrid'
    # 保存TextGrid文件
    # new_tg.write(save_path)



# 使用示例
if __name__ == "__main__":
    audio_path = r"data\test_audio.wav"  # 替换为实际音频文件路径
    vad_path = r"data\test_audio_processed.TextGrid"  # 替换为实际VAD文件路径
    plot_spectrum(audio_path, vad_path)
