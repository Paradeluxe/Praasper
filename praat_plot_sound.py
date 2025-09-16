import parselmouth
import numpy as np


def get_intensity_time_series(audio_path):
    """
    获取音频文件指定时间段内的强度时间序列
    
    参数:
    audio_path -- 音频文件路径
    start_time -- 起始时间(秒)
    end_time -- 结束时间(秒)
    
    返回:
    times -- 时间点列表
    intensities -- 对应时间点的强度值列表
    """
    # 加载音频文件
    sound = parselmouth.Sound(audio_path)
    
    # 计算整个音频的强度对象
    intensity = sound.to_intensity(time_step=0.02)  # 时间步长0.01秒（可调整）
    print(intensity)

    intensity_points = np.array(intensity.as_array()).flatten()
    time_points = np.array(intensity.xs())
    
    import matplotlib.pyplot as plt


    # 绘制强度时间序列图
    plt.figure(figsize=(10, 6))
    plt.plot(time_points, intensity_points)
    plt.title('Audio Intensity Time Series')
    plt.xlabel('Time (s)')
    plt.ylabel('Intensity (dB)')
    plt.grid(True)
    plt.show()

    # 返回指定时间段内的时间点和强度值
    return selected_time_points.tolist(), selected_intensity_points.tolist()
    

# 使用示例
if __name__ == "__main__":
    # 替换为你的音频文件路径
    audio_file = r"C:\Users\User\Desktop\Praasper\data\mandarin_sent.wav"
    
    # 指定时间段 (单位：秒)
    start = 2.22
    end = 2.65
    
    # 获取强度时间序列
    times, intensities = get_intensity_time_series(audio_file)
    
    # 打印结果
    print(f"时间点(秒)\t强度(dB)")
    for t, intensity in zip(times, intensities):
        print(f"{t:.4f}\t\t{intensity:.2f}")