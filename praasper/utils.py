import time
from scipy.signal import butter, filtfilt
import numpy as np

# 记录程序开始执行的时间
START_TIME = time.time()


def show_elapsed_time():
    """显示从程序开始执行到现在的时间差"""
    elapsed = time.time() - START_TIME
    minutes = int(elapsed // 60)
    seconds = elapsed % 60
    milliseconds = int((elapsed % 1) * 1000)
    return f"{minutes:02d}:{int(seconds):02d}:{milliseconds:03d}"



def bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist

    low = max(0.01, low)
    high = min(nyquist - 0.01, high)
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
    

def compute_boundary_snr(audio_arr, sr, onsets, offsets, window_ms=10):
    """
    Compute mean SNR (dB) across all onset/offset boundaries.
    
    For each onset at time t:
        noise  = audio[t - window_ms : t]
        speech = audio[t : t + window_ms]
    For each offset at time t:
        speech = audio[t - window_ms : t]
        noise  = audio[t : t + window_ms]
    
    SNR = 10 * log10(speech_power / noise_power)
    
    Edge cases:
        - Boundary within window_ms of audio edge -> clip window
        - Zero noise power -> cap at 60 dB
        - No boundaries -> return 0.0
    """
    if not onsets and not offsets:
        return 0.0
    
    window_samples = int(window_ms / 1000.0 * sr)
    if window_samples <= 0:
        window_samples = 1
        
    total_len = len(audio_arr)
    snrs = []
    
    for t in onsets:
        center = int(t * sr)
        noise_start = max(0, center - window_samples)
        noise_end = center
        speech_start = center
        speech_end = min(total_len, center + window_samples)
        
        if noise_end <= noise_start or speech_end <= speech_start:
            continue
            
        noise_power = np.mean(audio_arr[noise_start:noise_end].astype(np.float64) ** 2)
        speech_power = np.mean(audio_arr[speech_start:speech_end].astype(np.float64) ** 2)
        
        if noise_power <= 0:
            snrs.append(60.0)
        elif speech_power <= 0:
            snrs.append(0.0)
        else:
            snrs.append(10.0 * np.log10(speech_power / noise_power))
    
    for t in offsets:
        center = int(t * sr)
        speech_start = max(0, center - window_samples)
        speech_end = center
        noise_start = center
        noise_end = min(total_len, center + window_samples)
        
        if speech_end <= speech_start or noise_end <= noise_start:
            continue
            
        speech_power = np.mean(audio_arr[speech_start:speech_end].astype(np.float64) ** 2)
        noise_power = np.mean(audio_arr[noise_start:noise_end].astype(np.float64) ** 2)
        
        if noise_power <= 0:
            snrs.append(60.0)
        elif speech_power <= 0:
            snrs.append(0.0)
        else:
            snrs.append(10.0 * np.log10(speech_power / noise_power))
    
    return float(np.mean(snrs)) if snrs else 0.0


if __name__ == '__main__':
    pass