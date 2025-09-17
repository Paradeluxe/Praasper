import time
import librosa
try:
    from .cvt import *
except ImportError:
    from cvt import *

# 记录程序开始执行的时间
START_TIME = time.time()

def show_elapsed_time():
    """显示从程序开始执行到现在的时间差"""
    elapsed = time.time() - START_TIME
    minutes = int(elapsed // 60)
    seconds = elapsed % 60
    milliseconds = int((elapsed % 1) * 1000)
    return f"{minutes:02d}:{int(seconds):02d}:{milliseconds:03d}"


def read_audio(audio_path):
    y, sr = librosa.load(audio_path, mono=False)

    if y.ndim >= 2:
        y = y[0]
        
    return y, sr


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



def extract_cvt(character, lang):
    if lang == 'zh':
        return extract_cvt_zh(character)
    elif lang == "yue":
        return extract_cvt_yue(character)
    else:
        raise ValueError(f"{show_elapsed_time()}Language {lang} not yet supported.")


if __name__ == '__main__':
    print(extract_cvt('我', 'zh'))