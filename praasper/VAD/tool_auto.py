import os.path
import sys
from datetime import datetime

import numpy as np
from scipy.signal import butter, filtfilt
from textgrid import TextGrid
# import soundfile as sf
import librosa


class ReadSound:
    def __init__(self, fpath=None, arr=None, duration_seconds=None, frame_rate=None):

        self.fpath = fpath
        if fpath is None:
            if arr is None:
                raise ValueError("Need audio input. Receive None.")
            else:
                self.arr = arr
                self.duration_seconds = duration_seconds
                self.frame_rate = frame_rate

        else:  # 如果有fpath
            if arr is not None:
                self.arr = arr
                self.duration_seconds = duration_seconds
                self.frame_rate = frame_rate
            else:
                self.arr, self.frame_rate = librosa.load(fpath, sr=None, dtype=np.float32)
                # self.arr = (self.arr * 32767).astype('int16')  # 转换为 int16 类型
                self.duration_seconds = librosa.get_duration(y=self.arr, sr=self.frame_rate)



        try:
            self.arr = self.arr[:, 0]
        except IndexError:
            pass

        self.max = np.max(np.abs(self.arr))

    def __getitem__(self, ms):


        start = int(ms.start * self.frame_rate / 1000) if ms.start is not None else 0
        end = int(ms.stop * self.frame_rate / 1000) if ms.stop is not None else len(self.arr)

        start = min(start, len(self.arr))
        end = min(end, len(self.arr))

        return ReadSound(fpath=self.fpath, arr=self.arr[start:end], duration_seconds=(end - start) / self.frame_rate, frame_rate=self.frame_rate)

    def power(self):
        """
        计算整段信号的平均功率
        
        :return: 整段信号的平均功率
        """
        return np.mean(self.arr ** 2)
    
    def min_power_segment(self, segment_duration=1.0):
        """
        计算整段信号中每个segment_duration时长的最小功率
        
        :param segment_duration: 每个segment的时长，单位为秒
        :return: 每个segment的最小功率数组
        """
        if segment_duration > self.duration_seconds:
            # print("Segment duration must be shorter than audio duration.")
            segment_duration = self.duration_seconds
        num_segments = int(self.duration_seconds // segment_duration)
        segment_powers = []
        for i in range(num_segments):
            start = int(i * segment_duration * self.frame_rate)
            end = int((i + 1) * segment_duration * self.frame_rate)
            segment = self.arr[start:end]
            # ReadSound(fpath=self.fpath, arr=self.arr[start:end], duration_seconds=(end - start) / self.frame_rate, frame_rate=self.frame_rate)
            segment_powers.append(np.min(segment ** 2))
            timestamps = [[start, end]]
        # 找到最小功率的索引
        min_power_idx = np.argmin(segment_powers)
        start, end = timestamps[min_power_idx]
        # 返回最小功率段的 ReadSound 对象
        return ReadSound(
            fpath=self.fpath,
            arr=self.arr[start:end],
            duration_seconds=(end - start) / self.frame_rate,
            frame_rate=self.frame_rate
        )


    def get_array_of_samples(self):
        return self.arr
    
    def save(self, fpath):
        """
        使用 soundfile 保存音频文件

        :param fpath: 保存音频文件的路径
        """
        import soundfile as sf
        sf.write(fpath, self.arr, self.frame_rate)

    def __add__(self, other):
        """
        将两个ReadSound对象相加
        
        :param other: 另一个ReadSound对象
        :return: 新的ReadSound对象，包含相加后的音频数据
        """
        if not isinstance(other, ReadSound):
            raise TypeError("Only ReadSound objects can be added together.")
        
        # 检查采样率是否相同
        if self.frame_rate != other.frame_rate:
            # 如果采样率不同，将other重采样到self的采样率
            other_resampled = librosa.resample(
                other.arr, 
                orig_sr=other.frame_rate, 
                target_sr=self.frame_rate
            )
            other_arr = other_resampled
        else:
            other_arr = other.arr
        
        # 将两个音频数组相加
        combined_arr = np.concatenate([self.arr, other_arr])
        
        # 计算新的时长
        combined_duration = self.duration_seconds + other.duration_seconds
        
        # 创建新的ReadSound对象
        return ReadSound(
            fpath=None,
            arr=combined_arr,
            duration_seconds=combined_duration,
            frame_rate=self.frame_rate
        )

    def reverse(self):
        """
        反转音频数据的时间顺序
        
        :return: 新的ReadSound对象，包含反转后的音频数据
        """
        # 使用numpy的flip函数反转数组
        reversed_arr = np.flip(self.arr)
        
        # 创建新的ReadSound对象
        return ReadSound(
            fpath=None,
            arr=reversed_arr,
            duration_seconds=self.duration_seconds,
            frame_rate=self.frame_rate
        )
    



def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)


def get_current_time():
    # 获取当前时间
    now = datetime.now()
    # 格式化时间字符串
    formatted_time = now.strftime("%H:%M:%S.%f")[:-3]  # 去掉微秒部分的最后3个字符
    return formatted_time


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


def lowpass_filter(data, highcut, fs, order=4):
    nyquist = 0.5 * fs
    high = highcut / nyquist
    b, a = butter(order, high, btype='low', output="ba")
    filtered_data = filtfilt(b, a, data)
    return filtered_data


def isAudioFile(fpath):
    # 所有的音频后缀
    audio_extensions = [
        '.mp3',  # MPEG Audio Layer-3
        '.wav',   # Waveform Audio File Format
        ".WAV",
        '.ogg',   # Ogg
        '.flac',  # Free Lossless Audio Codec
        '.aac',   # Advanced Audio Codec
        '.m4a',   # MPEG-4 Audio Layer
        '.alac',  # Apple Lossless Audio Codec
        '.aiff',  # Audio Interchange File Format
        '.au',    # Sun/NeXT Audio File Format
        '.aup',   # Audio Unix/NeXT
        '.ra',    # RealAudio
        '.ram',   # RealAudio Metafile
        '.rv64',  # Raw 64-bit float (AIFF/AIFF-C)
        '.spx',   # Ogg Speex
        '.voc',   # Creative Voice
        '.webm',  # WebM (audio part)
        '.wma',   # Windows Media Audio
        '.xm',    # FastTracker 2 audio module
        '.it',    # Impulse Tracker audio module
        '.mod',   # Amiga module (MOD)
        '.s3m',   # Scream Tracker 3 audio module
        '.mtm',   # MultiTracker audio module
        '.umx',   # FastTracker 2 extended module
        '.dxm',   # Digital Tracker (DTMF) audio module
        '.f4a',   # FAudio (FMOD audio format)
        '.opus',  # Opus Interactive Audio Codec
    ]
    if any(fpath.endswith(ext) for ext in audio_extensions):
        return True
    else:
        return False


def get_frm_points_from_textgrid(audio_file_path):

    audio_dir = os.path.dirname(os.path.abspath(audio_file_path))
    audio_filename = os.path.splitext(os.path.basename(audio_file_path))[0]
    tg_file_path = os.path.join(audio_dir, audio_filename + ".TextGrid")
    if not os.path.exists(tg_file_path):
        return {"onset":[], "offset": []}
    tg = TextGrid(tg_file_path)
    tg.read(tg_file_path)
    dict_tg_time = {}
    for tier in tg.tiers:
        if tier.name == "interval":
            continue
        dict_tg_time[tier.name] = [p.time for p in tier]
    return dict_tg_time


def get_frm_points_from_01(file_path):
    if not os.path.exists(file_path):
        return None
    tg = TextGrid(file_path)
    tg.read(file_path)
    dict_tg = {}

    for tier in tg.tiers:
        dict_tg["onset"] = [p.minTime for p in tier if p.mark == "1"]
        dict_tg["offset"] = [p.maxTime for p in tier if p.mark == "1"]

    return dict_tg


def get_interval(file_path):
    tg = TextGrid(file_path)
    tg.read(file_path)

    tg_onsets = []
    tg_offsets = []

    for tier in tg.tiers:
        if tier.name == "onset":
            tg_onsets = [p.time * 1000 for p in tier]
        else:
            tg_offsets = [p.time * 1000 for p in tier]

    return list(zip(tg_onsets, tg_offsets))


def get_time(file_path):
    # 加载TextGrid文件
    tg = TextGrid(file_path)
    tg.read(file_path)
    # 找到名为'onset'的PointTier
    point_tier_onset = None

    for tier in tg.tiers:
        if tier.name == 'onset':
            point_tier_onset = tier
            break

    # 如果找到了PointTier，获取第一个点的时间
    if point_tier_onset:
        return point_tier_onset.points[0].time
        # print(f"The time of the first point in 'onset' tier is: {first_point_time}")
    else:
        return None
