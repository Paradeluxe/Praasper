import numpy as np
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
        step = max(1, num_segments // 2)  # step 为窗口一半，最小为1
        for i in range(0, num_segments, step):
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

