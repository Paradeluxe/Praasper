from textgrid import IntervalTier, TextGrid

try:
    import matplotlib.pyplot as plt
except ImportError:
    pass

try:
    from utils import *
    from VAD.core_auto import *

except ImportError:
    from praasper.utils import *
    from praasper.VAD.core_auto import *
import os

import unicodedata

default_params = {'onset': {'amp': '1.47', 'cutoff0': '60', 'cutoff1': '10800', 'numValid': '475', 'eps_ratio': '0.093'}, 'offset': {'amp': '1.47', 'cutoff0': '60', 'cutoff1': '10800', 'numValid': '475', 'eps_ratio': '0.093'}}



def purify_text(text):
    """
    清理文本中的无效字符，保留所有语言的文字字符
    
    :param text: 输入的文本
    :return: 清理后的文本
    """

    text = text.strip()
    # 只删除标点符号，保留所有语言的文字字符
    text = ''.join('' if unicodedata.category(c).startswith('P') else c for c in text)
    return text


def segment_audio(audio_obj, segment_duration=10, min_pause=0.2, params="self", verbose=False):
    wav_path = audio_obj.fpath

    print(f"[{show_elapsed_time()}] ({os.path.basename(wav_path)}) Start segmentation (<= {segment_duration}s)...")


    audio_obj = ReadSound(wav_path)

    # 获取 wav 文件所在的文件夹路径
    wav_folder = os.path.dirname(wav_path)
    all_txt_path = os.path.join(wav_folder, "params.txt")
    self_txt_path = wav_path.replace(".wav", ".txt")

    

    if params == "all":
        if os.path.exists(all_txt_path):
            with open(all_txt_path, "r") as f:
                params = eval(f.read())
        else:
            params = default_params
    
    elif params == "self":
        if os.path.exists(self_txt_path):
            with open(self_txt_path, "r") as f:
                params = eval(f.read())
        else:
            params = default_params

    elif params == "default":
        params = default_params

    else:  # 具体参数
        params = params
    

    segments = []

    y = audio_obj.arr
    sr = audio_obj.frame_rate

    audio_len = len(y) /sr

    start = 0.0 * 1000
    end = segment_duration * 1000
    while end <= audio_len * 1000:
        segment = audio_obj[start:end]
        # print(type(segment) == type(audio_obj))
        onsets = autoPraditorWithTimeRange(params, segment, "onset", verbose=False)
        offsets = autoPraditorWithTimeRange(params, segment, "offset", verbose=False)
        # print()
        # print(start, end)
        # print(onsets, offsets, audio_len * 1000)
        if not onsets or not offsets:
            segments[-1][1] = end
            # continue
        else:
            # 从最后一个onset开始往前遍历
            for i in range(len(onsets)-1, 0, -1):
                current_onset = onsets[i]
                # 找到当前onset之前的最后一个offset
                prev_offset = [xset for xset in offsets if xset < current_onset][-1]
                # prev_offset = offsets[i-1]
                if current_onset - prev_offset > min_pause:
                    # print()
                    # print(start)
                    # 若差值大于min_pause，则取得他们的均值
                    # print(prev_offset, current_onset)
                    target_offset = (current_onset + prev_offset) / 2
                    end = start + target_offset * 1000
                    break
            else:
                # print(start)
                # print(onsets[-1], offsets[0])
                # print(onsets)
                # print(offsets)
                # 若所有onset和对应offset差值都不大于min_pause，则取最后offset的均值
                target_offset = (onsets[-1] + [offset for offset in offsets if offset < onsets[-1]][-1]) / 2
                end = start + target_offset * 1000


            # end = start + (target_offset + onsets[-1]) / 2 * 1000

            segments.append([start, end])

        start = end
        end = start + segment_duration * 1000

        if end > audio_len * 1000:
            if audio_len * 1000 - start > 10:
                segments[-1][1] = audio_len * 1000
                break
            else:
                segments.append([start, audio_len * 1000])
                break
    # print(segments)
    if not segments:
        segments.append([0.0, audio_len * 1000])
    
    # print(segments)
    # exit()
    return segments


def get_vad(wav_path, ori_wav_path, min_pause=0.2, params="self", if_save=False, verbose=False):
    print(f"[{show_elapsed_time()}] ({os.path.basename(wav_path)}) VAD processing started...")


    audio_obj = ReadSound(wav_path)

    # 获取 wav 文件所在的文件夹路径
    wav_folder = os.path.dirname(wav_path)
    all_txt_path = os.path.join(wav_folder, "params.txt")
    self_txt_path = ori_wav_path.replace(".wav", "_vad.txt")
    if not os.path.exists(self_txt_path):
        self_txt_path = ori_wav_path.replace(".wav", ".txt")

    
    # print(f"[{show_elapsed_time()}] ({os.path.basename(wav_path)}) VAD params: {params}")
    # print(default_params)
    if params == "all":
        if os.path.exists(all_txt_path):
            with open(all_txt_path, "r") as f:
                params = eval(f.read())
        else:
            params = default_params
    
    elif params == "self":
        if os.path.exists(self_txt_path):
            with open(self_txt_path, "r") as f:
                params = eval(f.read())
        else:
            params = default_params
    elif params == "default":
        params = default_params

    else:  # 具体参数
        params = params
    # print(params)


    onsets = autoPraditorWithTimeRange(params, audio_obj, "onset", verbose=False)
    offsets = autoPraditorWithTimeRange(params, audio_obj, "offset", verbose=False)

    if verbose:   
        print(f"[{show_elapsed_time()}] ({os.path.basename(wav_path)}) VAD onsets: {onsets}")
        print(f"[{show_elapsed_time()}] ({os.path.basename(wav_path)}) VAD offsets: {offsets}")



    if onsets[0] >= offsets[0]:
        onsets = [0.0] + onsets
    
    if offsets[-1] <= onsets[-1]:
        offsets.append(audio_obj.duration_seconds)

    # Select the one offset that is closest to onset and earlier than onset
    valid_onsets = []
    valid_offsets = []
    for i, onset in enumerate(onsets):
        # print(onset)
        if i == 0:
            valid_offsets.append(offsets[-1])
            valid_onsets.append(onset)
        else:
            try:
                valid_offsets.append(max([offset for offset in offsets if onsets[i-1] < offset < onset]))
                valid_onsets.append(onset)

            except ValueError:
                pass
    


    onsets = sorted(valid_onsets)
    offsets = sorted(valid_offsets)

    tg = TextGrid()
    interval_tier = IntervalTier(name="interval", minTime=0., maxTime=audio_obj.duration_seconds)



    bad_onsets = []
    bad_offsets = []

    for i in range(len(onsets)-1):
        if onsets[i+1] - offsets[i] < min_pause:
            bad_onsets.append(onsets[i+1])
            bad_offsets.append(offsets[i])

    onsets = [x for x in onsets if x not in bad_onsets]
    offsets = [x for x in offsets if x not in bad_offsets]
    
    for onset, offset in zip(onsets, offsets):
        interval_tier.add(onset, offset, "+")


    tg.append(interval_tier)
    tg.write(wav_path.replace(".wav", "_vad.TextGrid"))  # 将TextGrid对象写入文件

    tg = TextGrid()
    tg.read(wav_path.replace(".wav", "_vad.TextGrid"))

    if not if_save:
        os.remove(wav_path.replace(".wav", "_vad.TextGrid"))
    else:
        print(f"[{show_elapsed_time()}] ({os.path.basename(wav_path)}) VAD results saved")
    
    
    return tg


if __name__ == "__main__":
    segments = segment_audio("data/test_audio.wav", segment_duration=3.5)
    # print(segments)
