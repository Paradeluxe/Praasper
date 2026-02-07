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
from tqdm import tqdm

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


def segment_audio(audio_obj, segment_duration=10, min_pause=0.2, params="folder", verbose=False, file_info=""):
    wav_path = audio_obj.fpath
    if verbose:
        print(f"[{show_elapsed_time()}] ({os.path.basename(wav_path)}) Start segmentation (<= {segment_duration}s)...")


    audio_obj = ReadSound(wav_path)

    folder_param_path = os.path.join(os.path.dirname(wav_path), "params_vad.txt")
    file_txt_path = wav_path.replace(".wav", "_vad.txt")

    match params:
        case "file":
            if os.path.exists(file_txt_path):
                with open(file_txt_path, "r") as f:
                    params = eval(f.read())
            else:
                params = default_params
        case "folder":
            if os.path.exists(folder_param_path):
                with open(folder_param_path, "r") as f:
                    params = eval(f.read())
            else:
                params = default_params
        case "default":
            params = default_params
        case _:  # 最好直接输入dict
            if type(params) == dict:
                params = params
            elif type(params) == str:
                params = eval(params)
            else:
                params = default_params
    
    params["offset"] = params["onset"]  # VAD特供

    segments = []

    y = audio_obj.arr
    sr = audio_obj.frame_rate

    audio_len = len(y) /sr

    start = 0.0 * 1000
    end = segment_duration * 1000
    total_length = audio_len * 1000
    
    desc = f"{file_info} Segmenting..."
    with tqdm(total=total_length, desc=desc, unit="%", bar_format="{l_bar}{bar}", leave=False) as pbar:
        while end <= total_length:
            segment = audio_obj[start:end]
            # print(type(segment) == type(audio_obj))
            onsets = autoPraditorWithTimeRange(params, segment, "onset", verbose=False)
            offsets = autoPraditorWithTimeRange(params, segment, "offset", verbose=False)
            # print()
            # print(start, end)
            # print(onsets, offsets, total_length)
            if not onsets or not offsets:
                try:
                    segments[-1][1] = end
                except IndexError:
                    pass
                # continue
            else:

                ################
                # 找到一个尽可能大的pause，使得onset和offset之间的差值大于pause
                ################

                # 从最后一个onset开始往前遍历
                tmp_pause = min_pause
                found = False
                while tmp_pause > 0 and not found:
                    for i in range(len(onsets)-1, 0, -1):
                        current_onset = onsets[i]

                        # 找到当前onset之前的最后一个offset
                        try:
                            prev_offset = [offset for offset in offsets if offset < current_onset][-1]
                        except IndexError:
                            continue

                        if current_onset - prev_offset > tmp_pause:
                            # 若差值大于tmp_pause，则取得他们的均值
                            target_offset = (current_onset + prev_offset) / 2
                            end = start + target_offset * 1000
                            found = True
                            break
                    if not found:
                        tmp_pause -= 0.01
                # -----------------

                segments.append([start, end])

            start = end
            end = start + segment_duration * 1000
            pbar.n = start
            pbar.refresh()

            if end > total_length:
                if total_length - start > 10:
                    segments[-1][1] = total_length
                    pbar.n = total_length
                    pbar.refresh()
                    break
                else:
                    segments.append([start, total_length])
                    pbar.n = total_length
                    pbar.refresh()
                    break
    # print(segments)
    if not segments:
        segments.append([0.0, total_length])
    
    # print(segments)
    # exit()
    return segments


def get_vad(wav_path, min_pause=0.2, params="folder", if_save=False, verbose=False):
    # print(f"[{show_elapsed_time()}] ({os.path.basename(wav_path)}) VAD processing started...")


    audio_obj = ReadSound(wav_path)

    folder_param_path = os.path.join(os.path.dirname(wav_path), "params_vad.txt")
    file_txt_path = wav_path.replace(".wav", "_vad.txt")

    match params:
        case "file":
            if os.path.exists(file_txt_path):
                with open(file_txt_path, "r") as f:
                    params = eval(f.read())
            else:
                params = default_params
        case "folder":
            if os.path.exists(folder_param_path):
                with open(folder_param_path, "r") as f:
                    params = eval(f.read())
            else:
                params = default_params
        case "default":
            params = default_params
        case _:  # 最好直接输入dict
            if type(params) == dict:
                params = params
            elif type(params) == str:
                params = eval(params)
            else:
                params = default_params
    
    params["offset"] = params["onset"]  # VAD模式特供

    # print(params)


    onsets = autoPraditorWithTimeRange(params, audio_obj, "onset", verbose=False)
    offsets = autoPraditorWithTimeRange(params, audio_obj, "offset", verbose=False)

    onsets = sorted(onsets)
    offsets = sorted(offsets)


    if verbose:   
        print(f"[{show_elapsed_time()}] ({os.path.basename(wav_path)}) VAD onsets: {onsets}")
        print(f"[{show_elapsed_time()}] ({os.path.basename(wav_path)}) VAD offsets: {offsets}")
        
    tg = TextGrid()
    interval_tier = IntervalTier(name="interval", minTime=0., maxTime=audio_obj.duration_seconds)
    tg.append(interval_tier)

    if onsets or offsets:
        if not onsets:
            onsets = [0.0]
        if not offsets:
            offsets = [audio_obj.duration_seconds]
    else:
        return tg
    
    if onsets[0] >= offsets[0]:
        onsets = [0.0] + onsets
    
    if offsets[-1] <= onsets[-1]:
        offsets.append(audio_obj.duration_seconds)

    # Select the one offset that is closest to onset and earlier than onset
    valid_onsets = []
    valid_offsets = []

    if len(onsets) <= len(offsets):
        for i, onset in enumerate(onsets):
            # print(onset)
            if i == 0:
                valid_offsets.append(offsets[-1])  # 最后一个offset
                valid_onsets.append(onsets[0])  # 第一个offset
            else:
                try:
                    valid_offsets.append(max([offset for offset in offsets if onsets[i-1] < offset < onset]))  # 不会影响到最后一个offset，应为需要夹在onset中间
                    valid_onsets.append(onset)  # 下一个onset

                except ValueError:
                    pass
    else:  # len(onsets) > len(offsets)
        reversed_offsets = list(reversed(offsets))
        for i, reversed_offset in enumerate(reversed_offsets):
            if i == 0:
                valid_onsets.append(onsets[0])
                valid_offsets.append(reversed_offset)
            else:
                try:
                    valid_onsets.append(min([onset for onset in onsets if reversed_offset < onset < reversed_offsets[i-1]]))
                    valid_offsets.append(reversed_offset)

                except ValueError:
                    pass
        valid_offsets = list(reversed(valid_offsets))
    


    onsets = sorted(valid_onsets)
    offsets = sorted(valid_offsets)



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
    # segments = segment_audio("data/test_audio.wav", segment_duration=3.5)
    # print(segments)
    auto_vad("data/test_audio.wav", min_pause=0.2, if_save=True, verbose=True)