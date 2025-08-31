import os
import subprocess
import shutil
import torch
from textgrid import TextGrid
import torch
import whisper
from textgrid import TextGrid, IntervalTier
from pypinyin import pinyin, Style


# Check-ups
# 检查CUDA是否可用
if torch.cuda.is_available():
    print("CUDA可用，当前使用的设备为:", torch.cuda.get_device_name(0))
else:
    print("CUDA不可用，将使用CPU进行计算")



# defs
def transcribe_wav_file(file_path, vad):
    """
    使用 Whisper 模型转录 .wav 文件
    
    :param file_path: .wav 文件的路径
    :param path_vad: VAD TextGrid 文件的路径
    :return: 转录结果
    """
    # 加载最佳模型（large-v3）并指定使用设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = whisper.load_model("large-v3-turbo", device=device)
    # 转录音频文件
    result = model.transcribe(file_path, word_timestamps=True)
    language = result["language"]

    print(result)


    # 加载 path_vad 对应的 TextGrid 文件
    try:
        vad_tg = TextGrid.fromFile(vad)
    except FileNotFoundError:
        print(f"错误：未找到文件 {vad}")
        raise

    # 提取所有 mark 为空字符串的 interval 的起止时间
    vad_intervals = []
    empty_mark_intervals = []
    for tier in vad_tg:
        for interval in tier:
            if interval.mark == "":
                empty_mark_intervals.append((interval.minTime, interval.maxTime))
            else:
                vad_intervals.append((interval.minTime, interval.maxTime))



    tg = TextGrid()
    tier = IntervalTier(name='transcription', minTime=0.0, maxTime=vad_tg.tiers[0].maxTime)
    
    for segment in result["segments"]:
        for idx, word in enumerate(segment["words"]):
            start_time = word["start"]
            end_time = word["end"]
            
            text = word["word"]

            for empty_mark_interval in empty_mark_intervals:
                if empty_mark_interval[0] <= end_time <= empty_mark_interval[1]:
                    end_time = empty_mark_interval[0]
                
                if empty_mark_interval[0] <= start_time <= empty_mark_interval[1]:
                    start_time = empty_mark_interval[1]
                

                if start_time < empty_mark_interval[0] < empty_mark_interval[1] < end_time:
                    pass




            print(start_time, end_time, text)
            tier.add(start_time, end_time, text)

    for vad_interval in vad_intervals:
        # 找到距离 vad_interval[0] 最近的 interval.minTime
        closest_interval = min(tier.intervals, key=lambda x: abs(x.minTime - vad_interval[0]))

        if closest_interval.minTime - vad_interval[0] != 0:
            closest_interval.minTime = vad_interval[0]

        # 找到距离 vad_interval[1] 最近的 interval.maxTime
        closest_interval = min(tier.intervals, key=lambda x: abs(x.maxTime - vad_interval[1]))

        if closest_interval.maxTime - vad_interval[1] != 0:
            closest_interval.maxTime = vad_interval[1]


    tg.append(tier)
    tg.write(file_path.replace(".wav", ".TextGrid"))




if __name__ == "__main__":
    data_path = os.path.abspath("data")
    input_dir = os.path.abspath("input")
    output_dir = os.path.abspath("output")

    fnames = [os.path.splitext(f)[0] for f in os.listdir(data_path) if f.endswith('.wav')]


    for fname in fnames:
        wav_path = os.path.join(data_path, fname + ".wav")

        textgrid_path = wav_path.replace(".wav", ".TextGrid")
        vad_path = wav_path.replace(".wav", "_VAD.TextGrid")
        char_path = wav_path.replace(".wav", "_chars.TextGrid")

        transcription = transcribe_wav_file(wav_path, vad=vad_path)




