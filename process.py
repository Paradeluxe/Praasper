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
    empty_mark_intervals = []
    for tier in vad_tg:
        for interval in tier:
            if interval.mark == "":
                empty_mark_intervals.append((interval.minTime, interval.maxTime))


    tg = TextGrid()
    tier = IntervalTier(name='transcription', minTime=0.0, maxTime=vad_tg.tiers[0].maxTime)
    
    for segment in result["segments"]:
        start_time = segment["start"]
        end_time = segment["end"]
        
        text = ' '.join(segment["text"])

        for empty_mark_interval in empty_mark_intervals:
            if empty_mark_interval[0] <= end_time <= empty_mark_interval[1]:
                end_time = empty_mark_interval[0]
            
            if empty_mark_interval[0] <= start_time <= empty_mark_interval[1]:
                start_time = empty_mark_interval[1]
            

            if start_time < empty_mark_interval[0] < empty_mark_interval[1] < end_time:
                pass

        print(start_time, end_time, text)
        tier.add(start_time, end_time, text)

    tg.append(tier)
    tg.write(file_path.replace(".wav", ".TextGrid"))





    
    # 为每个单字结果创建 TextGrid
    char_tg = TextGrid()
    char_tier = IntervalTier(name='characters', minTime=0.0, maxTime=vad_tg.tiers[0].maxTime)
    
    for segment in result["segments"]:
        for word in segment["words"]:
            text = word["word"].strip()
            start_time = word["start"]
            end_time = word["end"]
            
            # 处理每个单字
            for char in text:
                # 假设单字均匀分布在单词时间区间内，这里简单处理，实际可能需要更复杂逻辑
                char_duration = (end_time - start_time) / len(text)
                char_start = start_time
                char_end = start_time + char_duration
                
                # 处理静音区间的影响
                for empty_mark_interval in empty_mark_intervals:
                    if empty_mark_interval[0] <= char_end <= empty_mark_interval[1]:
                        char_end = empty_mark_interval[0]
                    if empty_mark_interval[0] <= char_start <= empty_mark_interval[1]:
                        char_start = empty_mark_interval[1]
                    if char_start < empty_mark_interval[0] < empty_mark_interval[1] < char_end:
                        pass
                
                if char_start < char_end:  # 确保起始时间小于结束时间
                    char_tier.add(char_start, char_end, char)
                
                start_time = char_end
    
    char_tg.append(char_tier)
    char_tg.write(file_path.replace(".wav", "_chars.TextGrid"))


if __name__ == "__main__":
    data_path = os.path.abspath("data")
    input_dir = os.path.abspath("input")
    output_dir = os.path.abspath("output")

    fnames = [os.path.splitext(f)[0] for f in os.listdir(data_path) if f.endswith('.wav')]


    for fname in fnames:
        wav_path = os.path.join(data_path, fname + ".wav")

        textgrid_path = wav_path.replace(".wav", ".TextGrid")
        vad_path = wav_path.replace(".wav", "_VAD.TextGrid")

        transcription = transcribe_wav_file(wav_path, vad=vad_path)



        # 若.TextGrid文件存在，则复制到input目录
        if os.path.exists(textgrid_path):
            shutil.copy2(wav_path, os.path.join(input_dir, os.path.basename(wav_path))) # 复制.wav文件到input目录
            shutil.copy2(textgrid_path, os.path.join(input_dir, os.path.basename(textgrid_path)))




        cmd_prefix = "" # "mamba run -n aligner "

        cmd = cmd_prefix + f"mfa align {input_dir} mandarin_mfa mandarin_mfa {output_dir} --clean --fine_tune"

        # 使用mamba运行MFA命令
        subprocess.run(cmd)






        mfa_path = os.path.join(output_dir, fname + ".TextGrid")              
        whisper_path = os.path.join(input_dir, fname + ".TextGrid")    



        # 读取MFA的TextGrid文件
        mfa_tg = TextGrid.fromFile(mfa_path)
        mfa_tier = [t for t in mfa_tg.tiers if t.name == "words"][0]

        # 读取VAD的TextGrid文件
        vad_tg = TextGrid.fromFile(vad_path)
        vad_tier = [t for t in vad_tg.tiers if t.name == "interval"][0]

        # 读取Whisper的TextGrid文件
        whisper_tg = TextGrid.fromFile(whisper_path)
        whisper_tier = [t for t in whisper_tg.tiers if t.name == "transcription"][0]


        # 遍历mfa_tier里的每一个非空mark的interval
        for mfa_interval in mfa_tier.intervals:

            if mfa_interval.mark == "":
                continue

            # 记录当前 mfa_interval 的起始和结束时间
            mfa_start = mfa_interval.minTime
            mfa_end = mfa_interval.maxTime
            max_overlap = 0
            best_vad_interval = None

            # 遍历 vad_tier 里的每个 interval
            for vad_interval in vad_tier.intervals:
                if vad_interval.mark != "":
                    vad_start = vad_interval.minTime
                    vad_end = vad_interval.maxTime
                    # 计算重叠时间
                    overlap_start = max(mfa_start, vad_start)
                    overlap_end = min(mfa_end, vad_end)
                    overlap_duration = max(0, overlap_end - overlap_start)

                    # 更新最大重叠时间和对应的 interval
                    if overlap_duration > max_overlap:
                        max_overlap = overlap_duration
                        best_vad_interval = vad_interval

            # 如果找到有重叠的 interval，将 mfa_interval 的 mark 写入
            if best_vad_interval:
                if best_vad_interval.mark == "sound":
                    best_vad_interval.mark = mfa_interval.mark
                else:
                    best_vad_interval.mark += mfa_interval.mark
            
            
        # 新建一个 TextGrid 对象
        final_tg = TextGrid()

        # 将三个 tier 添加到新的 TextGrid 对象中
        final_tg.append(vad_tier)

        # 导出到 final.TextGrid 文件
        final_tg.write("final.TextGrid")


