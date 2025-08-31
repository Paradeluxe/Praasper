import torch
import whisper
from textgrid import TextGrid, IntervalTier
from pypinyin import pinyin, Style

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
    result = model.transcribe(file_path)
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
            # elif interval.mark == "sound":
            #     # 获取当前 interval 的索引
            #     current_index = tier.intervals.index(interval)
            #     # 获取前一个 interval
            #     prev_interval = tier.intervals[current_index - 1] if current_index > 0 else None
            #     # 获取后一个 interval
            #     next_interval = tier.intervals[current_index + 1] if current_index < len(tier.intervals) - 1 else None

            #     if prev_interval and next_interval:
            #         # 截取音频的起止时间
            #         start_time = prev_interval.minTime
            #         end_time = next_interval.maxTime
            #         # 使用 whisper 模型对截取的音频片段进行转录，并指定语言
            #         try:
            #             sub_result = model.transcribe(file_path, clip_timestamps=(start_time, end_time), language=language)
            #             print(sub_result["text"])
            #         except Exception as e:
            #             print(f"转录音频片段 {start_time}-{end_time} 时出错: {e}")

    
    tg = TextGrid()
    tier = IntervalTier(name='transcription', minTime=0.0, maxTime=vad_tg.tiers[0].maxTime)
    
    for segment in result["segments"]:
        start_time = segment["start"]
        end_time = segment["end"]
        
        text = ' '.join(segment["text"])
        print(text)

        for empty_mark_interval in empty_mark_intervals:
            if empty_mark_interval[0] <= end_time <= empty_mark_interval[1]:
                end_time = empty_mark_interval[0]
            
            if empty_mark_interval[0] <= start_time <= empty_mark_interval[1]:
                start_time = empty_mark_interval[1]
            

            if start_time < empty_mark_interval[0] < empty_mark_interval[1] < end_time:
                pass

        print(start_time, end_time)
        tier.add(start_time, end_time, text)

    # tg.append(vad_tg.tiers[0])
    tg.append(tier)
    tg.write(file_path.replace(".wav", ".TextGrid"))
    # tg.write(r"C:\Users\User\Desktop\MFA\input\test_audio.TextGrid")

# 使用示例
if __name__ == "__main__":

    wav_file_path = r"cantonese_sample_short.wav"
    transcription = transcribe_wav_file(wav_file_path, path_vad="cantonese_sample_short_VAD.TextGrid")
