import textgrid
import librosa
import whisper

def process_audio_and_textgrid(audio_path, textgrid_path):
    """
    处理目标音频和textgrid文件，依次读取名为segment的IntervalTier中的每一格interval，
    并根据其minTime和maxTime找到对应范围内名为PV Intervals中的interval们。

    :param audio_path: 目标音频文件的路径
    :param textgrid_path: textgrid文件的路径
    """

    # 加载whisper模型
    model = whisper.load_model("large-v3-turbo", device="cuda")
    
    # 读取textgrid文件
    tg = textgrid.TextGrid.fromFile(textgrid_path)
    
    # 获取名为segment的IntervalTier
    segment_tier = None
    for tier in tg:
        if tier.name == 'segment':
            segment_tier = tier
            break
    
    if segment_tier is None:
        print("未找到名为'segment'的IntervalTier")
        return
    
    # 获取名为PV Intervals的IntervalTier
    pv_intervals_tier = None
    for tier in tg:
        if tier.name == 'PV Intervals':
            pv_intervals_tier = tier
            break
    
    if pv_intervals_tier is None:
        print("未找到名为'PV Intervals'的IntervalTier")
        return
    
    # 遍历segment IntervalTier中的每一格interval
    for segment_interval in segment_tier:
        min_time = segment_interval.minTime
        max_time = segment_interval.maxTime
        
        # 找到对应时间范围内的PV Intervals中的interval们
        matching_pv_intervals = []
        for pv_interval in pv_intervals_tier:
            if pv_interval.minTime >= min_time and pv_interval.maxTime <= max_time:
                matching_pv_intervals.append(pv_interval)
        
        # 此处可添加对匹配到的PV Intervals中的interval的处理逻辑
        # 目前只是打印匹配到的interval
        print()
        for end_idx in range(1, len(matching_pv_intervals) + 1):
            sub_segments = matching_pv_intervals[:end_idx]
            # 后续可根据 sub_segments 的时间范围截取音频并使用 whisper 模型进行中文转录
            # 这里假设音频已加载，以下为伪代码逻辑，实际需根据音频加载和时间截取逻辑实现
            # 计算子片段的最小和最大时间
            min_time = sub_segments[0].minTime
            max_time = sub_segments[-1].maxTime
            # 加载音频并截取对应时间范围
            audio, sr = librosa.load(audio_path, offset=min_time, duration=max_time - min_time)
            # 使用 whisper 模型进行中文转录
            result = model.transcribe(audio, language='zh', word_timestamps=True)
            print(result)

                
                

# 示例调用
if __name__ == "__main__":
    audio_path = r"data/test_audio.wav"
    textgrid_path = r"data/test_audio_valley_points.TextGrid"
    process_audio_and_textgrid(audio_path, textgrid_path)
