from funasr import AutoModel
from funasr.utils.postprocess_utils import lang_dict, emoji_dict, emo_set, event_set


# model_dir = "paraformer-zh"
class SelectWord:
    def __init__(self, model: str="iic/SenseVoiceSmall", vad_model: str="fsmn-vad", device: str="auto"):
        # 自动检测设备
        if device == "auto":
            import torch
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        
        self.model = AutoModel(
            model=model,
            vad_model=vad_model,
            # punc_model="ct-punc",
            vad_kwargs={"max_single_segment_time": 30000},
            device=device,
            disable_update=True,
            disable_pbar=True,
            disable_log=True,
            # use_timestamp=True
            download_model=False,
            # trust_remote_code=True,
        )

    def transcribe(self, input_path):

        res = self.model.generate(
            input=input_path,
            language="zh", 
            use_itn=False,
            #hotword="必须使用中文输出",
            # batch_size_s=60,
            # merge_length_s=15,

        )
        text = rich_transcription_postprocess_text_only(res[0]["text"])

        return text



def rich_transcription_postprocess_text_only(s):
    """
    修改版的rich_transcription_postprocess函数，只保留文字内容，去除所有表情符号和事件标记
    """
    # 替换所有语言标记为空
    for lang in lang_dict:
        s = s.replace(lang, "")
    
    # 移除所有特殊标记
    for special_tag in emoji_dict:
        s = s.replace(special_tag, "")
    
    # 移除所有表情符号和事件符号
    for emo in emo_set:
        s = s.replace(emo, "")
    
    for event in event_set:
        s = s.replace(event, "")
    
    # 清理多余的空格
    s = s.replace("The.", " ")
    s = " ".join(s.split())  # 移除多余的空格
    
    return s.strip()



if __name__ == "__main__":
    # input_path = r"tmp/15-1_5.wav" # f"{model.model_path}/example/en.mp3"
    input_path = r"input_single/15-1.wav"
    text = get_text_from_audio(input_path)
    print(text)