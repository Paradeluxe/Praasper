from funasr import AutoModel
from funasr.utils.postprocess_utils import lang_dict, emoji_dict, emo_set, event_set
try:
    from .utils import show_elapsed_time
except ImportError:
    from praasper.utils import show_elapsed_time


class SelectWord:
    def __init__(self, model: str="iic/SenseVoiceSmall", vad_model: str="fsmn-vad", device: str="auto", infer_mode: str="direct"):
        print(f"[{show_elapsed_time()}] Initializing ASR {model}")
        # 自动检测设备
        if device == "auto":
            import torch
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.infer_mode = infer_mode
        self.kwargs = {}
        
        if infer_mode == "direct":
            # 使用 FunASRNano 直接模式
            from funasr.models.fun_asr_nano.model import FunASRNano
            # from model import FunASRNano
            self.model, self.kwargs = FunASRNano.from_pretrained(
                model=model,
                disable_pbar=True,
                # checkpoint_callback=False,
                # trust_remote_code=False,
                disable_tqdm=True,
                device=self.device,
            )
            self.model.eval()


        else:
            # 默认使用 AutoModel
            self.model = AutoModel(
                model=model,
                vad_model=vad_model,
                # punc_model="ct-punc",
                vad_kwargs={"max_single_segment_time": 30000},
                device=self.device,
                # disable_update=True,
                # disable_pbar=True,
                # disable_log=True,
                # use_timestamp=True
                download_model=False,
                # trust_remote_code=True,
            )
        

    def transcribe(self, input_path, lang: str="zh"):

        if self.infer_mode == "direct":
            # 使用 FunASRNano 的 inference 方法
            res = self.model.inference(data_in=[input_path], **self.kwargs)
            # print(res)
            # text = res[0][0]["text"]
        else:
            # 使用 AutoModel 的 generate 方法
            res = self.model.generate(
                input=input_path,
                language=lang, 
                use_itn=False,
                #hotword="必须使用中文输出",
                # batch_size_s=60,
                # merge_length_s=15,

            )
            # text = res[0]["text"]
        # print(res)
        # exit()
        # text = rich_transcription_postprocess_text_only(text)
        # return text
        return res



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