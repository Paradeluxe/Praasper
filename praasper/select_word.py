from funasr import AutoModel
from funasr.utils.postprocess_utils import lang_dict, emoji_dict, emo_set, event_set
try:
    from .utils import show_elapsed_time
except ImportError:
    from praasper.utils import show_elapsed_time


class SelectWord:
    def __init__(self, model: str="iic/SenseVoiceSmall", vad_model: str="fsmn-vad", device: str="auto", api_key: str=None):
        print(f"[{show_elapsed_time()}] Initializing ASR {model}")
        self.api_key = api_key
        self.kwargs = {}
        
        if model.startswith("dashscope:"):
            self.infer_mode = "dashscope"
            self.dashscope_model = model.replace("dashscope:", "")
            print(f"[{show_elapsed_time()}] Using DashScope API: {self.dashscope_model}")
        elif device == "api" or model == "dashscope":
            self.infer_mode = "dashscope"
            self.dashscope_model = "fun-asr"
            print(f"[{show_elapsed_time()}] Using DashScope API (default model: fun-asr)")
        elif model == "FunAudioLLM/Fun-ASR-Nano-2512":    
            self.infer_mode = "direct"
            if device == "auto":
                import torch
                device = "cuda:0" if torch.cuda.is_available() else "cpu"
            self.device = device
            from funasr.models.fun_asr_nano.model import FunASRNano
            self.model, self.kwargs = FunASRNano.from_pretrained(
                model=model,
                disable_pbar=True,
                disable_tqdm=True,
                device=self.device,
            )
            self.model.eval()
        else:
            if device == "auto":
                import torch
                device = "cuda:0" if torch.cuda.is_available() else "cpu"
            self.device = device
            self.infer_mode = "funasr"
            self.model = AutoModel(
                model=model,
                vad_model=vad_model,
                vad_kwargs={"max_single_segment_time": 30000},
                device=self.device,
                disable_update=True,
                download_model=False,
            )
    

    def transcribe(self, input_path, lang: str="zh"):
        if self.infer_mode == "dashscope":
            return self._dashscope_transcribe(input_path, lang)
        elif self.infer_mode == "direct":
            res = self.model.inference(data_in=[input_path], **self.kwargs)
        else:
            res = self.model.generate(
                input=input_path,
                language=lang, 
                use_itn=False,
            )
        return res

    def _dashscope_transcribe(self, input_path, lang: str="zh"):
        import os
        from http import HTTPStatus
        import dashscope
        from dashscope.audio.asr import Transcription
        from urllib import request
        import json
        import time
        
        if self.api_key:
            dashscope.api_key = self.api_key
        elif os.environ.get("DASHSCOPE_API_KEY"):
            dashscope.api_key = os.environ.get("DASHSCOPE_API_KEY")
        else:
            raise ValueError("DashScope API key not found. Please set DASHSCOPE_API_KEY environment variable or provide api_key parameter.")
        
        dashscope.base_http_api_url = 'https://dashscope.aliyuncs.com/api/v1'
        
        file_url = input_path if input_path.startswith(('http://', 'https://', 'oss://')) else None
        
        if not file_url:
            raise ValueError("DashScope API requires a public URL. Local file upload is not supported. Please upload your audio file to a public URL first.")
        
        task_response = Transcription.async_call(
            model=self.dashscope_model,
            file_urls=[file_url],
            language_hints=[lang] if lang else None,
        )
        
        if task_response.status_code != HTTPStatus.OK:
            raise Exception(f"Failed to submit task: {task_response.message}")
        
        while True:
            transcription_response = Transcription.wait(task=task_response.output.task_id)
            
            if transcription_response.status_code == HTTPStatus.OK:
                results = transcription_response.output.get('results', [])
                if results and results[0].get('subtask_status') == 'SUCCEEDED':
                    url = results[0].get('transcription_url')
                    result = json.loads(request.urlopen(url).read().decode('utf8'))
                    text = result.get('text', '')
                    return [[{"text": text, "text_tn": text, "timestamps": [], "ctc_timestamps": []}]]
                elif results and results[0].get('subtask_status') == 'FAILED':
                    raise Exception(f"Transcription failed: {results[0]}")
            
            time.sleep(1)



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
    # text = get_text_from_audio(input_path)
    # print(text)