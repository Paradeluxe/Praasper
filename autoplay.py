import praasper
import gc
import torch

# 清空资源
def clear_resources():
    # 清理 CUDA 缓存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    # 强制垃圾回收
    gc.collect()

# 开头清空一次
clear_resources()
print("Initial resources cleared")

model = None
model = praasper.init_model(
    ASR="FunAudioLLM/Fun-ASR-Nano-2512",
    infer_mode="direct",
    device="auto",
    LLM= "Qwen/Qwen3-4B-Instruct-2507"# "Qwen/Qwen3-8B"
)

model.auto_vad(
    wav_path=rf"C:\Users\User\Desktop\Praasper\big_data\ep1_v4.wav",
)
# 释放资源
if model is not None:
    # 释放模型资源
    if hasattr(model, 'model') and model.model is not None:
        # 如果是直接模式，尝试释放模型
        if hasattr(model.model, 'to'):
            model.model.to('cpu')
        # 清空模型引用
        del model.model
    # 清空模型引用
    del model
    
    # 清空资源
    clear_resources()
    print("Resources released successfully")