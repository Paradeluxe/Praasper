# 此代码用于检查 Whisper 是否可以在 GPU 上运行
try:
    import torch

    # 检查 CUDA 是否可用
    if torch.cuda.is_available():
        print("CUDA 可用，GPU 可以正常工作")
    else:
        print("CUDA 不可用，GPU 无法正常工作")
except ImportError:
    print("未安装 torch 库，请运行 'pip install torch' 安装所需库")
except Exception as e:
    print(f"发生未知错误: {e}")
