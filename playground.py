import praasper
import gc
import os

gc.collect()
os.environ["modelscope_cache"] = r"E:\modelscope_cache"
model = praasper.init_model(
    ASR="FunAudioLLM/Fun-ASR-Nano-2512",
    # infer_mode="direct",
    device="auto",
)
model.annote(r"E:\Corpus\ma\audio")
