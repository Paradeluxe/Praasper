import praasper
import gc
import os

gc.collect()
os.environ["modelscope_cache"] = r"E:\modelscope_cache"

# 默认使用 FunASR-Nano 模型
model = praasper.init_model()

# 或者使用 DashScope API（需要公开 URL）
# model = praasper.init_model(
#     device="api",
#     api_key=open("api_key.txt", "r", encoding="utf-8").read().strip()
# )
# model.annote(r"E:\Corpus\ma\audio")

path = r"E:\Corpus\Frog_Story"
model.annote(
    path,
    min_pause=0.5,
    seg_dur=20,
    skip_existing=True
)
