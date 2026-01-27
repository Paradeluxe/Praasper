import praasper


model = praasper.init_model(
    ASR="FunAudioLLM/Fun-ASR-Nano-2512",
    infer_mode="direct",
    device="cpu",
    LLM= "Qwen/Qwen3-4B-Instruct-2507"# "Qwen/Qwen3-8B"
)

model.annote(
    input_path=r"E:\Corpus\pic_nam",
    min_pause=0.5,
    min_speech=.2,
    language="en",
    seg_dur=15.,
    skip_existing=True,
    enable_post_process=False
)