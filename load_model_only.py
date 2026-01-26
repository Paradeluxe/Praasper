import praasper


model = praasper.init_model(
    ASR="iic/SenseVoiceSmall",
    LLM= "Qwen/Qwen3-8B"
)

model.annote(
    input_path=r"E:\Corpus\Free_Speech_Macau",
    min_pause=0.8,
    min_speech=.2,
    language="yue",
    seg_dur=15.,
    skip_existing=False,
    enable_post_process=True
)