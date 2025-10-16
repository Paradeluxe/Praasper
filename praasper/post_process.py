from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


def init_LLM(LLM: str="Qwen/Qwen2.5-1.5B-Instruct"):
    global tokenizer, model
    model_name = LLM
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        # dtype=torch.bfloat16,  # 节省显存
        device_map="auto"            # 自动分配GPU/CPU
    )
    return tokenizer, model


def model_infer(messages):
    # 生成回复
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        do_sample=True,
        temperature=0.1,
        top_p=0.9
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    answer = response.split("\n")[-1]
    return answer


def post_process(text, language):
    
    sys_prompt = f"""扮演一位精通 {language} 的语言学家。你的任务是将用户提供的文本音译为 {language} 。

​核心要求：​​
- 如果输入的内容已经是 {language} 文本，直接返回。
- ​发音优先：​​ 确保音译结果的发音与原文高度接近。
- ​拼写自然：​​ 音译用词需符合目标语言的拼写规则和语言习惯。
- ​输出格式：​​ 请直接给出最佳音译结果，不要给出任何解释。
"""


    # 构建对话格式
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": text}
    ]

    answer = model_infer(messages)
    return answer


def is_single_language(text):
    """
    检测文本中所有字符是否属于同一种语言
    返回: (bool, set) - (是否单一语言, 检测到的语言集合)
    """
    # 语言检测函数
    def get_char_language(char):
        if '\u4e00' <= char <= '\u9fff':  # 中文字符范围
            return 'chinese'
        elif '\u3040' <= char <= '\u309f':  # 平假名
            return 'japanese'
        elif '\u30a0' <= char <= '\u30ff':  # 片假名
            return 'japanese'
        elif '\uac00' <= char <= '\ud7a3':  # 韩文字符
            return 'korean'
        elif 'a' <= char <= 'z' or 'A' <= char <= 'Z':  # 英文字母
            return 'english'
        elif '\u0400' <= char <= '\u04ff':  # 西里尔字母（俄语等）
            return 'cyrillic'
        elif '\u0600' <= char <= '\u06ff':  # 阿拉伯字母
            return 'arabic'
        else:
            return 'other'  # 标点、数字、空格等非文字字符
    
    # 收集所有字符的语言类型（忽略'other'类型）
    languages = set()
    for char in text:
        lang = get_char_language(char)
        if lang != 'other':  # 只关心实际的语言字符
            languages.add(lang)
    
    # 判断是否单一语言
    # return len(languages) <= 1, languages
    return len(languages) <= 1


def which_is_closer(text1, text2, text):
    prompt = f"发音对比：请严格判断“{text2}”和“{text1}”，哪一个在发音上与“{text}”更接近？你的回答必须且只能是“{text2}”或“{text1}”这两个选项之一，不要有任何其他内容。"

    messages = [
        # {"role": "system", "content": "你是一个AI助手"},
        {"role": "user", "content": prompt}
    ]
    answer = model_infer(messages)
    return answer



if __name__ == "__main__":
    # text1 = "不仅仅仅仅仅仅仅仅仅仅仅仅仅仅仅仅仅仅仅仅仅仅仅仅仅仅仅仅仅仅仅仅仅仅仅仅仅仅仅仅仅仅仅仅仅仅仅仅仅仅仅仅仅仅仅仅仅仅仅仅仅仅仅仅仅仅仅仅仅仅仅仅仅仅仅仅仅仅仅仅仅仅仅仅仅仅仅仅仅仅仅仅仅仅仅仅仅仅仅仅仅仅仅仅仅仅仅仅仅仅仅"
    # text2 = "瓶颈"
    # text = "平静"

    # print(which_is_closer(text1, text2, text))


    text = "温合适"


    answer = post_process(text, "zh")
    print(answer)
    # retry_count = 0
    # while answer not in [text1, text2]:
    #     answer = which_is_closer(text1, text2, text)
    #     retry_count += 1
    #     if retry_count > 5:
    #         break