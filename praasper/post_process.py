from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import numpy as np


# 全局变量，用于存储初始化的tokenizer和model
tokenizer = None  # 分词器
model = None      # 语言模型


def init_LLM(LLM: str="Qwen/Qwen2.5-1.5B-Instruct"):
    """
    初始化LLM模型和分词器
    
    参数:
        LLM: str - 模型名称或路径，默认为"Qwen/Qwen2.5-1.5B-Instruct"
    
    返回:
        tuple - (tokenizer, model) - 分词器和模型实例
    """
    global tokenizer, model
    # 检查是否已经初始化，避免重复加载
    if tokenizer is not None and model is not None:
        print(f"LLM already initialized, skipping...")
        return tokenizer, model
    
    model_name = LLM
    print(f"Initializing LLM: {model_name}")
    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        # dtype=torch.bfloat16,  # 节省显存
        device_map="auto"            # 自动分配GPU/CPU
    )
    print(f"LLM initialized successfully")
    return tokenizer, model


def model_infer(messages):
    """
    使用LLM模型生成回复
    
    参数:
        messages: list - 对话消息列表，每个元素为字典，包含role和content
    
    返回:
        str - 模型生成的回复文本
    """
    # 生成回复
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    # 编码输入文本
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    # 生成输出
    outputs = model.generate(
        **inputs,
        max_new_tokens=512,     # 最大新生成的token数
        do_sample=True,         # 使用采样策略
        temperature=0.1,        # 温度参数，控制输出的随机性
        top_p=0.9               # top-p采样参数
    )
    # 解码输出文本
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # 提取最后一行作为回答
    answer = response.split("\n")[-1]
    return answer


def post_process(text, language):
    """
    对文本进行后处理，主要用于音译
    
    参数:
        text: str - 输入文本
        language: str - 目标语言
    
    返回:
        str - 处理后的文本
    """
    # 系统提示词
    sys_prompt = f"""扮演一位精通 {language} 的语言学家。你的任务是将用户提供的文本音译为 {language} 。

核心要求：
- 如果输入的内容已经是 {language} 文本，直接返回。
- 发音优先： 确保音译结果的发音与原文高度接近。
- 拼写自然： 音译用词需符合目标语言的拼写规则和语言习惯。
- 输出格式： 请直接给出最佳音译结果，不要给出任何解释。
"""

    # 构建对话格式
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": text}
    ]

    # 使用模型生成回答
    answer = model_infer(messages)
    return answer


def get_char_language(char):
    """
    检测单个字符的语言类型
    
    参数:
        char: str - 单个字符
    
    返回:
        str - 语言类型
    """
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


def detect_main_language(text):
    """
    检测文本的主要语言
    
    参数:
        text: str - 输入文本
    
    返回:
        str - 主要语言类型
    """
    # 统计每种语言的出现次数
    lang_counts = {}
    for char in text:
        lang = get_char_language(char)
        if lang != 'other':
            lang_counts[lang] = lang_counts.get(lang, 0) + 1
    
    # 如果没有检测到语言字符，默认返回英语
    if not lang_counts:
        return 'english'
    
    # 返回出现次数最多的语言
    return max(lang_counts, key=lang_counts.get)


def text_to_ipa(text, language=None):
    """
    将文本转换为字符级别的表示（作为IPA的替代方案）
    
    参数:
        text: str - 输入文本
        language: str - 语言类型，如果为None则自动检测
    
    返回:
        list - 字符列表
    """
    # 直接返回字符列表作为音素的替代方案
    # 这样可以避免对外部依赖的要求
    return list(text)


def is_single_language(text):
    """
    检测文本中所有字符是否属于同一种语言
    
    参数:
        text: str - 输入文本
    
    返回:
        bool - 是否单一语言
    """
    # 收集所有字符的语言类型（忽略'other'类型）
    languages = set()
    for char in text:
        lang = get_char_language(char)
        if lang != 'other':  # 只关心实际的语言字符
            languages.add(lang)
    
    # 判断是否单一语言
    return len(languages) <= 1


def which_is_closer(text1, text2, text):
    """
    判断两个文本中哪个在发音上与目标文本更接近
    
    参数:
        text1: str - 第一个对比文本
        text2: str - 第二个对比文本
        text: str - 目标文本
    
    返回:
        str - 更接近的文本（text1或text2）
    """
    prompt = f"发音对比：请严格判断'{text2}'和'{text1}'，哪一个在发音上与'{text}'更接近？你的回答必须且只能是'{text2}'或'{text1}'这两个选项之一，不要有任何其他内容。"

    messages = [
        # {"role": "system", "content": "你是一个AI助手"},
        {"role": "user", "content": prompt}
    ]
    # 使用模型生成回答
    answer = model_infer(messages)
    return answer


def dtw_distance(seq1, seq2):
    """
    纯Python实现的DTW算法，计算两个序列之间的距离
    
    参数:
        seq1: list - 第一个序列
        seq2: list - 第二个序列
    
    返回:
        float - 两个序列之间的DTW距离
    """
    # 获取序列长度
    n, m = len(seq1), len(seq2)
    
    # 创建距离矩阵
    dtw_matrix = np.zeros((n+1, m+1))
    
    # 初始化第一行和第一列
    for i in range(1, n+1):
        dtw_matrix[i, 0] = float('inf')
    for j in range(1, m+1):
        dtw_matrix[0, j] = float('inf')
    dtw_matrix[0, 0] = 0
    
    # 填充距离矩阵
    for i in range(1, n+1):
        for j in range(1, m+1):
            # 计算当前元素的距离（使用字符的ASCII码差异）
            cost = abs(ord(seq1[i-1]) - ord(seq2[j-1]))
            # 取最小值
            dtw_matrix[i, j] = cost + min(dtw_matrix[i-1, j],    # 上
                                         dtw_matrix[i, j-1],    # 左
                                         dtw_matrix[i-1, j-1])  # 对角线
    
    return dtw_matrix[n, m]


def calculate_ipa_similarity(text1, text2):
    """
    使用DTW算法比较两个文本的相似度
    
    参数:
        text1: str - 第一个文本
        text2: str - 第二个文本
    
    返回:
        float - 相似度分数（0-1之间，值越大表示相似度越高）
    """
    # 将两个文本转换为字符列表
    ipa1 = text_to_ipa(text1)
    ipa2 = text_to_ipa(text2)
    
    # 如果任一转换失败，返回相似度0
    if not ipa1 or not ipa2:
        return 0.0
    
    # 使用自定义的DTW算法计算距离
    try:
        distance = dtw_distance(ipa1, ipa2)
        
        # 将距离转换为相似度分数（0-1之间）
        # 相似度 = 1 / (1 + 归一化距离)
        max_possible_distance = (len(ipa1) + len(ipa2)) * 255  # 最大ASCII差异
        normalized_distance = distance / max_possible_distance
        similarity = 1.0 / (1.0 + normalized_distance)
        
        return similarity
    except Exception as e:
        print(f"Error calculating DTW similarity: {e}")
        return 0.0



if __name__ == "__main__":
    # 测试粤语文本
    cantonese_text1 = "早晨"  # 粤语：早上好
    cantonese_text2 = "早唞"  # 粤语：晚安
    mandarin_text = "早上好"    # 普通话：早上好
    
    # 测试语言检测
    print(f"语言检测 - '早晨': {detect_main_language(cantonese_text1)}")
    print(f"语言检测 - '早唞': {detect_main_language(cantonese_text2)}")
    
    # 测试相似度比较
    similarity1 = calculate_ipa_similarity(cantonese_text1, cantonese_text2)
    similarity2 = calculate_ipa_similarity(cantonese_text1, mandarin_text)
    print(f"'早晨' vs '早唞' -> 相似度: {similarity1:.4f}")
    print(f"'早晨' vs '早上好' -> 相似度: {similarity2:.4f}")