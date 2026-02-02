from transformers import AutoModelForCausalLM, AutoTokenizer, T5ForConditionalGeneration
import torch
import numpy as np
import sys
import os
from langdetect import detect
import jellyfish
import re



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


def transliterate_to_en(text):
    """
    将任何语言的文本（单语或混语）按最近音译的方式转换为英文文本
    
    参数:
        text: str - 输入文本（可以是任何语言）
    
    返回:
        str - 音译后的英文文本（仅包含音译结果，无多余解释或备注）
    """
    # 构建对话格式
    messages = [
        {"role": "system", "content": "你是一位精通多语言音译的语言学家。你的任务是将用户提供的任何语言的文本（单语或混语）按最近音译的方式转换为英文文本。音译是指根据发音将一种语言的词语用另一种语言的字母拼写出来，而不是翻译成该语言的对应词汇。核心要求：1. 确保音译结果的发音与原文高度接近。2. 拼写自然，符合英语的拼写规则和语言习惯。3. 对于所有非拉丁字母文本（如中文、日语、韩语等），必须完全音译，绝对不能翻译。4. 如果输入的内容已经是英文文本或使用拉丁字母拼写的语言（如英语、法语、西班牙语、德语等），直接返回原文。5. 对于混合语言文本，必须严格保持原文中的英文部分或拉丁字母部分不变，只对非拉丁字母部分进行音译。6. 只返回音译结果，绝对不要添加任何解释、备注或额外信息。7. 输出必须简洁明了，只包含音译后的英文文本。示例：1. 中文'你好世界'的音译是'Ni Hao Shijie'，而不是'Ni Hao World'。2. 日语'こんにちは'的音译是'Konnichiwa'。3. 混合语言'你好 Hello'的处理结果是'Ni Hao Hello'。4. 英语'Hello world'直接返回'Hello world'。5. 法语'Bonjour'直接返回'Bonjour'。6. 混合语言'こんにちは world'的处理结果是'Konnichiwa world'，而不是'Konnichiwa shijie'。"},
        {"role": "user", "content": f"将以下文本按最近音译的方式转换为英文：{text}"}
    ]

    # 使用模型生成回答
    answer = model_infer(messages)
    return answer


# 全局变量，用于存储G2P模型和分词器
g2p_model = None
g2p_tokenizer = None

def init_g2p_model():
    """
    初始化G2P模型和分词器
    
    返回:
        tuple - (g2p_tokenizer, g2p_model) - G2P分词器和模型实例
    """
    global g2p_model, g2p_tokenizer
    # 检查是否已经初始化，避免重复加载
    if g2p_model is not None and g2p_tokenizer is not None:
        print(f"G2P model already initialized, skipping...")
        return g2p_tokenizer, g2p_model
    
    print(f"Initializing G2P model...")
    # 加载G2P模型和分词器
    g2p_model = T5ForConditionalGeneration.from_pretrained('charsiu/g2p_multilingual_byT5_tiny_16_layers_100')
    g2p_tokenizer = AutoTokenizer.from_pretrained('google/byt5-small')
    print(f"G2P model initialized successfully")
    return g2p_tokenizer, g2p_model

def text_to_ipa(text_list, language=None):
    """
    将文本列表转换为IPA表示
    
    参数:
        text_list: list - 输入文本列表
        language: str - 语言类型，如果为None则自动检测
    
    返回:
        list - IPA表示的列表
    """
    # 初始化G2P模型
    init_g2p_model()
    
    # 处理输入文本
    words = text_list
    # 添加语言前缀
    words = ['<eng-us>: '+i for i in words]
    
    # 编码输入
    out = g2p_tokenizer(words, padding=True, add_special_tokens=False, return_tensors='pt')
    
    # 生成IPA表示
    preds = g2p_model.generate(**out, num_beams=1, max_length=50)
    
    # 解码结果
    phones = g2p_tokenizer.batch_decode(preds.tolist(), skip_special_tokens=True)
    
    return phones




def jaro_winkler_similarity(s1, s2):
    """
    使用jellyfish库计算两个字符串之间的Jaro-Winkler相似度
    
    参数:
        s1: str - 第一个字符串
        s2: str - 第二个字符串
    
    返回:
        float - 两个字符串之间的Jaro距离（0-1之间，值越小表示越相似）
    """
    # 处理空字符串情况
    if not s1 or not s2:
        return 0.0

    
    # 使用jellyfish库的jaro_similarity函数计算相似度
    # 注意：jellyfish.jaro_similarity返回的是相似度（0-1之间，值越大表示越相似）
    similarity = jellyfish.jaro_similarity(s1, s2)
    
    # 返回距离（1 - 相似度）
    return 1.0 - similarity


def calculate_ipa_similarity(ipa1, ipa2):
    """
    使用Jaro算法比较两个IPA列表的相似度
    
    参数:
        ipa1: list - 第一个IPA表示的列表
        ipa2: list - 第二个IPA表示的列表
    
    返回:
        float - 相似度分数（0-1之间，值越大表示越相似）
    """
    # print(ipa1, ipa2)
    
    # 如果任一列表为空，返回相似度0
    if not ipa1 or not ipa2:
        return 0.0
    
    # 计算当前对IPA的Jaro相似度
    # 注意：jaro_distance返回的是距离（0-1），需要转换为相似度
    distance = jaro_winkler_similarity(remove_symbols("".join(ipa1)), remove_symbols("".join(ipa2)))
    similarity = 1.0 - distance

    return similarity

def remove_symbols(text):
    """
    移除字符串中的所有符号，只保留字母、数字和空格
    """
    # 移除非字母、数字、空格和中文字符的所有字符
    return re.sub(r'[^\w\s\u4e00-\u9fff]', '', text)


if __name__ == "__main__":
    # 测试英文文本
    test_text1 = ['Char', 'siu', 'is', 'a', 'Cantonese', 'style', 'of', 'barbecued', 'pork']
    test_text2 = ['Hello', 'world', 'this', 'is', 'a', 'test']

    # 将文本转换为IPA列表
    ipa_text1 = text_to_ipa(test_text1)
    ipa_text2 = text_to_ipa(test_text2)
    print(f"Original text 1: {test_text1}")
    print(f"IPA text 1: {ipa_text1}")
    print(f"Original text 2: {test_text2}")
    print(f"IPA text 2: {ipa_text2}")
    
    # 测试相似度比较
    similarity = calculate_ipa_similarity(ipa_text1, ipa_text2)
    print(f"Similarity: {similarity:.4f}")