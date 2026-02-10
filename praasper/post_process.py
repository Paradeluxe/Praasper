from transformers import AutoModelForCausalLM, AutoTokenizer
import jellyfish
import re
try:
    from .utils import show_elapsed_time
except ImportError:
    from praasper.utils import show_elapsed_time


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
    print(f"[{show_elapsed_time()}] Initializing LLM: {model_name}")
    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        # dtype=torch.bfloat16,  # 节省显存
        device_map="auto"            # 自动分配GPU/CPU
    )
    print(f"[{show_elapsed_time()}] LLM initialized successfully")
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


class G2PModel:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self._initialize()
    
    def _initialize(self):
        if self.model is not None and self.tokenizer is not None:
            print(f"G2P model already initialized, skipping...")
            return
        print(f"[{show_elapsed_time()}] Initializing G2P model...")
        # 加载G2P模型和分词器
        from transformers import T5ForConditionalGeneration
        self.model = T5ForConditionalGeneration.from_pretrained('charsiu/g2p_multilingual_byT5_tiny_16_layers_100', tie_word_embeddings=False)
        self.tokenizer = AutoTokenizer.from_pretrained('google/byt5-small', tie_word_embeddings=False)
        print(f"[{show_elapsed_time()}] G2P model initialized successfully")
    
    def text_to_ipa(self, text):
        """
        将文本转换为IPA表示
        
        参数:
            text: str - 输入文本
        
        返回:
            list - IPA表示的列表
        """
        # 编码输入
        out = self.tokenizer('<eng-us>: ' + text, padding=True, add_special_tokens=False, return_tensors='pt')
        
        # 生成IPA表示
        preds = self.model.generate(**out, num_beams=1, max_length=50)
        
        # 解码结果
        phones = self.tokenizer.batch_decode(preds.tolist(), skip_special_tokens=True)
        
        return phones
    
    def jaro_winkler_similarity(self, s1, s2):
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
        return similarity
    
    def calculate_ipa_similarity(self, text1, text2):
        """
        使用Jaro算法比较两个文本列表的相似度
        
        参数:
            text1: list - 第一个文本列表
            text2: list - 第二个文本列表
        
        返回:
            float - 相似度分数（0-1之间，值越大表示越相似）
        """
        # print(text1, text2)

        # 如果任一列表为空，返回相似度0
        if not text1 or not text2:
            return 0.0

        ipa1 = self.text_to_ipa(text1)
        ipa2 = self.text_to_ipa(text2)

        ipa1 = remove_symbols("" .join(ipa1))
        ipa2 = remove_symbols("" .join(ipa2))
        
        # 计算当前对IPA的Jaro相似度
        # 注意：jaro_distance返回的是距离（0-1），需要转换为相似度
        distance = self.jaro_winkler_similarity(ipa1, ipa2)
        similarity = 1.0 - distance

        return similarity





def remove_symbols(text):
    """
    移除字符串中的所有符号，只保留字母、数字和空格
    """
    # 移除非字母、数字、空格和中文字符的所有字符
    return re.sub(r'[^\w\s\u4e00-\u9fff]', '', text)

def calculate_time_diff(list1, list2):
    """
    比较两个包含(onset, offset)对的列表，计算最小累计距离
    
    参数:
        list1: list - 第一个包含(onset, offset)对的列表
        list2: list - 第二个包含(onset, offset)对的列表
    
    返回:
        float - 最小累计距离
    """
    # 确保两个列表都不为空
    if not list1 or not list2:
        return float('inf')
    
    # 确定哪个列表更短，作为参考
    if len(list1) > len(list2):
        list1, list2 = list2, list1
    
    n, m = len(list1), len(list2)
    
    # 初始化动态规划表
    dp = [[float('inf')] * (m + 1) for _ in range(n + 1)]
    dp[0][0] = 0
    
    # 填充动态规划表
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            # 计算当前对的距离
            onset_diff = abs(list1[i-1][0] - list2[j-1][0])
            offset_diff = abs(list1[i-1][1] - list2[j-1][1])
            current_distance = onset_diff + offset_diff
            
            # 选择最小距离
            dp[i][j] = min(dp[i-1][j-1] + current_distance, dp[i][j-1])
    
    # 返回最小累计距离
    return dp[n][m]


if __name__ == "__main__":
    # 测试英文文本
    test_text1 = 'Char siu is a Cantonese style of barbecued pork'
    test_text2 = 'Hello world this is a test'


    
    # 测试相似度比较
    similarity = calculate_ipa_similarity(test_text1, test_text2)
    print(f"Similarity: {similarity:.4f}")
    
    # 测试compare_offset_lists函数
    list1 = [(0, 1), (2, 3), (4, 5)]
    list2 = [(0.1, 1.1), (2.2, 3.2), (4.3, 5.3)]
    distance = calculate_time_diff(list1, list2)
    print(f"Distance between list1 and list2: {distance}")
    
    # 测试不同长度的列表
    list3 = [(0, 1), (2, 3)]
    list4 = [(0., 1.), (2., 3.), (4.3, 5.3)]
    distance2 = calculate_time_diff(list3, list4)
    print(f"Distance between list3 and list4: {distance2}")