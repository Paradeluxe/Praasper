from post_process import init_LLM, transliterate_to_en, text_to_ipa, calculate_ipa_similarity

# 初始化LLM模型
tokenizer, model = init_LLM()

# 测试用例
test_cases = [
    "你好世界",  # 中文
    "こんにちは",  # 日语
    "안녕하세요",  # 韩语
    "Bonjour",  # 法语
    "Hola",  # 西班牙语
    "Hello world",  # 英语（应该直接返回）
    "你好 Hello",  # 混合语言
    "こんにちはworld"  # 混合语言
]

print("测试 transliterate_to_en 函数：")
print("-" * 50)

# 存储每个测试用例的音译结果和IPA表示
results = []

# 测试每个用例
for test_text in test_cases:
    try:
        result = transliterate_to_en(test_text)
        ipa = text_to_ipa(result.split())
        results.append((test_text, result, ipa))
        print(f"输入: {test_text}")
        print(f"输出: {result}")
        print(f"音素: {ipa}")
        print("-" * 50)
    except Exception as e:
        print(f"输入: {test_text}")
        print(f"错误: {e}")
        print(f"音素: {text_to_ipa(test_text.split())}")
        print("-" * 50)

# 两两对比测试用例的IPA相似度
print("\n测试两两对比的IPA相似度：")
print("-" * 100)

for i in range(len(results)):
    for j in range(i + 1, len(results)):
        test_text1, result1, ipa1 = results[i]
        test_text2, result2, ipa2 = results[j]
        
        try:
            similarity = calculate_ipa_similarity(ipa1, ipa2)
            print(f"'{test_text1}' vs '{test_text2}'")
            print(f"  音译1: {result1}")
            print(f"  音译2: {result2}")
            print(f"  IPA1: {ipa1}")
            print(f"  IPA2: {ipa2}")
            print(f"  相似度: {similarity:.4f}")
            print("-" * 100)
        except Exception as e:
            print(f"'{test_text1}' vs '{test_text2}'")
            print(f"  错误: {e}")
            print("-" * 100)