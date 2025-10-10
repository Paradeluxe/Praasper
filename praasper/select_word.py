from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# 构建提示

def which_is_closer(text1, text2, text):

    prompt = f"发音对比：请严格判断“{text2}”和“{text1}”，哪一个在发音上与“{text}”更接近？你的回答必须且只能是“{text2}”或“{text1}”这两个选项之一，不要有任何其他内容。"


    # 加载模型和tokenizer（自动从Hugging Face Hub下载）
    model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,  # 节省显存
        device_map="auto"            # 自动分配GPU/CPU
    )

    # 构建对话格式
    messages = [
        # {"role": "system", "content": "你是一个AI助手"},
        {"role": "user", "content": prompt}
    ]

    # 生成回复
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=1024,
        do_sample=True,
        temperature=0.1,
        top_p=0.9
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    answer = response.split("\n")[-1]
    return answer


if __name__ == "__main__":
    text1 = "不仅仅仅仅仅仅仅仅仅仅仅仅仅仅仅仅仅仅仅仅仅仅仅仅仅仅仅仅仅仅仅仅仅仅仅仅仅仅仅仅仅仅仅仅仅仅仅仅仅仅仅仅仅仅仅仅仅仅仅仅仅仅仅仅仅仅仅仅仅仅仅仅仅仅仅仅仅仅仅仅仅仅仅仅仅仅仅仅仅仅仅仅仅仅仅仅仅仅仅仅仅仅仅仅仅仅仅仅仅仅仅"
    text2 = "瓶颈"
    text = "平静"
    # print(which_is_closer(text1, text2, text))

    answer = ""
    retry_count = 0
    while answer not in [text1, text2]:
        answer = which_is_closer(text1, text2, text)
        retry_count += 1
        if retry_count > 5:
            break
