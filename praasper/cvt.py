import pypinyin
from pycantonese import characters_to_jyutping


def extract_cvt_yue(character):
    """
    给定一个中文单字，返回其对应的粤语拼音（声母、韵母、声调）
    
    :param character: 单个中文字符
    :return: 包含声母、韵母、声调的字典
    """
    if len(character) != 1:
        return ("", [character], "")
    
    # 导入放在函数内避免依赖问题（实际使用时建议放在顶部）
    
    # 获取粤语拼音（Jyutping）
    jyutping_list = characters_to_jyutping(character)
    if not jyutping_list:
        return ("", [], "")
    
    # 取第一个读音（忽略多音字）
    full_pinyin = jyutping_list[0][0]
    
    # 分离声调（Jyutping 声调在末尾数字1-6）
    tone = ""
    for char in reversed(full_pinyin):
        if char.isdigit():
            tone = char
            base = full_pinyin.rstrip('123456')
            break
    else:
        base = full_pinyin
    
    # 分离声母和韵母
    initials = ["b", "p", "m", "f", "d", "t", "n", "l", "g", "k", "ng", "h", 
                "gw", "kw", "w", "z", "c", "s", "j"]
    initial = ""
    final = base
    
    # 检查最长匹配（先匹配2字母声母）
    for s in sorted(initials, key=len, reverse=True):
        if base.startswith(s):
            initial = s
            final = base[len(s):]
            break
    
    # 处理韵母（将字符串转为列表并合并 ng）
    final_list = []
    i = 0
    while i < len(final):
        if i < len(final) - 1 and final[i:i+2] == "ng":
            final_list.append("ng")
            i += 2
        else:
            final_list.append(final[i])
            i += 1
    
    return (
        initial,
        final_list,
        tone
    )


def extract_cvt_zh(character):
    """
    给定一个中文单字，返回其对应的拼音（声母、韵母、声调）
    
    :param character: 单个中文字符
    :return: 包含声母、韵母、声调的字典
    """
    # if len(character) != 1:
        # raise ValueError("只能输入单个中文字符")
    cvts = []
    for char in character:
        # 获取拼音信息
        pinyin_result = pypinyin.pinyin(char, style=pypinyin.TONE3, heteronym=False)[0][0]
        
        # 提取声母、韵母和声调
        initial = pypinyin.pinyin(char, style=pypinyin.INITIALS, heteronym=False)[0][0]
        final_with_tone = pypinyin.pinyin(char, style=pypinyin.FINALS_TONE3, heteronym=False)[0][0]
        
        # 分离韵母和声调
        tone = ''
        final = final_with_tone
        for char in final_with_tone:
            if char.isdigit():
                tone = char
                final = final_with_tone.replace(char, '')
                break
        
        

        final = list(final)
        # 如果n和g相邻，合并成ng
        new_final = []
        i = 0
        while i < len(final):
            if i < len(final) - 1 and final[i] == 'n' and final[i+1] == 'g':
                new_final.append('ng')
                i += 2
            else:
                new_final.append(final[i])
                i += 1
        final = new_final

        cvts.append((
            initial if initial else '',
            final,
            tone if tone else ''
        ))
    
    return cvts
