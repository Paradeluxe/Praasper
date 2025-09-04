import pypinyin

def get_pinyin_info(character):
    """
    给定一个中文单字，返回其对应的拼音（声母、韵母、声调）
    
    :param character: 单个中文字符
    :return: 包含声母、韵母、声调的字典
    """
    if len(character) != 1:
        raise ValueError("只能输入单个中文字符")
    
    # 获取拼音信息
    pinyin_result = pypinyin.pinyin(character, style=pypinyin.TONE3, heteronym=False)[0][0]
    
    # 提取声母、韵母和声调
    initial = pypinyin.pinyin(character, style=pypinyin.INITIALS, heteronym=False)[0][0]
    final_with_tone = pypinyin.pinyin(character, style=pypinyin.FINALS_TONE3, heteronym=False)[0][0]
    
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


    
    return (
        initial if initial else '',
        final,
        tone if tone else ''
    )


if __name__ == '__main__':
    print(get_pinyin_info('边'))