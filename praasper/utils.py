import time
try:
    from .cvt import *
except ImportError:
    from cvt import *

# 记录程序开始执行的时间
START_TIME = time.time()

def show_elapsed_time():
    """显示从程序开始执行到现在的时间差"""
    elapsed = time.time() - START_TIME
    minutes = int(elapsed // 60)
    seconds = elapsed % 60
    milliseconds = int((elapsed % 1) * 1000)
    return f"{minutes:02d}:{int(seconds):02d}:{milliseconds:03d}"





def extract_cvt(character, lang):
    if lang == 'zh':
        return extract_cvt_zh(character)
    elif lang == "yue":
        return extract_cvt_yue(character)
    else:
        raise ValueError(f"{show_elapsed_time()}Language {lang} not yet supported.")


if __name__ == '__main__':
    print(extract_cvt('边'))