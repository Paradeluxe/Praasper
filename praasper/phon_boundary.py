try:
    from .utils import *
    from .cvt import *
except ImportError:
    from utils import *
    from cvt import *


from numba.core.compiler import PassManager
from textgrid import TextGrid, IntervalTier

def find_phon_peak(audio_path, tg_path, lang, verbose=False):
    tg = TextGrid()
    tg.read(tg_path)
    intervals = [interval for interval in tg.tiers[0] if interval.mark != ""]

    phon_tier = IntervalTier(name="phon", maxTime=tg.tiers[0].maxTime)

    for idx, interval in enumerate(intervals):
        con, vow, tone = extract_cvt(interval.mark, lang=lang)[0]

        if con == "":
            phon_tier.addInterval(Interval(interval.minTime, interval.maxTime, vow + tone))
        elif con in ["b", "t", "p", "k"]:
            pass

        elif con in ["z", "zh", "s", "sh", "c", "ch", "j", "x"]:
            pass
            


        
