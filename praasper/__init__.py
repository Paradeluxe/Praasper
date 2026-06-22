import os
import shutil
import numpy as np
import gc
import torch
from itertools import product
import copy
import random
import concurrent.futures
from tqdm import tqdm

try:
    from .VAD.tool_auto import *
    from .select_word import *
    from .utils import *
    from .process import *

except ImportError:
    from praasper.VAD.tool_auto import *
    from praasper.select_word import *
    from praasper.utils import *
    from praasper.process import *

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'  # 设置镜像源
# os.environ["TRANSFORMERS_VERBOSITY"] = "error"
# os.environ["DISABLE_TQDM"] = "1"

# 清空资源
def clear_resources():
    # 清理 CUDA 缓存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    # 强制垃圾回收
    gc.collect()


# ── VAD param validation ────────────────────────────────────────────────────────
VAD_REQUIRED_KEYS = {"amp", "cutoff0", "cutoff1", "numValid", "eps_ratio"}


def validate_vad_params(params):
    """
    Validate that params is a dict with required VAD structure.

    Expected format:
        {
            "onset": {"amp": str, "cutoff0": str, "cutoff1": str, "numValid": str, "eps_ratio": str},
            "offset": {"amp": str, "cutoff0": str, "cutoff1": str, "numValid": str, "eps_ratio": str}
        }

    Raises ValueError if validation fails.
    """
    if not isinstance(params, dict):
        raise ValueError(f"params must be a dict, got {type(params).__name__}")

    for section in ("onset", "offset"):
        if section not in params:
            raise ValueError(f"params missing required section '{section}'")
        if not isinstance(params[section], dict):
            raise ValueError(f"params['{section}'] must be a dict, got {type(params[section]).__name__}")
        missing = VAD_REQUIRED_KEYS - set(params[section].keys())
        if missing:
            raise ValueError(f"params['{section}'] missing required keys: {sorted(missing)}")


def load_params_from_file(path):
    """
    Load VAD params from a .txt file.

    The file must contain a valid Python dict literal that can be eval()'d.
    The loaded dict is validated via validate_vad_params() before being returned.

    Raises ValueError if the file cannot be read or the content is not valid VAD params.
    """
    if not os.path.isfile(path):
        raise ValueError(f"params file not found: {path}")

    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = f.read()
        params = eval(raw)
    except Exception as e:
        raise ValueError(f"Failed to eval params from '{path}': {e}") from e

    validate_vad_params(params)
    return params


# ── Default params (exposed for users to copy and modify) ─────────────────────
_default_params = {'onset': {'amp': '1.2', 'cutoff0': '0', 'cutoff1': '5400', 'numValid': '2000', 'eps_ratio': '0.03'},
                   'offset': {'amp': '1.2', 'cutoff0': '0', 'cutoff1': '5400', 'numValid': '2000', 'eps_ratio': '0.03'}}


class init_model:

    # Expose default params so users can copy and modify them
    default_params = _default_params

    def __init__(
        self,
        ASR: str=None,
        infer_mode: str="local",
        device: str="auto",
        api_key: str=None,
        cache_dir: str=None,
        effort: str="normal",
    ):
        self.infer_mode = infer_mode
        self.api_key = api_key
        self.cache_dir = cache_dir
        self.effort = effort

        # ── 如果指定了 cache_dir，设置模型缓存目录 ──
        if cache_dir:
            os.environ['MODELSCOPE_CACHE'] = cache_dir
            os.environ['HF_HOME'] = cache_dir
            os.environ['TRANSFORMERS_CACHE'] = cache_dir
            os.makedirs(cache_dir, exist_ok=True)

        if infer_mode == "api":
            # ── API mode: DashScope, no local hardware needed ──
            if not api_key and not os.getenv("DASHSCOPE_API_KEY"):
                raise ValueError(
                    "DashScope API key not found. "
                    "Provide api_key= parameter or set DASHSCOPE_API_KEY environment variable."
                )
            if ASR is None:
                ASR = "dashscope:fun-asr"
            self.ASR = ASR
            self.device = "cpu"  # unused, placeholder

        else:
            # ── Local mode ──
            if ASR is None:
                ASR = "FunAudioLLM/Fun-ASR-Nano-2512"
            self.ASR = ASR

            # ── Resolve ffmpeg: prefer system PATH, fall back to bundled static-ffmpeg ──
            # Prevents UnboundLocalError in funasr when both torchaudio backend
            # (torchcodec) and system ffmpeg are missing.
            from ._ffmpeg import resolve_ffmpeg
            _ffmpeg_path = resolve_ffmpeg(
                log_func=lambda msg: print(f"[{show_elapsed_time()}] {msg}")
            )
            if not _ffmpeg_path:
                print(f"[{show_elapsed_time()}] WARNING: no ffmpeg available — "
                      f"FunASR audio loading may fail. "
                      f"Install ffmpeg system-wide, or `pip install static-ffmpeg`.")

            # ── Hardware detection ──────────────────────────────────────
            if device == "cpu":
                self.device = "cpu"
                print(f"[{show_elapsed_time()}] Hardware: cpu")
            else:
                print(f"[{show_elapsed_time()}] Checking hardware ({device})...")
                if torch.cuda.is_available():
                    # ✅ GPU OK
                    self.device = "cuda"
                    print(f"[{show_elapsed_time()}] CUDA detected, using GPU "
                          f"({torch.cuda.get_device_name(0)}, "
                          f"torch CUDA {torch.version.cuda})")

                else:
                    # ❌ torch has no CUDA — diagnose and guide
                    self._diagnose_cuda(device)

        self.model = SelectWord(
            model=self.ASR,
            infer_mode=self.infer_mode,
            device=self.device,
            api_key=self.api_key,
            cache_dir=self.cache_dir
        )

        self.params = _default_params.copy()


    def _diagnose_cuda(self, device):
        """Detect NVIDIA driver and print the exact pip command to fix CUDA torch."""
        import subprocess, re

        # ── Try nvidia-smi to read driver CUDA version ──
        cuda_ver = None
        try:
            r = subprocess.run(
                ["nvidia-smi"], capture_output=True, text=True, timeout=10
            )
            if r.returncode == 0:
                m = re.search(r"CUDA Version:\s*(\d+\.\d+)", r.stdout)
                if m:
                    cuda_ver = m.group(1)
        except Exception:
            pass

        if cuda_ver:
            cu = cuda_ver.replace(".", "")  # "13.0" → "130"
            msg = (
                f"[{show_elapsed_time()}] GPU driver detected (CUDA {cuda_ver}), "
                f"but PyTorch was not installed with CUDA support.\n"
                f"[{show_elapsed_time()}] Run:\n"
                f"[{show_elapsed_time()}]   pip install --force-reinstall torch torchaudio "
                f"--index-url https://download.pytorch.org/whl/cu{cu}\n"
                f"[{show_elapsed_time()}] Then restart your script."
            )
            if device == "cuda":
                raise RuntimeError(msg)
            else:
                self.device = "cpu"
                print(msg)
                print(f"[{show_elapsed_time()}] Falling back to CPU for now.")
        else:
            if device == "cuda":
                raise RuntimeError(
                    f"[{show_elapsed_time()}] device='cuda' but no NVIDIA GPU or driver detected."
                )
            else:
                self.device = "cpu"
                print(f"[{show_elapsed_time()}] No NVIDIA GPU detected, using CPU.")

    def annote(
        self,
        input_path: str,
        seg_dur=15.,
        min_pause=0.2,
        skip_existing: bool=False,
        verbose: bool=False,
        params=None,
        effort=None,
    ):
        """
        Annotate audio file(s) with word-level timestamps.

        Parameters
        ----------
        input_path : str
            Path to a .wav file or a directory containing .wav files.
        seg_dur : float, default 15.0
            Maximum duration (seconds) per audio segment.
        min_pause : float, default 0.2
            Minimum pause (seconds) between speech segments.
        skip_existing : bool, default False
            If True, skip files that already have a output TextGrid.
        verbose : bool, default False
            Print verbose progress messages.
        params : None or dict or str, default None
            VAD parameters to use. Three modes:
              - None           : run auto_vad() grid search to find optimal params (default).
              - dict           : use the given params dict directly (must have onset/offset structure).
              - str (filepath) : load params from a .txt file via eval(), then use them.
        """
        # ── Resolve and validate params ──────────────────────────────────────
        if params is not None:
            if isinstance(params, str):
                # Filepath: load from .txt file
                params = load_params_from_file(params)
            elif isinstance(params, dict):
                validate_vad_params(params)
            else:
                raise TypeError(
                    f"params must be None, a dict, or a str filepath, got {type(params).__name__}"
                )
            self.params = params
        # else: params is None → auto_vad() will run and set self.params below


        if os.path.isdir(input_path):
            file_map = {os.path.splitext(f)[0]: f for f in os.listdir(input_path) if f.lower().endswith('.wav')}
            fnames = list(file_map.keys())
            print(f"[{show_elapsed_time()}] {len(fnames)} valid audio files detected in {input_path}")

        else:
            fnames = [os.path.splitext(os.path.basename(input_path))[0]]
            file_map = {fnames[0]: os.path.basename(input_path)}
            input_path = os.path.dirname(input_path)
            print(f"[{show_elapsed_time()}] {fnames[0]} is detected in {input_path}")


        if not fnames:
            return

        for idx, fname in enumerate(fnames):
            wav_path = os.path.join(input_path, file_map.get(fname, fname + ".wav"))

            dir_name = os.path.dirname(os.path.dirname(wav_path))
            # Thread-safe: unique tmp dir per input_path to avoid collisions
            # when multiple annote() calls run concurrently (e.g., grid search).
            _safe_name = __import__('hashlib').md5(input_path.encode()).hexdigest()[:8]
            tmp_path = os.path.join(dir_name, f"tmp_{_safe_name}")
            # 获取输入文件夹的名称
            input_folder_name = os.path.basename(input_path)
            # 在output目录下创建与输入文件夹同名的子目录
            output_path = os.path.join(dir_name, "output", input_folder_name)
            final_path = os.path.join(output_path, os.path.splitext(os.path.basename(wav_path))[0] + ".TextGrid")


            file_info = f"[{show_elapsed_time()}] {idx+1}/{len(fnames)} ({os.path.basename(wav_path)})"

            # 检查结果文件是否已存在，如果存在且skip_existing为True则跳过处理
            if skip_existing and os.path.exists(final_path):
                print(f"{file_info} Result exists, skipped")
                continue

            try:
                # 尝试加载音频文件
                audio_obj = ReadSound(wav_path)
            except Exception as e:
                print(f"[{show_elapsed_time()}] Error loading audio file {wav_path}: {str(e)}")
                continue

            # 仅在音频加载成功后创建临时目录（很正确）
            if os.path.exists(tmp_path):
                shutil.rmtree(tmp_path)
                if verbose:
                    print(f"[{show_elapsed_time()}] Temporary directory {tmp_path} removed.")
            os.makedirs(tmp_path, exist_ok=False)

            os.makedirs(output_path, exist_ok=True)


            # auto search best params — only run when params was not provided
            if params is None:
                _effort = effort if effort is not None else self.effort
                self.auto_vad(
                    wav_path=wav_path,
                    min_pause=min_pause,
                    file_info=file_info,
                    seg_dur=seg_dur,
                    effort=_effort,
                )

            final_tg = TextGrid()
            final_tg.tiers.append(IntervalTier(name="words", minTime=0., maxTime=audio_obj.duration_seconds))
            final_tg.tiers[0].strict = False

            segments = segment_audio(audio_obj, segment_duration=seg_dur, params=self.params, min_pause=min_pause, file_info=file_info)

            # shared log collector for verbose output from worker threads
            vad_logs = []
            _log_lock = __import__('threading').Lock()

            def process_segments(segment):
            # for start, end in segments:
                start, end = segment
                # count += 1

                # print(f"[{show_elapsed_time()}] Processing segment: {start/1000:.3f} - {end/1000:.3f} ({count})")
                audio_clip = audio_obj[start:end]
                clip_path = os.path.join(tmp_path, f"{os.path.splitext(os.path.basename(wav_path))[0]}_{start}_{end}.wav")
                audio_clip.save(clip_path)
                # WSL D: drive flush: ensure file is visible to reader threads
                with open(clip_path, 'r+b') as _f:
                    _f.flush()
                    os.fsync(_f.fileno())

                audio_clip_result = self.model.transcribe(clip_path)
                audio_clip_single_words = audio_clip_result[0][0]["ctc_timestamps"]
                # print(audio_clip_single_words)

                # Force GC to release soundfile handles held by FunASR before
                # librosa opens the same file in get_vad (avoids Windows
                # file-sharing "System error" / FileNotFoundError).
                gc.collect()


                # try:
                vad_tg = get_vad(clip_path, params=self.params, min_pause=min_pause, verbose=verbose)
                with _log_lock:
                    vad_logs.extend(getattr(vad_tg, '_log', []))
                # except Exception as e:
                #     print(f"[{show_elapsed_time()}] ({os.path.basename(clip_path)}) VAD Error: {e}")
                #     return [None, None, None]

                intervals = vad_tg.tiers[0].intervals
                valid_intervals = [interval for interval in intervals if interval.mark not in ["", None]]# and interval.maxTime - interval.minTime > min_speech]
                # print(valid_intervals)

                # 整体转录模式
                for word in audio_clip_single_words:
                    s_w = word["start_time"]
                    e_w = word["end_time"]
                    t_w = word['token']

                    best_interval = None
                    max_overlap = -1
                    min_distance = float('inf')
                    closest_interval = None

                    for interval in valid_intervals:
                        s_i, e_i = interval.minTime, interval.maxTime

                        # 计算重叠时间
                        overlap_start = max(s_w, s_i)
                        overlap_end = min(e_w, e_i)
                        overlap = max(0, overlap_end - overlap_start)

                        # 如果有重叠且比当前最大重叠大
                        if overlap > max_overlap:
                            max_overlap = overlap
                            best_interval = interval

                        # 计算距离（无重叠时使用）
                        if overlap == 0:
                            # 计算单词与间隔的距离
                            if e_w <= s_i:
                                # 单词在间隔前
                                distance = s_i - e_w
                            else:
                                # 单词在间隔后
                                distance = s_w - e_i

                            if distance < min_distance:
                                min_distance = distance
                                closest_interval = interval

                    # 选择最佳间隔
                    if best_interval is not None:
                        selected_interval = best_interval
                    else:
                        selected_interval = closest_interval

                    # 将单词添加到选定的间隔
                    if selected_interval is not None:
                        if selected_interval.mark == "+":
                            selected_interval.mark = ""

                        selected_interval.mark += f"{t_w}"
                        # selected_interval.words.append((t_w, s_w, e_w))

                timestamps = []
                for idx, valid_interval in enumerate(valid_intervals):
                    s, e = valid_interval.minTime, valid_interval.maxTime

                    # 获取间隔的文本（由前面的单词拼接而成）
                    text = valid_interval.mark

                    text = purify_text(text)
                    if not text or text == "+":
                        continue


                    s_point = s + start/1000
                    e_point = e + start/1000

                    if e_point >= audio_obj.duration_seconds:
                        e_point = audio_obj.duration_seconds

                    timestamps.append([s_point, e_point, text])

                return timestamps

            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:  # 使用线程池并发处理
                result = list(tqdm(executor.map(process_segments, segments), total=len(segments), desc=f"{file_info} Processing segments", leave=False))

            # print collected verbose logs from main thread
            if verbose and vad_logs:
                for line in vad_logs:
                    print(line)

            #############################
            # 因为在建立textgrid的时候使用了strict=False的mode，有可能存在某个tier是重复的
            # 需要检查并调整
            #############################
            results = result[0]
            for r in result[1:]:
                results.extend(r)
            for s_point, e_point, text in results:
                if s_point is None or e_point is None or text is None:
                    continue
                final_tg.tiers[0].add(s_point, e_point, text)

            # 检查并合并重叠的interval
            tier = final_tg.tiers[0]
            i = 1
            while i < len(tier.intervals):
                prev = tier.intervals[i - 1]
                curr = tier.intervals[i]
                # 如果当前interval与上一个interval重叠
                if curr.minTime < prev.maxTime:
                    # 合并：取最早开始和最晚结束
                    new_min = prev.minTime
                    new_max = max(prev.maxTime, curr.maxTime)
                    # 替换上一个interval
                    prev.maxTime = new_max
                    prev.mark += curr.mark
                    # 删除当前interval
                    del tier.intervals[i]
                    # 从头重新检查
                    i = 1
                else:
                    i += 1

            # ----------------------------
            final_tg.write(final_path)

        if os.path.exists(tmp_path):
            shutil.rmtree(tmp_path)
        # print(f"--------------- Processing completed ---------------")


    def export_params(self, path):
        """
        Export the current VAD params to a .txt file.

        The file is written as a valid Python dict literal so it can be
        loaded again via annote(params='/path/to/file.txt').

        Parameters
        ----------
        path : str
            Destination file path (should have .txt extension).
        """
        with open(path, "w", encoding="utf-8") as f:
            f.write(repr(self.params))
        print(f"[{show_elapsed_time()}] Params exported to {path}")


    def auto_vad(self, wav_path, min_pause=0.2, verbose=False, file_info="", seg_dur=10., effort="normal"):
        """
        自动选取最优的VAD参数，根据随机选取的 seg_dur 秒音频。

        effort: "low" (3 combos), "normal" (22 combos), or "high" (100 combos)
        """



        # print(f"[{show_elapsed_time()}] ({os.path.basename(wav_path)}) VAD processing started...")


        dir_name = os.path.dirname(os.path.dirname(wav_path))
        tmp_path = os.path.join(dir_name, "tmp")

        audio_obj = ReadSound(wav_path)
        wav_name = os.path.basename(wav_path)
        if verbose:
            print(f"[{show_elapsed_time()}] ({wav_name}) Full audio duration: {audio_obj.duration_seconds:.3f}")

        sample_dur = seg_dur
        MIN_PART_RATIO = 0.3   # each part is sample_dur * (1 + ratio) to leave wiggle room
        NUM_PARTS_TARGET = 3   # aim for beginning / middle / end coverage

        if audio_obj.duration_seconds > sample_dur:
            max_start = audio_obj.duration_seconds - sample_dur
            min_part_len = sample_dur * (1 + MIN_PART_RATIO)

            # split audio into parts; each part must be long enough to host a sample_dur segment
            max_parts = max(1, int(audio_obj.duration_seconds // min_part_len))
            n_parts = min(NUM_PARTS_TARGET, max_parts)
            part_len = audio_obj.duration_seconds / n_parts

            # pick one random candidate per part, then choose the one with highest energy
            best_start = 0.0
            best_energy = -1.0
            for b in range(n_parts):
                part_start = b * part_len
                part_end = (b + 1) * part_len

                cand_min = part_start
                cand_max = min(part_end, max_start + sample_dur) - sample_dur
                if cand_max < cand_min:
                    continue

                candidate_start = random.uniform(cand_min, cand_max)
                candidate_end = candidate_start + sample_dur
                candidate_audio = audio_obj[candidate_start*1000:candidate_end*1000]
                arr = candidate_audio.arr
                if hasattr(arr, 'numpy'):
                    arr = arr.numpy()
                energy = float(np.mean(arr ** 2))
                if energy > best_energy:
                    best_energy = energy
                    best_start = candidate_start
            start_time = best_start
            end_time = start_time + sample_dur
        else:
            start_time = 0
            end_time = audio_obj.duration_seconds

        # 截取选定的音频段
        selected_audio = audio_obj[start_time*1000:end_time*1000]
        if verbose:
            print(f"[{show_elapsed_time()}] ({wav_name}) Selected audio: {start_time:.3f} - {end_time:.3f}")

        # 确保临时目录存在
        os.makedirs(tmp_path, exist_ok=True)
        # 将selected_audio保存到tmp文件夹
        base_name = os.path.splitext(os.path.basename(wav_path))[0]
        selected_audio_path = os.path.join(tmp_path, f"{base_name}_selected_{start_time:.0f}_{end_time:.0f}.wav")
        selected_audio.save(selected_audio_path)


        standard_result = self.model.transcribe(selected_audio_path)
        standard_transcript = standard_result[0][0]["text_tn"]  # text_tn 没有标点符号
        standard_transcript_timestamps = standard_result[0][0]["timestamps"]
        # print(standard_transcript_timestamps)

        # exit()
        # params = default_params
        # params["offset"] = params["onset"]  # VAD模式特供


        def generate_param_grid(params):
            """生成参数组合列表"""
            keys = params.keys()
            values = params.values()
            for combination in product(*values):
                yield dict(zip(keys, combination))


        # df_res = pd.DataFrame(columns=["params", "transcript", "similarity"])
        # res = {}
        result = []

        # ── Effort-based grid selection ─────────────────────────────────────
        if effort == "low":
            param_grid = {
                "amp":       [1.2],
                "eps_ratio": [0.02, 0.03, 0.04],
            }
            do_stage2 = False
        elif effort == "normal":
            param_grid = {
                "amp":       [1.1, 1.2, 1.3],
                "eps_ratio": [0.02, 0.025, 0.03, 0.035, 0.04, 0.05],
            }
            do_stage2 = True
        elif effort == "high":
            param_grid = {
                "amp":       [1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 1.4, 1.5],
                "eps_ratio": [0.02, 0.025, 0.03, 0.035, 0.04, 0.05],
                "cutoff0":   [0, 200],
            }
            do_stage2 = True
        else:
            raise ValueError(f"Unknown effort level: {effort!r}")

        _filter_cache = {}  # bandpass reuse across combos

        def grid_search_optimal_params(params_replace):
        # for params_replace in generate_param_grid(param_grid):
            # res_key = (params_replace["amp"], params_replace["cutoff0"])
            adjusted_params = _default_params.copy()
            adjusted_params["offset"] = adjusted_params["onset"]
            # print(adjusted_params)
            for p in params_replace:
                adjusted_params["onset"][p] = str(params_replace[p])
            # print(params_replace)
            adjusted_params["offset"] = adjusted_params["onset"]

            # ── Bandpass once, reuse for onset/offset VAD + SNR ──
            c0 = float(adjusted_params["onset"]["cutoff0"])
            c1 = float(adjusted_params["onset"]["cutoff1"])
            key = (c0, c1)
            if key not in _filter_cache:
                _audio_raw = np.array(selected_audio.get_array_of_samples())
                _filter_cache[key] = bandpass_filter(_audio_raw, c0, c1, selected_audio.frame_rate)
            _pre_filtered = _filter_cache[key]

            # print(adjusted_params)
            # print(f"[{show_elapsed_time()}] Testing {adjusted_params['onset']}")
            # exit()

            onsets = autoPraditorWithTimeRange(adjusted_params, selected_audio, "onset", verbose=False, pre_filtered=_pre_filtered)
            offsets = autoPraditorWithTimeRange(adjusted_params, selected_audio, "offset", verbose=False, pre_filtered=_pre_filtered)

            onsets = sorted(onsets)
            offsets = sorted(offsets)


            if verbose:
                # collect verbose info so the main thread can print it (avoids concurrent prints from workers)
                verbose_info = (onsets, offsets)
            else:
                verbose_info = None



            mean_snr = 0.0
            total_overlap = 0.0
            num_intervals = 0

            if onsets or offsets:
                if not onsets:
                    onsets = [0.0]
                if not offsets:
                    offsets = [selected_audio.duration_seconds]


                if onsets[0] >= offsets[0]:
                    onsets = [0.0] + onsets

                if offsets[-1] <= onsets[-1]:
                    offsets.append(selected_audio.duration_seconds)

                # Select the one offset that is closest to onset and earlier than onset
                valid_onsets = []
                valid_offsets = []

                if len(onsets) <= len(offsets):
                    for i, onset in enumerate(onsets):
                        # print(onset)
                        if i == 0:
                            valid_offsets.append(offsets[-1])  # 最后一个offset
                            valid_onsets.append(onsets[0])  # 第一个offset
                        else:
                            try:
                                valid_offsets.append(max([offset for offset in offsets if onsets[i-1] < offset < onset]))  # 不会影响到最后一个offset，应为需要夹在onset中间
                                valid_onsets.append(onset)  # 下一个onset

                            except ValueError:
                                pass
                else:  # len(onsets) > len(offsets)
                    reversed_offsets = list(reversed(offsets))
                    for i, reversed_offset in enumerate(reversed_offsets):
                        if i == 0:
                            valid_onsets.append(onsets[0])
                            valid_offsets.append(reversed_offset)
                        else:
                            try:
                                valid_onsets.append(min([onset for onset in onsets if reversed_offset < onset < reversed_offsets[i-1]]))
                                valid_offsets.append(reversed_offset)

                            except ValueError:
                                pass
                    valid_offsets = list(reversed(valid_offsets))



                onsets = sorted(valid_onsets)
                offsets = sorted(valid_offsets)

                num_intervals = len(onsets)

                # 根据min pause 调整onset和offset
                bad_onsets = []
                bad_offsets = []

                for i in range(len(onsets)-1):
                    if onsets[i+1] - offsets[i] < min_pause:
                        bad_onsets.append(onsets[i+1])
                        bad_offsets.append(offsets[i])

                onsets = [x for x in onsets if x not in bad_onsets]
                offsets = [x for x in offsets if x not in bad_offsets]

                mean_snr = compute_boundary_snr(
                    selected_audio.arr, selected_audio.frame_rate, onsets, offsets,
                    lowcut=float(adjusted_params["onset"]["cutoff0"]),
                    highcut=float(adjusted_params["onset"]["cutoff1"]),
                    pre_filtered=_pre_filtered,
                )

                total_overlap = 0.0
                for onset, offset in zip(onsets[1:-1], offsets[1:-1]):
                    for ts in standard_transcript_timestamps:
                        ts_start, ts_end = ts["start_time"], ts["end_time"]
                        overlap_ratio = max(0.0, min(offset, ts_end) - max(onset, ts_start)) / (ts_end - ts_start)
                        # print(ts_start, ts_end, onset, offset, overlap_ratio)
                        # exit()
                        # print(overlap)

                        total_overlap += overlap_ratio

            # 复制变量
            mean_snr_copy = copy.deepcopy(mean_snr)
            total_overlap_copy = copy.deepcopy(total_overlap)
            num_intervals_copy = copy.deepcopy(num_intervals)
            adjusted_params_copy = copy.deepcopy(adjusted_params)

            # 然后可以像原来一样使用复制后的变量
            return [mean_snr_copy, total_overlap_copy, num_intervals_copy, adjusted_params_copy, verbose_info]

        # ── Stage 1: Coarse grid (amp × eps_ratio) ──────────────────────────
        stage1_combos = list(generate_param_grid(param_grid))
        stage1_results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = {executor.submit(grid_search_optimal_params, combo): combo for combo in stage1_combos}
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures),
                               desc=f"{file_info} Stage 1/2 coarse", leave=False):
                result = future.result()
                if verbose and result[4] is not None:
                    _onsets, _offsets = result[4]
                    print(f"[{show_elapsed_time()}] ({wav_name}) VAD onsets: {_onsets}")
                    print(f"[{show_elapsed_time()}] ({wav_name}) VAD offsets: {_offsets}")
                stage1_results.append(result)

        import statistics
        all_intervals_1 = [r[2] for r in stage1_results if r[2] > 0]
        mean_intval_1 = statistics.mean(all_intervals_1) if all_intervals_1 else 1.0
        best1 = max(stage1_results, key=lambda x: (x[0], x[1], -abs(x[2] - mean_intval_1)))
        best_amp = float(best1[3]['onset']['amp'])
        best_eps = float(best1[3]['onset']['eps_ratio'])

        # ── Stage 2: Fix best amp/eps, vary numValid (DBSCAN min points) ──
        if do_stage2:
            fine_grid = {
                "amp":       [best_amp],
                "eps_ratio": [best_eps],
                "numValid":  [500, 1000, 2000, 5000],
            }
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                stage2_combos = list(generate_param_grid(fine_grid))
                stage2_results = []
                futures = {executor.submit(grid_search_optimal_params, combo): combo for combo in stage2_combos}
                for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures),
                                   desc=f"{file_info} Stage 2/2 numValid", leave=False):
                    r = future.result()
                    if verbose and r[4] is not None:
                        _onsets, _offsets = r[4]
                        print(f"[{show_elapsed_time()}] ({wav_name}) VAD onsets: {_onsets}")
                        print(f"[{show_elapsed_time()}] ({wav_name}) VAD offsets: {_offsets}")
                    stage2_results.append(r)

            result = stage1_results + stage2_results
        else:
            result = stage1_results

        # ── Rank by max mean_snr, then max total_overlap, then closest to mean #intval ─────
        all_intervals = [r[2] for r in result if r[2] > 0]
        mean_intval = statistics.mean(all_intervals) if all_intervals else 1.0

        best = max(result, key=lambda x: (x[0], x[1], -abs(x[2] - mean_intval)))
        max_snr, max_overlap, max_intervals, best_params, _ = best

        if verbose:
            print(f"[{show_elapsed_time()}] ({wav_name}) VAD chosen: SNR={max_snr:.2f} dB, overlap={max_overlap:.3f}, #intval={max_intervals}, params: {best_params['onset']}")

        self.params = best_params

    def release_resources(self):
        """释放模型资源"""
        if hasattr(self, 'model') and self.model is not None:
            # 如果是直接模式，尝试释放模型
            if hasattr(self.model, 'to'):
                self.model.to('cpu')
            # 清空模型引用
            del self.model
        # 清空资源
        clear_resources()
        print("Resources released successfully")

# 导出clear_resources函数
__all__ = ['init_model', 'clear_resources']
