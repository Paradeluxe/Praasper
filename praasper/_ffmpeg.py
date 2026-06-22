"""Resolve ffmpeg: prefer system PATH; fall back to static-ffmpeg bundled binary.

Returns path to a callable ffmpeg, or None. Logs status via log_func callback
so the caller controls formatting (Praasper uses [HH:MM:mmm] show_elapsed_time).

Behavior matrix:
  System ffmpeg found       -> return system path, 0s
  Bundled already cached    -> return bundled path, <100ms
  Bundled not yet extracted -> static-ffmpeg downloads + extracts (~25s first time).
                               It prints its own progress bar to stderr.
  Both fail                 -> return None, log error.
"""
import os
import shutil


def resolve_ffmpeg(log_func=None):
    """Returns path to ffmpeg (str) or None. log_func(str) for status messages.

    Args:
        log_func: callable (msg: str) -> None for status messages. If None, silent.
    """
    log = log_func if log_func else lambda *a, **k: None

    # 1) System PATH first — free, no download
    p = shutil.which("ffmpeg")
    if p:
        log(f"ffmpeg: {p}  (system PATH)")
        return p

    # 2) No system ffmpeg — fall back to bundled
    log("ffmpeg: not found on PATH, falling back to bundled binary...")

    try:
        import static_ffmpeg
    except ImportError:
        log("ffmpeg: ERROR — static-ffmpeg not installed. "
            "Run `pip install static-ffmpeg` or install ffmpeg system-wide.")
        return None

    # Detect cache state BEFORE add_paths() so we can give accurate log
    bin_dir = os.path.join(os.path.dirname(static_ffmpeg.__file__), "bin")
    cached = False
    if os.path.isdir(bin_dir):
        for plat in os.listdir(bin_dir):
            crumb = os.path.join(bin_dir, plat, "installed.crumb")
            if os.path.exists(crumb):
                cached = True
                break

    if cached:
        log("ffmpeg: bundled binary already cached, extracting...")
    else:
        log("ffmpeg: downloading bundled binary (~90MB, one-time per venv)...")
        log("ffmpeg: (static-ffmpeg will print its own progress bar)")

    try:
        static_ffmpeg.add_paths(weak=False)
    except Exception as e:
        log(f"ffmpeg: download/extract failed: {e}")
        return None

    p = shutil.which("ffmpeg")
    if p:
        log(f"ffmpeg: {p}  (bundled via static-ffmpeg)")
        return p

    log("ffmpeg: ERROR — bundled binary extracted but not callable")
    return None


if __name__ == "__main__":
    p = resolve_ffmpeg(log_func=lambda m: print(f"[TEST] {m}"))
    print(f"[TEST] resolved: {p}")