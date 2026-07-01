"""Praasper CLI: VAD-Enhanced ASR Framework for Researchers."""

import argparse
import sys
import os


def _validate_path(path):
    """Validate that a path exists (file or directory)."""
    if not os.path.exists(path):
        raise argparse.ArgumentTypeError(f"path does not exist: {path}")
    return path


def _validate_effort(value):
    allowed = {"low", "normal", "high"}
    if value not in allowed:
        raise argparse.ArgumentTypeError(f"effort must be one of {sorted(allowed)}, got {value!r}")
    return value


def _validate_device(value):
    allowed = {"auto", "cuda", "cpu"}
    if value not in allowed:
        raise argparse.ArgumentTypeError(f"device must be one of {sorted(allowed)}, got {value!r}")
    return value


def _validate_infer_mode(value):
    allowed = {"local", "api"}
    if value not in allowed:
        raise argparse.ArgumentTypeError(f"infer-mode must be one of {sorted(allowed)}, got {value!r}")
    return value


def _build_parser():
    parser = argparse.ArgumentParser(
        prog="praasper",
        description="VAD-Enhanced ASR Framework — transcribe audio with word-level timestamps.",
    )
    sub = parser.add_subparsers(dest="command", title="commands")

    # ── transcribe ──────────────────────────────────────────────────────────
    tx = sub.add_parser("transcribe", help="Transcribe audio file(s) to TextGrid")
    tx.add_argument(
        "input", type=_validate_path,
        help="Path to a .wav file or a directory of audio files.",
    )
    tx.add_argument(
        "--device", type=_validate_device, default="auto",
        help="Hardware: auto (default), cuda, or cpu.",
    )
    tx.add_argument(
        "--effort", type=_validate_effort, default="normal",
        help="Grid search depth: low (3 combos), normal (22), high (100). Default: normal.",
    )
    tx.add_argument(
        "--infer-mode", type=_validate_infer_mode, default="local",
        help="ASR backend: local (FunASR-Nano, default) or api (DashScope cloud).",
    )
    tx.add_argument(
        "--api-key", default=None,
        help="DashScope API key (required when --infer-mode=api).",
    )
    tx.add_argument(
        "--asr", default=None,
        help="ASR model name override (advanced). Default: FunAudioLLM/Fun-ASR-Nano-2512.",
    )
    tx.add_argument(
        "--cache-dir", default=None,
        help="Directory for caching ASR models.",
    )
    tx.add_argument(
        "--seg-dur", type=float, default=15.0,
        help="Maximum segment duration in seconds. Default: 15.",
    )
    tx.add_argument(
        "--min-pause", type=float, default=0.2,
        help="Minimum pause between utterances in seconds. Default: 0.2.",
    )
    tx.add_argument(
        "--skip-existing", action="store_true",
        help="Skip files that already have an output TextGrid.",
    )
    tx.add_argument(
        "--verbose", action="store_true",
        help="Print verbose progress messages.",
    )
    tx.add_argument(
        "--params", type=_validate_path, default=None,
        help="Path to a VAD parameters .txt file (skips auto grid search).",
    )

    # ── version ─────────────────────────────────────────────────────────────
    ver = sub.add_parser("version", help="Show Praasper version and exit.")
    return parser


def main(args=None):
    """CLI entry point. args=None uses sys.argv."""
    parser = _build_parser()

    if args is None:
        args = sys.argv[1:]

    parsed = parser.parse_args(args)

    if parsed.command is None:
        parser.print_help()
        return 1

    if parsed.command == "version":
        from importlib.metadata import version
        print(f"praasper {version('praasper')}")
        return 0

    if parsed.command == "transcribe":
        import praasper

        model = praasper.init_model(
            infer_mode=parsed.infer_mode,
            device=parsed.device,
            api_key=parsed.api_key,
            cache_dir=parsed.cache_dir,
            ASR=parsed.asr,
            effort=parsed.effort,
        )

        model.annote(
            input_path=parsed.input,
            seg_dur=parsed.seg_dur,
            min_pause=parsed.min_pause,
            skip_existing=parsed.skip_existing,
            verbose=parsed.verbose,
            params=parsed.params,
        )
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
