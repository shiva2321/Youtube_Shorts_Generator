"""Self-test utilities for Anime_clips.

This script is meant to quickly verify:
- faster-whisper loads on CUDA
- ffmpeg has NVENC available
- the core script accepts key flags

Run:
  python self_test.py

Optional:
  python self_test.py --video "path/to/video.mkv"

If --video is provided, it will do a tiny 5-second export to validate crop/scale filters.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys


def _run(cmd: list[str]) -> tuple[int, str]:
    p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="ignore")
    out = (p.stdout or "") + (p.stderr or "")
    return p.returncode, out


def _print_section(title: str):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def check_faster_whisper_cuda() -> bool:
    _print_section("faster-whisper CUDA")
    try:
        import ctranslate2  # type: ignore

        n = ctranslate2.get_cuda_device_count() if hasattr(ctranslate2, "get_cuda_device_count") else -1
        print("ctranslate2 cuda device count:", n)
    except Exception as e:
        print("ctranslate2 import failed:", type(e).__name__, e)
        return False

    try:
        from faster_whisper import WhisperModel

        WhisperModel("small", device="cuda", compute_type="float16")
        print("CUDA OK (loaded model on GPU)")
        return True
    except Exception as e:
        print("CUDA FAILED:", type(e).__name__, e)
        return False


def check_ffmpeg_nvenc(ffmpeg_path: str) -> bool:
    _print_section("ffmpeg NVENC")
    if not os.path.exists(ffmpeg_path):
        print("ffmpeg not found:", ffmpeg_path)
        return False

    rc, out = _run([ffmpeg_path, "-hide_banner", "-encoders"])
    if rc != 0:
        print("ffmpeg -encoders failed")
        print(out)
        return False

    ok = "h264_nvenc" in out
    print("h264_nvenc present:", ok)
    return ok


def check_cli_help(py: str, script: str) -> bool:
    _print_section("auto_important_clips.py CLI flags")
    rc, out = _run([py, script, "-h"])
    if rc != 0:
        print("-h failed")
        print(out)
        return False

    required_flags = [
        "--target-res",
        "--asr-backend",
        "--asr-model",
        "--asr-device",
        "--asr-compute-type",
        "--asr-threads",
    ]
    ok = True
    for f in required_flags:
        hit = f in out
        print(f"{f}: {hit}")
        ok &= hit

    if not ok:
        print("\nHelp output was:\n")
        print(out)

    return ok


def tiny_export(py: str, script: str, video: str, outdir: str) -> bool:
    _print_section("tiny export (5s)")
    os.makedirs(outdir, exist_ok=True)
    cmd = [
        py,
        script,
        "--video",
        video,
        "--num-clips",
        "1",
        "--skip-intro-sec",
        "0",
        "--skip-outro-sec",
        "0",
        "--len-preset",
        "60-90",
        "--outdir",
        outdir,
        "--workdir",
        os.path.join(outdir, "work"),
        "--language",
        "auto",
        "--asr-backend",
        "faster-whisper",
        "--asr-device",
        "cuda",
        "--asr-compute-type",
        "float16",
        "--aspect",
        "9:16",
        "--aspect-mode",
        "fit",
        "--target-res",
        "720x1280",
        "--quality",
        "fast",
        "--jobs",
        "1",
        "--encoder",
        "auto",
    ]
    print("RUN:", " ".join(cmd))
    p = subprocess.run(cmd)
    print("exit code:", p.returncode)
    return p.returncode == 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="", help="Optional: run a tiny end-to-end export using this video")
    args = ap.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    script = os.path.join(base_dir, "auto_important_clips.py")

    # Keep consistent with the repo's hardcoded ffmpeg in auto_important_clips.py.
    # If you move it later, update both.
    from auto_important_clips import FFMPEG  # noqa: E402

    py = sys.executable

    ok = True
    ok &= check_cli_help(py, script)
    ok &= check_ffmpeg_nvenc(FFMPEG)
    ok &= check_faster_whisper_cuda()

    if args.video:
        ok &= tiny_export(py, script, args.video, os.path.join(base_dir, "_selftest_out"))

    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())

