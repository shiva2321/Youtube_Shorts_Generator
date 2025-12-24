"""Quick ASR/CUDA sanity checks for this project.

Run:
  python asr_diagnose.py

It checks:
- faster-whisper CUDA init
- whether cuDNN 9 DLLs are discoverable on PATH
- ctranslate2 CUDA device count

This is intentionally small and has no external deps beyond your venv.
"""

from __future__ import annotations

import os
import sys


CUDNN_DLLS = [
    "cudnn_ops64_9.dll",
    "cudnn_cnn64_9.dll",
    "cudnn64_9.dll",
]


def _find_on_path(filename: str) -> str | None:
    for p in (os.environ.get("PATH") or "").split(os.pathsep):
        p = (p or "").strip().strip('"')
        if not p:
            continue
        cand = os.path.join(p, filename)
        if os.path.exists(cand):
            return cand
    return None


def _shorten_path(p: str, max_len: int = 120) -> str:
    p = p.replace("\\", "/")
    return p if len(p) <= max_len else (p[:50] + "..." + p[-60:])


def main() -> int:
    print("python:", sys.version)

    print("\n[env] PATH entries:")
    for i, p in enumerate((os.environ.get("PATH") or "").split(os.pathsep), start=1):
        p = (p or "").strip().strip('"')
        if not p:
            continue
        print(f"  {i:02d}. {_shorten_path(p)}")

    print("\n[DLL check] cuDNN on PATH")
    missing = []
    for dll in CUDNN_DLLS:
        hit = _find_on_path(dll)
        if hit:
            print("  FOUND", dll, "->", hit)
        else:
            print("  MISSING", dll)
            missing.append(dll)

    print("\n[ctranslate2] CUDA device count")
    try:
        import ctranslate2  # type: ignore

        n = ctranslate2.get_cuda_device_count() if hasattr(ctranslate2, "get_cuda_device_count") else -1
        print("  cuda devices:", n)
    except Exception as e:
        print("  ctranslate2 import failed:", type(e).__name__, e)
        n = 0

    print("\n[faster-whisper] CUDA init")
    try:
        from faster_whisper import WhisperModel

        WhisperModel("small", device="cuda", compute_type="float16")
        print("  CUDA OK")
    except Exception as e:
        print("  CUDA FAILED:", type(e).__name__, e)
        if missing:
            print("\nFix: install cuDNN 9 and add its bin folder to PATH so these DLLs are found:")
            for dll in missing:
                print("  -", dll)
            print("\nWorkaround: run with CPU ASR:")
            print("  --asr-device cpu")
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
