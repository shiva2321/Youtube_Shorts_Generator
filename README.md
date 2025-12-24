# Anime_clips

Generate “important” clips from a video by:
- transcribing audio to an SRT
- scoring candidate highlight windows
- exporting clips (optionally with aspect crop/pad + target resolution)

This repo is Windows-first (paths are currently hardcoded in `auto_important_clips.py`).

## Setup

Create/activate your venv, then install deps:

```powershell
# From repo root
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
pip install -r requirements.txt
```

## GPU transcription (RTX 3060 / CUDA)

For fastest transcription, use **faster-whisper** on CUDA.

Recommended settings:
- `--asr-backend faster-whisper`
- `--asr-device cuda`
- `--asr-compute-type float16`
- `--asr-model small` (good speed/quality)

Example:

```powershell
python auto_important_clips.py --video "C:\path\video.mkv" --num-clips 5 --outdir out_clips --workdir work \
  --asr-backend faster-whisper --asr-device cuda --asr-compute-type float16 --asr-model small
```

Notes:
- The first run downloads the model; later runs reuse it.
- Transcripts are cached to `work/<video>.srt` with config metadata in `work/<video>.asr.json`.

## Parallel processing (stable defaults)

- Clip exporting uses parallel workers via `--jobs`.
- When NVENC is used, the script caps parallel exports to **3** to avoid GPU contention.

## Cropping / aspect / target resolution

- `--aspect source` preserves the original frame (no crop/pad)
- `--aspect-mode fit` pads to the requested aspect (no cropping)
- `--aspect-mode fill` crops center to the requested aspect
- `--aspect-mode smart` chooses `fill` for action-heavy clips and `fit` for calmer clips
- `--target-res WxH` (e.g. `1080x1920`) scales the final output to exactly that size

Example vertical 9:16 @ 1080x1920:

```powershell
python auto_important_clips.py --video "C:\path\video.mkv" --num-clips 5 --outdir out_clips --workdir work \
  --aspect 9:16 --aspect-mode smart --target-res 1080x1920
```

## Quick self-test

```powershell
python self_test.py
```

Optional end-to-end test:

```powershell
python self_test.py --video "C:\path\video.mkv"
```

## UI

Run:

```powershell
python clip_generator_ui.py
```

## Portable setup (no hard-coded paths)

`auto_important_clips.py` now discovers tools at runtime.

### ffmpeg / ffprobe

Preferred:
- Install ffmpeg and ensure `ffmpeg` and `ffprobe` are available on `PATH`.

Or set explicit paths (recommended for CI / portability):

```powershell
$env:ANIMECLIPS_FFMPEG  = "C:\\path\\to\\ffmpeg.exe"
$env:ANIMECLIPS_FFPROBE = "C:\\path\\to\\ffprobe.exe"
```

### whisper.cpp (optional)

Only needed if you choose `--asr-backend whispercpp`.

```powershell
$env:ANIMECLIPS_WHISPER_CLI       = "C:\\path\\to\\whisper-cli.exe"
$env:ANIMECLIPS_WHISPER_MODEL     = "C:\\path\\to\\ggml-medium.bin"
$env:ANIMECLIPS_WHISPER_VAD_MODEL = "C:\\path\\to\\ggml-silero-v5.1.2.bin"
```

If you don’t set these, just use `--asr-backend faster-whisper`.

### GPU ASR (Windows)

GPU ASR requires cuDNN DLLs to be discoverable on PATH. If CUDA init fails, the app will automatically fall back to CPU.

Run diagnostics:

```powershell
python asr_diagnose.py
```
