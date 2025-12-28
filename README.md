# Anime Clips Generator

Automatically generate engaging short-form video clips from anime (or any video) by intelligently selecting and formatting key moments. Perfect for creating YouTube Shorts, TikTok, or Instagram Reels content.

## 📋 Features

### Core Functionality
- **Intelligent Clip Selection**: Transcribes video audio, analyzes subtitles, and scores candidate clips based on engagement potential (questions, exclamations, dialogue intensity)
- **Smart Video Processing**: Scales to target resolution (e.g., 1080x1920), applies aspect fitting/filling/cropping
- **Multi-threaded Export**: Parallel FFmpeg processing optimized for RTX 3060 and other GPUs
- **Anime-Specific Template**: Auto-detects anime content and applies contextual emoji overlays (⚔️ 🔥 😳 🤯 for relevant scenes)
- **Configurable Overlays**: Top/bottom text bars with gradient support, ASS subtitle styling, colored CTAs (Subscribe/Like/Channel)
- **Language Detection**: Auto-detects Japanese vs. English audio for optimal processing

### Recent Improvements (December 2025)
- ✅ **Varied emoji support** for anime clips (not just ⭐; context-aware based on text keywords)
- ✅ **Colored CTA overlays** (Subscribe/Like keywords use accent colors via ASS tags)
- ✅ **No truncation on overlays** (bottom text preserved in full, never shortened to "Subsc...")
- ✅ **Proper ASS subtitle handling** (respects override tags, wraps correctly)
- ✅ **Each clip is independent** (target-length = length per clip, NOT total duration)

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# Create virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install packages
pip install -r requirements.txt
```

### 2. Basic Usage

```bash
# Generate 5 clips of 3 minutes each from anime video
python auto_important_clips.py \
  --video "path/to/Naruto_030.mp4" \
  --outdir "output/clips" \
  --num-clips 5 \
  --target-length 180 \
  --template-style anime \
  --aspect-mode fit \
  --target-res 1080x1920 \
  --channel-name "@MyChannel"
```

### 3. Launch UI (Recommended)

```bash
python clip_generator_ui.py
```

---

## 🎯 Key Command-Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--video` | str | **Required** | Path to input video file |
| `--outdir` | str | **Required** | Output directory for clips |
| `--num-clips` | int | 5 | Number of clips to generate |
| `--target-length` | int | 180 (sec) | **Length per clip** (NOT total) |
| `--template-style` | str | viral_shorts | Overlay style: `simple`, `viral_shorts`, `anime`, `neon_vibes`, `cinematic` |
| `--aspect-mode` | str | fit | `fit` (pad), `fill` (crop) |
| `--target-res` | str | 1080x1920 | Output resolution (e.g., `1080x1920` for vertical) |
| `--channel-name` | str | "" | Your channel name (for CTA: "Subscribe @YourChannel") |
| `--language` | str | auto | `auto`, `english`, `japanese` |
| `--hook-seed` | int | 0 | Seed for deterministic emoji selection (increment to vary) |
| `--encoder` | str | auto | `auto`, `libx264`, `h264_nvenc` |
| `--jobs` | int | 2 | Parallel export workers |
| `--overlay-top-text` | str | {hook} | Template for top text (supports {hook}, {punchline}, {primary}, {channel}) |
| `--overlay-bottom-text` | str | {punchline} | Template for bottom text |

---

## 🎨 Overlay Templates

### Available Templates

| Name | Style | Best For |
|------|-------|----------|
| `simple` | Minimal shadow text | Clean, minimal look |
| `viral_shorts` | Gradient blue with emojis | High-energy shorts |
| `anime` | Solid blue bars + varied emojis | Anime clips with context emojis |
| `neon_vibes` | Neon pink/purple gradient | Trendy, eye-catching |
| `cinematic` | Dark subtle bars | Professional/cinematic |
| `glass_modern` | Transparent modern look | Modern aesthetic |

### Example: Using Anime Template

```bash
python auto_important_clips.py \
  --video "anime.mkv" \
  --num-clips 5 \
  --target-length 180 \
  --template-style anime \
  --channel-name "@MyAnimeChannel" \
  --hook-seed 42  # Deterministic emoji selection
```

The `anime` template will:
- Detect content type and use blue overlay bars
- Pick **context-aware emojis** based on clip text
  - `⚔️ 🔥 💥 ⚡` for fight scenes
  - `🤯 😳 😱` for shock moments
  - `😂 🤣 💀` for funny scenes
  - `🥺 😢 💔` for sad scenes
  - etc.
- Color Subscribe/Like CTAs for visibility (accent color for keyword, white for channel)
- Wrap text intelligently to prevent overflow

---

## 💾 Project Structure

```
Anime_clips/
├── auto_important_clips.py       # Core clip generation engine (~1600 lines)
├── clip_generator_ui.py          # GUI interface (Tkinter-based)
├── check_cuda.py                 # CUDA diagnostics
├── asr_diagnose.py               # ASR (audio transcription) diagnostics
├── self_test.py                  # Self-test suite
├── requirements.txt              # Python dependencies
├── README.md                      # Original readme
├── README_NEW.md                  # This file (comprehensive)
├── FUTURE_IMPROVEMENTS.md        # Planned enhancements
└── output/                       # Generated clips & transcripts
    └── <VideoName>/
        ├── *_clip_01_*.mp4
        ├── *_clip_02_*.mp4
        ├── *_transcript.srt
        └── *_summary.txt
```

---

## 📊 How Clip Duration Works

**IMPORTANT**: `--target-length` specifies the length of **each individual clip**, NOT the total.

Examples:

```bash
# Generate 5 clips, each 3 minutes long (15 minutes total)
--num-clips 5 --target-length 180

# Generate 10 clips, each 60 seconds long (10 minutes total)
--num-clips 10 --target-length 60

# Generate 20 clips, each 15 seconds long (5 minutes total) [TikTok-friendly]
--num-clips 20 --target-length 15
```

---

## 🔧 Configuration

### FFmpeg Path
If FFmpeg is not on PATH:
```bash
# Set environment variable (PowerShell)
$env:FFMPEG = "C:\path\to\ffmpeg.exe"
```

### GPU Selection
- **NVIDIA (CUDA)**: Automatically detects and uses `h264_nvenc` if available
- **CPU**: Falls back to `libx264` (slower but works anywhere)
- **Force specific encoder**: `--encoder libx264` or `--encoder h264_nvenc`

### Parallel Jobs
For RTX 3060 with NVENC:
```bash
--jobs 3  # Cap at 3 to avoid GPU memory contention
```

For CPU:
```bash
--jobs 4  # Use 4 CPU workers
```

---

## 📊 Output

After processing, you'll find in `output/<VideoName>/`:

| File | Purpose |
|------|---------|
| `*_clip_01_*.mp4` | Exported clip #1 (H.264 video, AAC audio) |
| `*_clip_02_*.mp4` | Exported clip #2 |
| `*_transcript.srt` | Full video transcript (SRT format) |
| `*_summary.txt` | Metadata: clip count, duration, hooks, punchlines |

Example summary:
```
Video: Naruto_030 - The Sharingan Revived.mkv
Processed: 2025-12-28 14:32:01
Model: small
Language: english
Resolution: 1080x1920
Aspect Mode: fit
Requested Clips: 5
Generated Clips: 5
Clip Length (per clip): 180s
Total Duration: 900.0 seconds (15.0 minutes)

Clips:
Clip 1:
  Time: 00:00:15.000 --> 00:03:15.000 (180.0s)
  Hook: Wait for it…
  Punchline: This changes everything...
  Full text: [transcript excerpt]
```

---

## 🐛 Troubleshooting

### No clips exported
- Check `work/log_*.txt` files for FFmpeg errors
- Ensure FFmpeg is installed and on PATH
- Verify input video is accessible and valid codec
- Run `python check_cuda.py` to diagnose GPU issues

### Emojis not displaying
- **Use `--template-style anime`** (other templates may have limited emoji support)
- Check that output clips play in a player that supports emoji rendering
- Verify Arial font is installed (default emoji-aware font on Windows)

### Slow processing
- Use GPU encoding: `--encoder h264_nvenc` (NVIDIA RTX series)
- Reduce `--target-res` (e.g., `720x1280` instead of `1080x1920`)
- Increase `--jobs` (but cap at 3-4 for NVENC to avoid memory overflow)
- Use smaller Whisper model: `--whisper-model base`

### Audio language not detected
- Manually set: `--language english` or `--language japanese`
- Run `python asr_diagnose.py` to debug ASR issues

### Overlay text truncated or corrupted
- This was fixed in December 2025; if you see "Subsc..." instead of "Subscribe @Channel", rebuild ASS files
- Try regenerating clips with latest code
- Check `work/overlay_*.ass` files for ASS syntax

---

## ✂️ Smart Intro/Outro Detection (Beta)

Some episodes place important scenes *before* the intro (cold open) or *after* the outro (post-credits), so a simple “skip first N seconds” can miss good moments.

This project now supports a **smart intro/outro detector** that (when enabled):
- **Avoids transcribing** the OP/ED to save Whisper time (`--smart-trim`)
- **Avoids selecting/exporting clips** that overlap detected OP/ED ranges (`--auto-skip-intro-outro`)

### Optional dependency (recommended)

For best results, install `inaSpeechSegmenter` (it uses TensorFlow):

```bash
pip install inaSpeechSegmenter tensorflow
```

If it’s not installed, the app **falls back gracefully** and simply won’t skip anything.

### CLI usage

```bash
python auto_important_clips.py \
  --video "path/to/episode.mp4" \
  --outdir "output/episode" \
  --num-clips 5 \
  --target-length 180 \
  --smart-trim \
  --auto-skip-intro-outro
```

Notes:
- Timestamps in the generated transcript are still aligned to the **original video**.
- You can still force manual skips with `--skip-intro` / `--skip-outro`.

### UI

In the **Input & Transcription** tab, enable:
- **“Smartly remove intro/outro (Beta)”**

---

## 📝 Examples

### Example 1: Quick TikTok Clips (15 seconds each)
```bash
python auto_important_clips.py \
  --video "episode.mkv" \
  --outdir "clips_tiktok" \
  --num-clips 10 \
  --target-length 15 \
  --template-style viral_shorts \
  --target-res 1080x1920 \
  --channel-name "@MyAnimeChannel"
```

**Result**: 10 clips × 15 sec = 150 sec total

### Example 2: YouTube Shorts (60 seconds each)
```bash
python auto_important_clips.py \
  --video "episode.mkv" \
  --outdir "clips_youtube" \
  --num-clips 8 \
  --target-length 60 \
  --template-style anime \
  --target-res 1080x1920 \
  --channel-name "@MyChannel" \
  --hook-seed 99
```

**Result**: 8 clips × 60 sec = 480 sec total

### Example 3: Professional Cinematic (2 minutes each)
```bash
python auto_important_clips.py \
  --video "movie.mkv" \
  --outdir "professional_clips" \
  --num-clips 5 \
  --target-length 120 \
  --template-style cinematic \
  --aspect-mode fill \
  --target-res 1920x1080 \
  --encoder h264_nvenc
```

**Result**: 5 clips × 120 sec = 600 sec total

### Example 4: Long-Form Commentary (3 minutes each)
```bash
python auto_important_clips.py \
  --video "interview.mkv" \
  --outdir "commentary_clips" \
  --num-clips 5 \
  --target-length 180 \
  --template-style glass_modern \
  --channel-name "@Commentary" \
  --jobs 3
```

**Result**: 5 clips × 180 sec = 900 sec total

---

## 🔬 Testing

Run self-tests:
```bash
python self_test.py
python check_cuda.py
python asr_diagnose.py
```

---

## 🏗️ Architecture Overview

### Processing Pipeline

1. **Input Validation** → Check video exists and is accessible
2. **Language Detection** → Sample audio to identify Japanese vs. English
3. **Transcription** → Extract audio, transcribe to SRT using Whisper (small model by default)
4. **Candidate Generation** → Parse SRT, merge adjacent subtitles into clip-sized chunks
5. **Clip Scoring** → Heuristic scoring based on:
   - Keywords (fight, secret, surprise, funny, etc.)
   - Punctuation (questions, exclamations)
   - Clip duration match to target
   - Position in video (avoid intro/outro)
6. **Clip Selection** → Select top N clips while avoiding temporal overlap
7. **Filter Chain Building** → Generate FFmpeg complex filter for each clip:
   - Scaling & aspect ratio handling
   - Background blur (fit mode)
   - Colored bars (top/bottom)
   - ASS subtitle overlay (with emoji + colored CTA)
8. **Parallel Export** → Multi-threaded FFmpeg encoding with GPU acceleration
9. **Cleanup** → Remove temp files, generate summary report

### Key Classes & Functions

**Data Structures**:
- **`OverlayTemplate`**: Dataclass defining visual style (colors, text, emojis, opacity, ASS colors)
- **`Candidate`**: Dataclass for clip metadata (start, end, text, score)

**Core Functions**:
- **`build_filter_chain()`**: Constructs FFmpeg complex filter for single clip (1200+ lines)
- **`_pick_top_emojis_for_text()`**: Context-aware emoji selection based on clip content keywords
- **`_style_bottom_text_ass()`**: ASS coloring for Subscribe/Like CTAs (preserves full text)
- **`_wrap_for_ass()`**: Smart wrapping for ASS subtitles (handles override tags)
- **`compute_heuristic_score()`**: Scores clips based on engagement signals
- **`main()`**: CLI entry point with argparse + orchestration

---

## 🚦 Current Status

| Feature | Status | Notes |
|---------|--------|-------|
| Core clip generation | ✅ Stable | Tested on RTX 3060, handles anime well |
| Emoji overlays | ✅ Complete | Context-aware (fight/shock/funny), varied per clip |
| CTA coloring | ✅ Complete | Subscribe/Like/Channel with accent colors |
| ASS subtitle support | ✅ Fixed | No truncation, proper escape handling |
| Per-clip duration | ✅ Working | Each clip independent length (not total) |
| UI | ✅ Functional | Tkinter-based, supports all major options |
| CUDA optimization | ✅ Working | NVENC support with memory limits (3 workers max) |
| Language detection | ✅ Accurate | Whisper-based multi-sample detection |
| Parallel export | ✅ Optimized | ThreadPoolExecutor with GPU-aware worker limits |

---

## 📚 See Also

- **FUTURE_IMPROVEMENTS.md** — Planned enhancements (batch processing, ML scoring, analytics, etc.)
- **auto_important_clips.py** — Source code with inline comments
- **clip_generator_ui.py** — GUI launcher script

