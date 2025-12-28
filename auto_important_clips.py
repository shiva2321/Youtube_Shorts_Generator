import argparse
import os
import re
import subprocess
import sys
import textwrap
import platform
import shutil
import datetime
import warnings
import hashlib
import time
import unicodedata
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple, Optional

# Suppress FP16 warnings on CPU
warnings.filterwarnings("ignore")


# ==========================================
# 1. Configuration & Data Structures
# ==========================================

@dataclass
class OverlayTemplate:
    """Configuration for overlay styles."""
    name: str
    top_bar_color: str = "black"
    bottom_bar_color: str = "black"
    top_bar_opacity: float = 1.0
    bottom_bar_opacity: float = 0.9
    use_gradient: bool = False
    gradient_colors: List[str] = field(default_factory=lambda: ["#000000", "#000000"])
    text_style: str = "outline"  # "outline", "shadow", "glow"
    text_color: str = "white"
    border_width: int = 3
    show_branding: bool = False
    show_subscribe: bool = False
    header_text: str = ""
    emoji_prefix: str = ""
    emoji_suffix: str = ""


OVERLAY_TEMPLATES = {
    "simple": OverlayTemplate(name="simple", top_bar_opacity=0.8, bottom_bar_opacity=0.8, text_style="shadow"),
    "viral_shorts": OverlayTemplate(name="viral_shorts", header_text="LIKE & SUBSCRIBE", show_branding=True,
                                    show_subscribe=True, use_gradient=True, gradient_colors=["#0066ff", "#0044aa"],
                                    text_style="outline", border_width=4, emoji_prefix="🔥 ", emoji_suffix=" 👀"),
    "neon_vibes": OverlayTemplate(name="neon_vibes", use_gradient=True, gradient_colors=["#ff00cc", "#333399"],
                                  text_style="glow", show_branding=True, emoji_prefix="✨ ", emoji_suffix=" ✨"),
    "glass_modern": OverlayTemplate(name="glass_modern", top_bar_color="black", bottom_bar_color="black",
                                    top_bar_opacity=0.4, bottom_bar_opacity=0.4, text_style="shadow", border_width=2,
                                    show_branding=True, emoji_prefix="👉 ", emoji_suffix=" 👈"),
    "cinematic": OverlayTemplate(name="cinematic", top_bar_color="black", bottom_bar_color="black", top_bar_opacity=1.0,
                                 bottom_bar_opacity=1.0, text_style="shadow", text_color="#f0f0f0",
                                 emoji_prefix="", emoji_suffix=""),
    "anime": OverlayTemplate(name="anime", top_bar_color="#0066cc", bottom_bar_color="#0066cc",
                             top_bar_opacity=0.85, bottom_bar_opacity=0.85, text_style="outline",
                             text_color="#ffffff", border_width=4, emoji_prefix="⭐ ", emoji_suffix=" ⭐"),
}


@dataclass
class Candidate:
    start: float
    end: float
    text: str
    score: float = 0.0


# ==========================================
# 2. Whisper Transcription Integration
# ==========================================

def format_timestamp(seconds: float) -> str:
    """Converts seconds to SRT timestamp format (HH:MM:SS,mmm)."""
    td = datetime.timedelta(seconds=seconds)
    total_seconds = int(td.total_seconds())
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    millis = int((seconds - int(seconds)) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def get_video_duration(video_path: str) -> float:
    """Get the duration of a video file using FFmpeg."""
    try:
        cmd = [FFMPEG, "-i", video_path, "-hide_banner"]
        output = subprocess.check_output(cmd, stderr=subprocess.STDOUT, universal_newlines=True)

        # Look for duration in output
        duration_match = re.search(r"Duration: (\d{2}):(\d{2}):(\d{2})\.(\d{2})", output)
        if duration_match:
            hours = int(duration_match.group(1))
            minutes = int(duration_match.group(2))
            seconds = int(duration_match.group(3))
            centiseconds = int(duration_match.group(4))
            return hours * 3600 + minutes * 60 + seconds + centiseconds / 100
        return 0
    except Exception as e:
        # Try another approach with ffprobe
        try:
            cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                   "-of", "default=noprint_wrappers=1:nokey=1", video_path]
            output = subprocess.check_output(cmd, universal_newlines=True).strip()
            return float(output)
        except:
            print(f"Error getting video duration: {e}")
            # Fallback to a reasonable default duration for anime episodes
            return 1400  # ~23 minutes


def detect_audio_language(video_path: str) -> str:
    """
    Accurately detect the language of the video by analyzing audio samples.
    Returns "japanese" for Japanese audio, "english" for English audio,
    or "auto" if detection is uncertain.
    """
    try:
        import whisper
        import torch
        import numpy as np

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[Language Detection] Using {device} for language detection")

        # Extract multiple audio samples for more accurate detection
        # We'll take samples from different parts of the video
        duration = get_video_duration(video_path)

        # Define sample points (skip intro, take from multiple points)
        sample_points = []
        if duration < 300:  # Short video under 5 minutes
            # For short videos, just take one sample from the middle
            sample_points = [(duration * 0.5, min(30, duration * 0.5))]
        else:
            # For longer videos, take multiple samples
            sample_points = [
                (max(90, duration * 0.1), 30),  # 10% in (after intro)
                (duration * 0.4, 30),  # 40% in
                (duration * 0.7, 30)  # 70% in
            ]

        # Load tiny model for quick language detection
        print("[Language Detection] Loading lightweight model for language detection...")
        model = whisper.load_model("tiny", device=device)

        # Process each sample
        language_votes = {}

        for i, (start_time, sample_duration) in enumerate(sample_points):
            # Extract audio sample
            temp_audio = os.path.join(os.path.dirname(video_path), f"temp_lang_detect_{i}.wav")
            print(f"[Language Detection] Extracting sample {i + 1}/{len(sample_points)} at {start_time:.1f}s...")
            extract_audio(video_path, temp_audio, start_time, sample_duration)

            # Detect language
            audio = whisper.load_audio(temp_audio)
            audio = whisper.pad_or_trim(audio)
            mel = whisper.log_mel_spectrogram(audio).to(device)
            _, probs = model.detect_language(mel)

            # Get top language and confidence
            detected_lang = max(probs, key=probs.get)
            confidence = probs[detected_lang]

            print(f"[Language Detection] Sample {i + 1}: {detected_lang} ({confidence:.1%} confidence)")

            # Add to votes, weighted by confidence
            language_votes[detected_lang] = language_votes.get(detected_lang, 0) + confidence

            # Clean up
            if os.path.exists(temp_audio):
                os.remove(temp_audio)

        # Determine final language based on weighted votes
        if language_votes:
            final_lang = max(language_votes, key=language_votes.get)
            total_confidence = sum(language_votes.values())
            final_confidence = language_votes[final_lang] / total_confidence if total_confidence > 0 else 0

            print(f"[Language Detection] Final detection: {final_lang} with {final_confidence:.1%} confidence")

            # Map to our supported languages
            if final_lang == "ja" and final_confidence > 0.4:
                return "japanese"
            elif final_lang == "en" and final_confidence > 0.4:
                return "english"

            # For other languages with high confidence, return auto
            if final_confidence > 0.6:
                print(f"[Language Detection] Detected {final_lang} but using 'auto' for best results")
                return "auto"

        # Clean up GPU memory
        if device == "cuda":
            torch.cuda.empty_cache()

    except Exception as e:
        print(f"[Language Detection] Error during audio language detection: {e}")

    # Default to auto if detection fails or is uncertain
    print("[Language Detection] Could not confidently detect language, using automatic detection")
    return "auto"


def extract_audio(video_path: str, output_path: str, start_time: float = 0, duration: float = 0) -> bool:
    """Extract audio from video for faster processing."""
    try:
        cmd = [FFMPEG, "-y", "-hide_banner", "-loglevel", "error"]

        if start_time > 0:
            cmd.extend(["-ss", str(start_time)])

        cmd.extend(["-i", video_path, "-vn", "-c:a", "pcm_s16le", "-ar", "16000", "-ac", "1"])

        if duration > 0:
            cmd.extend(["-t", str(duration)])

        cmd.append(output_path)

        subprocess.run(cmd, check=True)
        return True
    except Exception as e:
        print(f"Error extracting audio: {e}")
        return False


def transcribe_video(video_path: str, model_size: str, language: str, transcript_output: Optional[str] = None) -> str:
    """
    Transcribes video using OpenAI Whisper with optimized GPU usage.
    Avoids chunking for better quality and performance.
    """
    try:
        import whisper
        import torch
    except ImportError:
        print("\n[!] Error: 'openai-whisper' or 'torch' not installed.")
        print("    Please run: pip install openai-whisper torch")
        sys.exit(1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n[Whisper] Loading model '{model_size}' on {device.upper()}...")

    # Set output path
    if transcript_output:
        srt_path = transcript_output
    else:
        # Default location next to the video
        srt_path = os.path.splitext(video_path)[0] + ".srt"

    # Ensure directory exists
    os.makedirs(os.path.dirname(os.path.abspath(srt_path)), exist_ok=True)

    # Extract audio first - this is much faster than having Whisper read the video
    print("[Whisper] Extracting audio from video (this speeds up processing)...")
    audio_path = os.path.join(os.path.dirname(srt_path), "temp_audio.wav")
    extract_audio(video_path, audio_path)

    # Get video duration
    duration = get_video_duration(video_path)
    print(f"[Whisper] Video duration: {duration:.2f} seconds")

    # Check available GPU memory
    gpu_memory_gb = 0
    if device == "cuda":
        try:
            # Get available GPU memory in GB
            free_memory = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()
            gpu_memory_gb = free_memory / (1024 ** 3)
            print(f"[Whisper] Available GPU memory: {gpu_memory_gb:.2f} GB")
        except:
            print("[Whisper] Could not determine available GPU memory")

    # Determine if we need to process in chunks based on model size and available memory
    # These are conservative estimates for RTX 3060 (6GB)
    memory_requirements = {
        "tiny": 1,  # ~1GB
        "base": 1.5,  # ~1.5GB
        "small": 3,  # ~3GB
        "medium": 5,  # ~5GB
        "large": 10  # ~10GB
    }

    # For long videos, we need more memory
    memory_multiplier = min(1.5, max(1.0, duration / 1800))  # Increase for videos > 30 min
    required_memory = memory_requirements.get(model_size, 5) * memory_multiplier

    # Process in one go if we have enough memory, otherwise use chunking
    use_chunking = (device == "cuda" and gpu_memory_gb > 0 and gpu_memory_gb < required_memory)

    if use_chunking:
        print(
            f"[Whisper] Memory requirements ({required_memory:.1f}GB) exceed available GPU memory, using optimized chunking")
        # Optimize chunk size based on available memory
        chunk_size = min(300, max(60, int(600 * (gpu_memory_gb / required_memory))))
        print(f"[Whisper] Using {chunk_size}s chunks for optimal GPU performance")
    else:
        print("[Whisper] Processing entire audio in one pass for best quality")
        chunk_size = 0  # Process all at once

    try:
        # Load model with appropriate precision for GPU
        model = whisper.load_model(model_size, device=device)

        # Determine precision based on model size and device
        use_half = False
        if device == "cuda":
            # For RTX 3060, use FP16 only for tiny and base models
            if model_size in ["tiny", "base"]:
                try:
                    model = model.half()
                    use_half = True
                    print("[Whisper] Using half precision (FP16) for faster processing")
                except Exception as e:
                    print(f"[Whisper] Warning: Could not use half precision: {e}")
                    print("[Whisper] Falling back to full precision (FP32)")
            else:
                print("[Whisper] Using full precision (FP32) for better accuracy with larger model")

        # Language handling
        lang_arg = None if language == "auto" else language
        if language == "japanese": lang_arg = "ja"
        if language == "english": lang_arg = "en"

        print(f"[Whisper] Transcribing with language setting: {language}")

        if chunk_size <= 0 or duration <= chunk_size:
            # Process entire audio in one go
            start_time = time.time()

            # Use more aggressive compression for long files to fit in memory
            if duration > 1800 and device == "cuda":  # > 30 minutes
                print("[Whisper] Long audio detected, using memory-efficient processing")
                # For very long files, we need to be careful with memory
                result = model.transcribe(
                    audio_path,
                    language=lang_arg,
                    verbose=False,
                    fp16=use_half,
                    beam_size=3  # Smaller beam size to save memory
                )
            else:
                # Standard processing for shorter files
                result = model.transcribe(
                    audio_path,
                    language=lang_arg,
                    verbose=False,
                    fp16=use_half
                )

            elapsed = time.time() - start_time
            print(f"[Whisper] Transcription completed in {elapsed:.2f} seconds")

            # Report detected language if auto was used
            if language == "auto" and "language" in result:
                detected_code = result.get("language", "unknown")
                language_names = {
                    "en": "English",
                    "ja": "Japanese",
                    "zh": "Chinese",
                    "ko": "Korean",
                    "fr": "French",
                    "de": "German",
                    "es": "Spanish",
                    "ru": "Russian"
                }
                detected_name = language_names.get(detected_code, f"Unknown ({detected_code})")
                print(f"[Whisper] Detected language: {detected_name}")

            # Write SRT
            with open(srt_path, "w", encoding="utf-8") as f:
                for i, segment in enumerate(result["segments"]):
                    start = format_timestamp(segment["start"])
                    end = format_timestamp(segment["end"])
                    text = segment["text"].strip()
                    f.write(f"{i + 1}\n{start} --> {end}\n{text}\n\n")
        else:
            # Process in optimized chunks with overlap for better transitions
            print(f"[Whisper] Processing in optimized chunks with overlap for seamless transitions")

            all_segments = []
            detected_language = None
            overlap = 5  # 5 second overlap between chunks

            # Process video in chunks with overlap
            for chunk_start in range(0, int(duration), chunk_size - overlap):
                # Adjust last chunk to not exceed duration
                chunk_duration = min(chunk_size, duration - chunk_start)
                if chunk_duration <= overlap:
                    break  # Skip processing if this chunk would be too small

                print(
                    f"[Whisper] Processing chunk {chunk_start / 60:.1f}-{(chunk_start + chunk_duration) / 60:.1f} minutes...")

                # Extract audio chunk
                chunk_audio = os.path.join(os.path.dirname(srt_path), f"temp_chunk_{chunk_start}.wav")
                extract_audio(video_path, chunk_audio, chunk_start, chunk_duration)

                # Transcribe chunk
                start_time = time.time()
                chunk_result = model.transcribe(chunk_audio, language=lang_arg, verbose=False, fp16=use_half)
                elapsed = time.time() - start_time

                # Store detected language from first chunk if using auto
                if chunk_start == 0 and language == "auto" and "language" in chunk_result:
                    detected_language = chunk_result.get("language")

                # Adjust timestamps to account for chunk position
                for segment in chunk_result["segments"]:
                    # Adjust timestamps
                    segment["start"] += chunk_start
                    segment["end"] += chunk_start

                    # Only include segments that start within this chunk's primary range
                    # (excluding the overlap region with the next chunk)
                    if chunk_start == 0 or segment["start"] >= chunk_start:
                        # For chunks after the first one, exclude segments that start in the overlap region
                        # with the previous chunk (they were already included)
                        if chunk_start == 0 or segment["start"] >= chunk_start + overlap:
                            all_segments.append(segment)

                # Clean up chunk audio
                os.remove(chunk_audio)

                print(f"[Whisper] Chunk processed in {elapsed:.2f} seconds")

                # Free up GPU memory
                torch.cuda.empty_cache()

            # Report detected language if auto was used
            if detected_language:
                language_names = {
                    "en": "English",
                    "ja": "Japanese",
                    "zh": "Chinese",
                    "ko": "Korean",
                    "fr": "French",
                    "de": "German",
                    "es": "Spanish",
                    "ru": "Russian"
                }
                detected_name = language_names.get(detected_language, f"Unknown ({detected_language})")
                print(f"[Whisper] Detected language: {detected_name}")

            # Sort segments by start time
            all_segments.sort(key=lambda x: x["start"])

            # Write combined SRT
            with open(srt_path, "w", encoding="utf-8") as f:
                for i, segment in enumerate(all_segments):
                    start = format_timestamp(segment["start"])
                    end = format_timestamp(segment["end"])
                    text = segment["text"].strip()
                    f.write(f"{i + 1}\n{start} --> {end}\n{text}\n\n")
    finally:
        # Clean up
        if os.path.exists(audio_path):
            os.remove(audio_path)

        # Final GPU cleanup
        if device == "cuda":
            torch.cuda.empty_cache()

    print(f"[Whisper] Saved transcript to {srt_path}")
    return srt_path


def parse_srt(srt_path: str) -> List[Candidate]:
    """Parses an SRT file into Candidate objects."""
    candidates = []
    if not os.path.exists(srt_path):
        return candidates

    with open(srt_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Regex to split SRT blocks
    blocks = re.split(r'\n\n+', content.strip())

    for block in blocks:
        lines = block.split('\n')
        if len(lines) >= 3:
            # Parse Time: 00:00:00,000 --> 00:00:05,000
            times = lines[1].split(' --> ')
            if len(times) != 2: continue

            def time_to_sec(t_str):
                h, m, s = t_str.replace(',', '.').split(':')
                return int(h) * 3600 + int(m) * 60 + float(s)

            try:
                start = time_to_sec(times[0])
                end = time_to_sec(times[1])
                text = " ".join(lines[2:])
                candidates.append(Candidate(start, end, text))
            except:
                continue
    return candidates


# ==========================================
# 3. System & Tool Utilities
# ==========================================

def get_system_font_path(font_name: str) -> str:
    """Smart cross-platform font finder."""
    if os.path.exists(font_name): return os.path.abspath(font_name)
    system = platform.system()
    search_dirs = []
    if system == "Windows":
        search_dirs = [os.path.join(os.environ["WINDIR"], "Fonts")]
    elif system == "Darwin":
        search_dirs = ["/Library/Fonts", "/System/Library/Fonts"]
    else:
        search_dirs = ["/usr/share/fonts", os.path.expanduser("~/.fonts")]

    # For Japanese/CJK support, prioritize these fonts
    if any(c for c in font_name if unicodedata.east_asian_width(c) in ('F', 'W')):
        priority_fonts = ["yumin", "msgothic", "meiryo", "hiragino", "noto", "droid"]
        for d in search_dirs:
            for root, _, files in os.walk(d):
                for f in files:
                    if f.lower().endswith((".ttf", ".otf")):
                        for pf in priority_fonts:
                            if pf in f.lower():
                                return os.path.join(root, f)

    target = font_name.lower().replace(" ", "")
    for d in search_dirs:
        for root, _, files in os.walk(d):
            for f in files:
                if f.lower().endswith((".ttf", ".otf")) and target in f.lower().replace("-", ""):
                    return os.path.join(root, f)

    # Fallback to a system font that supports CJK if needed
    for d in search_dirs:
        for root, _, files in os.walk(d):
            for f in files:
                if f.lower().endswith((".ttf", ".otf")) and any(x in f.lower() for x in ["arial", "helvetica", "sans"]):
                    return os.path.join(root, f)
    return ""


def resolve_exe(tool_name: str) -> str:
    """Find executable."""
    path = shutil.which(tool_name)
    if path: return path
    if platform.system() == "Windows":
        common = [r"C:\Program Files\ffmpeg\bin\ffmpeg.exe", r"C:\ffmpeg\bin\ffmpeg.exe"]
        for p in common:
            if os.path.exists(p): return p
    raise FileNotFoundError(f"Could not find {tool_name}.")


def get_video_hash(video_path: str) -> str:
    """Generate a unique hash for the video file to identify it."""
    if not os.path.exists(video_path):
        return "unknown"

    # Use file size and name as a simple hash
    file_size = os.path.getsize(video_path)
    basename = os.path.basename(video_path)
    hash_input = f"{basename}_{file_size}"
    return hashlib.md5(hash_input.encode()).hexdigest()[:10]


def sanitize_filename(filename: str) -> str:
    """Sanitize a string to be used as a filename."""
    # Remove invalid characters
    s = re.sub(r'[\\/*?:"<>|]', "", filename)
    # Replace spaces with underscores
    s = s.replace(' ', '_')
    # Limit length
    return s[:100]


def escape_text_for_ffmpeg(text: str) -> str:
    """Escape text for FFmpeg drawtext filter."""
    # Escape special characters
    text = text.replace("'", "\\'").replace(':', '\\:').replace(',', '\\,')
    return text


# ==========================================
# 4. Text Processing (Hooks/Punchlines)
# ==========================================

def _smart_extract(text: str, mode: str) -> str:
    """Heuristic extraction for hooks and punchlines."""
    # Split by punctuation
    sentences = re.split(r'(?<=[.!?。！？])\s*', text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 2]

    if not sentences: return text[:50]

    if mode == "hook":
        # Strategy: Questions or Exclamations first
        for s in sentences[:4]:
            if any(p in s for p in ["?", "？"]) and len(s) < 60: return s
        for s in sentences[:4]:
            if any(p in s for p in ["!", "！"]) and len(s) < 50: return s
        return sentences[0]

    elif mode == "punchline":
        # Strategy: Last sentence usually holds the conclusion
        return sentences[-1] if len(sentences[-1]) < 80 else sentences[-1][:80]

    return text


def get_hook_punchline(text: str) -> Tuple[str, str]:
    """Extract hook and punchline from text."""
    # For Japanese content, we need special handling
    is_japanese = any(unicodedata.east_asian_width(c) in ('F', 'W') for c in text)

    if is_japanese:
        # Japanese content typically has shorter sentences
        # Just use first and last parts directly
        parts = text.split('。')
        if len(parts) >= 2:
            hook = parts[0] + '。' if not parts[0].endswith('。') else parts[0]
            punchline = parts[-1]
            return hook, punchline

    # Default extraction for other languages
    return _smart_extract(text, "hook"), _smart_extract(text, "punchline")


# ==========================================
# 5. FFmpeg Filter Construction
# ==========================================

def build_filter_chain(clip: Candidate, idx: int, cfg: argparse.Namespace, resolution: Tuple[int, int]) -> str:
    target_w, target_h = resolution
    filters = []

    # 1. Scale/Crop
    if cfg.aspect_mode == "fill":
        filters.append(f"scale={target_w}:{target_h}:force_original_aspect_ratio=increase,crop={target_w}:{target_h}")
    else:  # fit
        filters.append(
            f"scale={target_w}:{target_h}:force_original_aspect_ratio=decrease,pad={target_w}:{target_h}:(ow-iw)/2:(oh-ih)/2:color=black")

    # 2. Overlay
    if cfg.template_style:
        tpl = OVERLAY_TEMPLATES.get(cfg.template_style, OVERLAY_TEMPLATES["simple"])
        hook, punchline = get_hook_punchline(clip.text)

        # Format text with emojis (only once)
        try:
            top_txt = cfg.overlay_top_text.format(hook=hook, punchline=punchline, channel=cfg.channel_name, i=idx)
            bot_txt = cfg.overlay_bottom_text.format(hook=hook, punchline=punchline, channel=cfg.channel_name, i=idx)

            # Add emojis only once
            if tpl.emoji_prefix and not top_txt.startswith(tpl.emoji_prefix):
                top_txt = f"{tpl.emoji_prefix}{top_txt}"
            if tpl.emoji_suffix and not top_txt.endswith(tpl.emoji_suffix):
                top_txt = f"{top_txt}{tpl.emoji_suffix}"
        except:
            top_txt, bot_txt = hook, punchline

        # Escape text for FFmpeg
        top_txt_escaped = escape_text_for_ffmpeg(top_txt)
        bot_txt_escaped = escape_text_for_ffmpeg(bot_txt)

        # Check if text contains Japanese/CJK characters
        has_cjk = any(unicodedata.east_asian_width(c) in ('F', 'W') for c in top_txt + bot_txt)

        # Get appropriate font for text content (especially for CJK characters)
        font_path = get_system_font_path(cfg.overlay_font)
        if not font_path and has_cjk:
            # If text contains CJK characters and no font specified, try to find a suitable one
            font_path = get_system_font_path("Noto Sans CJK")

        # Fix the font path format for FFmpeg
        if font_path:
            font_path = font_path.replace('\\', '/').replace(':', '\\:')
            font_arg = f":fontfile='{font_path}'"
        else:
            font_arg = ""

        # Adjust font size based on content type
        fs_base = int(target_h * 0.05)  # Base font size
        if has_cjk:
            fs_base = int(target_h * 0.04)  # Smaller for CJK characters

        # Bars - improved positioning
        if tpl.use_gradient:
            c1, c2 = tpl.gradient_colors
            filters.append(f"drawbox=x=0:y=0:w=iw:h={int(target_h * 0.18)}:color={c1}@0.9:t=fill")
            filters.append(f"drawbox=x=0:y={int(target_h * 0.82)}:w=iw:h={int(target_h * 0.18)}:color={c2}@0.9:t=fill")
        else:
            if tpl.top_bar_opacity > 0:
                filters.append(
                    f"drawbox=x=0:y=0:w=iw:h={int(target_h * 0.18)}:color={tpl.top_bar_color}@{tpl.top_bar_opacity}:t=fill")
            if tpl.bottom_bar_opacity > 0:
                filters.append(
                    f"drawbox=x=0:y={int(target_h * 0.82)}:w=iw:h={int(target_h * 0.18)}:color={tpl.bottom_bar_color}@{tpl.bottom_bar_opacity}:t=fill")

        # Text - improved positioning and styling
        if tpl.text_style == "outline":
            border = f":borderw={tpl.border_width}:bordercolor=black"
        elif tpl.text_style == "glow":
            border = f":shadowx=0:shadowy=0:shadowcolor=black@0.8:box=1:boxcolor=black@0.2:boxborderw=5"
        else:  # shadow
            border = ":shadowx=2:shadowy=2:shadowcolor=black@0.8"

        # Fix the vertical positioning - use fixed pixel values instead of expressions
        top_y_pos = int(target_h * 0.09)  # 9% of height
        bottom_y_pos = int(target_h * 0.91)  # 91% of height

        # Top text - centered with fixed vertical positioning
        filters.append(
            f"drawtext=text='{top_txt_escaped}'{font_arg}:fontcolor={tpl.text_color}:fontsize={fs_base}"
            f"{border}:x=(w-text_w)/2:y={top_y_pos}:line_spacing=5")

        # Bottom text - centered with fixed vertical positioning
        filters.append(
            f"drawtext=text='{bot_txt_escaped}'{font_arg}:fontcolor={tpl.text_color}:fontsize={int(fs_base * 0.9)}"
            f"{border}:x=(w-text_w)/2:y={bottom_y_pos}-text_h:line_spacing=5")

    return ",".join(filters)


# ==========================================
# 6. Main Logic
# ==========================================

def run_ffmpeg(args):
    cmd, log = args
    with open(log, "w") as f: subprocess.run(cmd, stdout=f, stderr=f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--workdir", default="work")
    parser.add_argument("--transcript", help="SRT path")
    parser.add_argument("--transcript-output", help="Where to save the transcript")
    parser.add_argument("--output-prefix", help="Prefix for output filenames")

    # Whisper Settings
    parser.add_argument("--whisper-model", default="small", choices=["small", "medium", "large"])
    parser.add_argument("--language", default="auto", choices=["auto", "english", "japanese"])

    # Clip Settings
    parser.add_argument("--num-clips", type=int, default=5)
    parser.add_argument("--skip-intro", type=float, default=0)
    parser.add_argument("--skip-outro", type=float, default=0)

    # Export Settings
    parser.add_argument("--target-res", default="1080x1920")
    parser.add_argument("--aspect-mode", default="fit")
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--encoder", default="auto")

    # Overlay
    parser.add_argument("--template-style", default="viral_shorts")
    parser.add_argument("--channel-name", default="")
    parser.add_argument("--overlay-font", default="Arial")
    parser.add_argument("--overlay-top-text", default="{hook}")
    parser.add_argument("--overlay-bottom-text", default="{punchline}")

    args = parser.parse_args()

    try:
        global FFMPEG
        FFMPEG = resolve_exe("ffmpeg")
    except:
        print("Error: FFmpeg not found.")
        sys.exit(1)

    os.makedirs(args.outdir, exist_ok=True)
    os.makedirs(args.workdir, exist_ok=True)

    # Get video info for naming
    video_basename = os.path.basename(args.video)
    video_name = os.path.splitext(video_basename)[0]
    video_hash = get_video_hash(args.video)

    # Set output prefix if not provided
    if not args.output_prefix:
        args.output_prefix = sanitize_filename(video_name)

    # Check for anime content and set appropriate template and font
    if "anime" in video_name.lower() or "naruto" in video_name.lower() or "manga" in video_name.lower():
        print(f"Anime content detected, using anime template")
        args.template_style = "anime"
        args.overlay_font = "Noto Sans CJK JP"  # Better for Japanese text

    # Auto-detect language from audio if set to auto
    if args.language == "auto":
        print("Analyzing audio to detect language...")
        detected_language = detect_audio_language(args.video)
        if detected_language != "auto":
            args.language = detected_language
            print(f"Auto-detected language from audio: {args.language}")

    # 1. Handle Transcript / Whisper
    srt_file = args.transcript

    if not srt_file or not os.path.exists(srt_file):
        # Check if transcript output path is specified
        transcript_output = args.transcript_output

        # Check if SRT already exists next to video
        potential_srt = os.path.splitext(args.video)[0] + ".srt"
        if os.path.exists(potential_srt):
            print(f"Found existing transcript: {potential_srt}")
            srt_file = potential_srt
        else:
            # Check if there's a transcript in the workdir with the video hash
            work_transcript = os.path.join(args.workdir, f"{video_hash}.srt")
            if os.path.exists(work_transcript):
                print(f"Found existing transcript in workdir: {work_transcript}")
                srt_file = work_transcript
            else:
                # Need to transcribe
                print(f"No transcript found. Initializing Whisper ({args.whisper_model})...")

                # Save transcript in the output directory
                transcript_output = os.path.join(args.outdir, f"{args.output_prefix}_transcript.srt")

                # Transcribe the video
                srt_file = transcribe_video(video_path=args.video,
                                            model_size=args.whisper_model,
                                            language=args.language,
                                            transcript_output=transcript_output)

                # No need to make a duplicate copy
                print(f"Transcript saved to output directory")

    # 2. Parse & Select Clips
    candidates = parse_srt(srt_file)
    if not candidates:
        print("Transcript empty or invalid. Falling back to time-segmentation.")
        # Fallback logic - use video duration if available
        duration = get_video_duration(args.video)
        if duration <= 0:
            duration = 1400  # Default to ~23 minutes for anime episodes

        # Create segments of 60 seconds each
        segment_duration = 60
        candidates = [
            Candidate(i * segment_duration, (i + 1) * segment_duration, f"Clip {i + 1}")
            for i in range(min(args.num_clips, int(duration // segment_duration)))
        ]

    # Skip intro if specified
    if args.skip_intro > 0:
        print(f"Skipping first {args.skip_intro} seconds (intro)")
        candidates = [c for c in candidates if c.start >= args.skip_intro]

    # Simple Selection (In real usage, you'd score them here)
    # We filter out very short segments or merge them
    # For now, we group segments into 60s chunks roughly
    final_clips = []
    current_chunk = []
    current_dur = 0

    for cand in candidates:
        dur = cand.end - cand.start
        if current_dur + dur < 60:
            current_chunk.append(cand)
            current_dur += dur
        else:
            # Finalize chunk
            if current_chunk:
                start = current_chunk[0].start
                end = current_chunk[-1].end
                text = " ".join([c.text for c in current_chunk])
                final_clips.append(Candidate(start, end, text))
            current_chunk = [cand]
            current_dur = dur

    # Add the last chunk if not empty
    if current_chunk:
        start = current_chunk[0].start
        end = current_chunk[-1].end
        text = " ".join([c.text for c in current_chunk])
        final_clips.append(Candidate(start, end, text))

    selected = final_clips[:args.num_clips]
    print(f"Selected {len(selected)} clips for export.")

    # 3. Export
    tasks = []
    try:
        w, h = args.target_res.lower().split("x")
        res = (int(w), int(h))
    except:
        res = (1080, 1920)

    for i, clip in enumerate(selected):
        # Create descriptive filename
        clip_start_time = format_timestamp(clip.start).replace(':', '_').replace(',', '_')
        out_name = f"{args.output_prefix}_clip_{i + 1:02d}_{clip_start_time}.mp4"
        out_path = os.path.join(args.outdir, out_name)
        log_path = os.path.join(args.workdir, f"log_{args.output_prefix}_{i + 1}.txt")

        vf = build_filter_chain(clip, i + 1, args, res)

        # Hardware Accel Check - Optimize for RTX 3060
        v_codec = "libx264"
        enc_opts = ["-preset", "fast", "-crf", "23"]

        # For RTX 3060, use NVENC with optimized settings
        if args.encoder == "nvenc" or (args.encoder == "auto" and shutil.which("nvidia-smi")):
            v_codec = "h264_nvenc"
            # Use these optimized settings for RTX 3060
            enc_opts = ["-preset", "p2", "-tune", "hq", "-cq", "23", "-b:v", "0"]

        cmd = [
            FFMPEG, "-y", "-hide_banner", "-loglevel", "error",
            "-ss", str(clip.start), "-t", str(clip.end - clip.start),
            "-i", args.video, "-vf", vf,
            "-c:v", v_codec, *enc_opts,
            "-c:a", "aac", "-b:a", "192k",
            out_path
        ]
        tasks.append((cmd, log_path))

    # Parallel Export - Optimize for RTX 3060
    # For video encoding, 2-3 parallel jobs is optimal for RTX 3060
    max_w = min(args.jobs, 3, os.cpu_count() or 2)
    print(f"Exporting with {max_w} workers (optimized for RTX 3060)...")

    with ThreadPoolExecutor(max_workers=max_w) as exc:
        futs = [exc.submit(run_ffmpeg, t) for t in tasks]
        for i, f in enumerate(as_completed(futs)):
            try:
                f.result()
                print(f"Clip {i + 1}/{len(tasks)} exported: {os.path.basename(tasks[i][0][-1])}")
            except Exception as e:
                print(f"Export failed for clip {i + 1}: {e}")

    # Create a summary file with clip information
    summary_path = os.path.join(args.outdir, f"{args.output_prefix}_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"Video: {video_basename}\n")
        f.write(f"Processed: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Transcript: {os.path.basename(srt_file)}\n")
        f.write(f"Model: {args.whisper_model}\n")
        f.write(f"Language: {args.language}\n\n")
        f.write("Clips:\n")

        for i, clip in enumerate(selected):
            hook, punchline = get_hook_punchline(clip.text)
            f.write(f"Clip {i + 1}:\n")
            f.write(f"  Time: {format_timestamp(clip.start)} - {format_timestamp(clip.end)}\n")
            f.write(f"  Hook: {hook}\n")
            f.write(f"  Punchline: {punchline}\n")
            f.write(f"  Full text: {clip.text[:100]}{'...' if len(clip.text) > 100 else ''}\n\n")

    print(f"Processing complete. Output saved to: {args.outdir}")
    print(f"Summary file created: {summary_path}")

    # Clean up any temporary text files that might have been created
    for f in os.listdir(args.workdir):
        if f.startswith("t_") and f.endswith(".txt") or f.startswith("b_") and f.endswith(".txt"):
            try:
                os.remove(os.path.join(args.workdir, f))
            except:
                pass


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding='utf-8')
    main()