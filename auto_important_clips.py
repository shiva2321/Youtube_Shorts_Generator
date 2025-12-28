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
    "simple": OverlayTemplate(
        name="simple",
        top_bar_opacity=0.8,
        bottom_bar_opacity=0.8,
        text_style="shadow",
        emoji_prefix="",
        emoji_suffix=""
    ),
    "viral_shorts": OverlayTemplate(
        name="viral_shorts",
        header_text="LIKE & SUBSCRIBE",
        show_branding=True,
        show_subscribe=True,
        use_gradient=True,
        gradient_colors=["#0066ff", "#0044aa"],
        text_style="outline",
        border_width=4,
        emoji_prefix="🔥 ",
        emoji_suffix=" 👀"
    ),
    "neon_vibes": OverlayTemplate(
        name="neon_vibes",
        header_text="DON'T MISS THIS",
        use_gradient=True,
        gradient_colors=["#ff00cc", "#333399"],
        text_style="glow",
        show_branding=True,
        show_subscribe=True,
        emoji_prefix="✨ ",
        emoji_suffix=" ✨"
    ),
    "glass_modern": OverlayTemplate(
        name="glass_modern",
        top_bar_color="black",
        bottom_bar_color="black",
        top_bar_opacity=0.4,
        bottom_bar_opacity=0.4,
        text_style="shadow",
        border_width=2,
        show_branding=True,
        show_subscribe=True,
        emoji_prefix="👉 ",
        emoji_suffix=" 👈"
    ),
    "cinematic": OverlayTemplate(
        name="cinematic",
        top_bar_color="black",
        bottom_bar_color="black",
        top_bar_opacity=1.0,
        bottom_bar_opacity=1.0,
        text_style="shadow",
        text_color="#f0f0f0",
        show_branding=True,
        show_subscribe=True,
        emoji_prefix="",
        emoji_suffix=""
    ),
    "anime": OverlayTemplate(
        name="anime",
        top_bar_color="#0066cc",
        bottom_bar_color="#0066cc",
        top_bar_opacity=0.85,
        bottom_bar_opacity=0.85,
        text_style="outline",
        text_color="#ffffff",
        border_width=4,
        show_branding=True,
        show_subscribe=True,
        emoji_prefix="⭐ ",
        emoji_suffix=" ⭐"
    ),
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
        duration = get_video_duration(video_path)

        # Define sample points (skip intro, take from multiple points)
        sample_points = []
        if duration < 300:  # Short video under 5 minutes
            sample_points = [(duration * 0.5, min(30, duration * 0.5))]
        else:
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
            import torch
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
            f"[Whisper] Memory requirements ({required_memory:.1f}GB) exceed available GPU memory, using optimized chunking"
        )
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
        if language == "japanese":
            lang_arg = "ja"
        if language == "english":
            lang_arg = "en"

        print(f"[Whisper] Transcribing with language setting: {language}")

        if chunk_size <= 0 or duration <= chunk_size:
            # Process entire audio in one go
            start_time = time.time()

            # Use more aggressive compression for long files to fit in memory
            if duration > 1800 and device == "cuda":  # > 30 minutes
                print("[Whisper] Long audio detected, using memory-efficient processing")
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
            print("[Whisper] Processing in optimized chunks with overlap for seamless transitions")

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
                    f"[Whisper] Processing chunk {chunk_start / 60:.1f}-{(chunk_start + chunk_duration) / 60:.1f} minutes..."
                )

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
                    segment["start"] += chunk_start
                    segment["end"] += chunk_start

                    # Include segments excluding overlapping regions
                    if chunk_start == 0 or segment["start"] >= chunk_start:
                        if chunk_start == 0 or segment["start"] >= chunk_start + overlap:
                            all_segments.append(segment)

                # Clean up chunk audio
                os.remove(chunk_audio)

                print(f"[Whisper] Chunk processed in {elapsed:.2f} seconds")

                # Free up GPU memory
                if device == "cuda":
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
            import torch
            torch.cuda.empty_cache()

    print(f"[Whisper] Saved transcript to {srt_path}")
    return srt_path


def parse_srt(srt_path: str) -> List[Candidate]:
    """Parses an SRT file into Candidate objects."""
    candidates: List[Candidate] = []
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
            if len(times) != 2:
                continue

            def time_to_sec(t_str: str) -> float:
                h, m, s = t_str.replace(',', '.').split(':')
                return int(h) * 3600 + int(m) * 60 + float(s)

            try:
                start = time_to_sec(times[0])
                end = time_to_sec(times[1])
                text = " ".join(lines[2:])
                candidates.append(Candidate(start, end, text))
            except Exception:
                continue
    return candidates


# ==========================================
# 3. System & Tool Utilities
# ==========================================

def get_system_font_path(font_name: str) -> str:
    """Smart cross-platform font finder."""
    if os.path.exists(font_name):
        return os.path.abspath(font_name)
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
    if path:
        return path
    if platform.system() == "Windows":
        common = [r"C:\Program Files\ffmpeg\bin\ffmpeg.exe", r"C:\ffmpeg\bin\ffmpeg.exe"]
        for p in common:
            if os.path.exists(p):
                return p
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

    if not sentences:
        return text[:50]

    if mode == "hook":
        # Strategy: Questions or Exclamations first
        for s in sentences[:4]:
            if any(p in s for p in ["?", "？"]) and len(s) < 60:
                return s
        for s in sentences[:4]:
            if any(p in s for p in ["!", "！"]) and len(s) < 50:
                return s
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
# 4b. Clip Scoring & Merging for Better Selection
# ==========================================

def compute_heuristic_score(cand: Candidate, video_duration: Optional[float] = None) -> float:
    """
    Simple heuristic scoring:
    - Boost questions/exclamations
    - Penalize very short or very long segments
    - Prefer segments away from intros/outros
    """
    text = cand.text.strip()
    length = len(text)
    dur = max(0.1, cand.end - cand.start)

    score = 0.0

    # Basic content features
    if "?" in text or "？" in text:
        score += 2.0
    if "!" in text or "！" in text:
        score += 1.5

    # Numbers / "listicle" style
    if re.search(r"\b\d+\b", text):
        score += 0.8

    # Some "interest" keywords (customizable)
    keywords = [
        "secret", "crazy", "insane", "best", "worst", "mistake", "tips",
        "how to", "you won't", "you will", "the reason", "truth", "behind",
        "story", "learn", "amazing", "incredible", "unbelievable",
    ]
    lowered = text.lower()
    for kw in keywords:
        if kw in lowered:
            score += 1.0

    # Duration: prefer ~8–25s segments
    if dur < 5:
        score -= 1.0
    elif 5 <= dur <= 25:
        score += 1.5
    elif dur > 45:
        score -= 0.8

    # Penalize extremely short text
    if length < 15:
        score -= 0.8

    # Position in video
    if video_duration:
        mid = (cand.start + cand.end) / 2
        rel = mid / video_duration
        # lightly penalize first/last 5% unless very strong
        if rel < 0.05 or rel > 0.95:
            score -= 0.5
        # central 20–80% gets small boost
        if 0.2 <= rel <= 0.8:
            score += 0.5

    return score


def merge_adjacent_candidates(
        candidates: List[Candidate],
        max_clip_len: float = 60.0,
        min_clip_len: float = 8.0
) -> List[Candidate]:
    """
    Merge subtitle segments into more natural clip-sized chunks.
    Tries to keep each clip between min_clip_len and max_clip_len seconds.
    """
    merged: List[Candidate] = []
    if not candidates:
        return merged

    current: List[Candidate] = []
    curr_start = candidates[0].start
    curr_end = candidates[0].end
    text_parts: List[str] = []

    for cand in candidates:
        cand_dur = cand.end - cand.start

        # If no current chunk, start one
        if not current:
            current.append(cand)
            curr_start, curr_end = cand.start, cand.end
            text_parts = [cand.text]
            continue

        new_end = cand.end
        tentative_dur = new_end - curr_start

        if tentative_dur <= max_clip_len:
            # merge into current
            current.append(cand)
            curr_end = new_end
            text_parts.append(cand.text)
        else:
            # finalize current
            combined_text = " ".join(text_parts).strip()
            dur = max(0.1, curr_end - curr_start)
            if dur >= min_clip_len:
                merged.append(Candidate(curr_start, curr_end, combined_text))
            # start new chunk
            current = [cand]
            curr_start, curr_end = cand.start, cand.end
            text_parts = [cand.text]

    # finalize last chunk
    if current:
        combined_text = " ".join(text_parts).strip()
        dur = max(0.1, curr_end - curr_start)
        if dur >= min_clip_len:
            merged.append(Candidate(curr_start, curr_end, combined_text))

    return merged


def llm_score_candidate(
        cand: Candidate,
        clip_index: int,
        video_context: Optional[str] = None
) -> float:
    """
    OPTIONAL: Hook for scoring clips using an external LLM (local or remote).
    By default, returns 0.0 (no contribution).

    You can implement your own logic here, for example:
    - call a local server (Ollama, LM Studio, etc.)
    - call OpenAI / another API
    - cache scores based on (start, end, text)

    Return a score in range roughly [-3, +3] to combine with heuristic.
    """
    # Example pseudo-implementation (commented out):
    # prompt = f"... use cand.text and maybe video_context ..."
    # score = call_my_llm(prompt)
    # return float(score)
    return 0.0


# ==========================================
# 5. FFmpeg Filter Construction
# ==========================================

def build_filter_chain(clip: Candidate, idx: int, cfg: argparse.Namespace, resolution: Tuple[int, int]) -> str:
    target_w, target_h = resolution
    filters: List[str] = []

    # Enhanced Scale/Crop with smarter aspect ratio handling
    if cfg.aspect_mode == "fill":
        # Fill mode: cover the entire frame but may crop edges
        filters.append(
            f"scale={target_w}:{target_h}:force_original_aspect_ratio=increase,"
            f"crop={target_w}:{target_h}"
        )
    elif cfg.aspect_mode == "fit":
        # Fit mode: ensure the entire original video is visible with letterboxing/pillarboxing
        # Use a slight blur effect on the background to make it look more professional
        filters.append(
            f"split[original][bg];"
            f"[bg]scale={target_w}:{target_h}:force_original_aspect_ratio=increase,crop={target_w}:{target_h},boxblur=10[blurbg];"
            f"[original]scale={target_w}:{target_h}:force_original_aspect_ratio=decrease[scaled];"
            f"[blurbg][scaled]overlay=(W-w)/2:(H-h)/2"
        )
    else:  # fallback to basic fit
        filters.append(
            f"scale={target_w}:{target_h}:force_original_aspect_ratio=decrease,"
            f"pad={target_w}:{target_h}:(ow-iw)/2:(oh-ih)/2:color=black"
        )

    # 2. Overlay
    if cfg.template_style:
        tpl = OVERLAY_TEMPLATES.get(cfg.template_style, OVERLAY_TEMPLATES["simple"])
        hook, punchline = get_hook_punchline(clip.text)

        # Decide what goes on top and bottom
        try:
            clip_duration = clip.end - clip.start
            is_intro_clip = (idx == 1 and clip.start < 90)  # first 90 s of video
            is_outro_clip = (clip_duration < 30 and len(clip.text) < 120)

            # Choose primary text based on clip position/nature
            if is_intro_clip:
                primary_text = hook
            elif is_outro_clip:
                primary_text = punchline
            else:
                # Use longer of hook/punchline that still fits nicely
                primary_text = hook if len(hook) >= len(punchline) else punchline

            # Top text
            top_txt = cfg.overlay_top_text.format(
                hook=hook,
                punchline=punchline,
                primary=primary_text,
                channel=cfg.channel_name,
                i=idx,
            )

            # Bottom text: prefer subscribe CTA if we have channel
            if cfg.channel_name and tpl.show_subscribe:
                bottom_default = f"Subscribe {cfg.channel_name}"
            elif tpl.show_branding and cfg.channel_name:
                bottom_default = cfg.channel_name
            else:
                bottom_default = punchline or hook

            bot_txt = cfg.overlay_bottom_text.format(
                hook=hook,
                punchline=punchline,
                primary=primary_text,
                channel=cfg.channel_name,
                i=idx,
            )

            # If user left default placeholders, enforce our bottom_default
            if cfg.overlay_bottom_text in ("{punchline}", "{hook}", "{primary}"):
                bot_txt = bottom_default

            # Add emojis only once to top text
            if tpl.emoji_prefix and not top_txt.startswith(tpl.emoji_prefix):
                top_txt = f"{tpl.emoji_prefix}{top_txt}"
            if tpl.emoji_suffix and not top_txt.endswith(tpl.emoji_suffix):
                top_txt = f"{top_txt}{tpl.emoji_suffix}"
        except Exception:
            top_txt = hook
            bot_txt = punchline or cfg.channel_name or ""

        # Soft limit on text length to avoid overflowing the screen
        def shorten_for_overlay(t: str, max_chars: int = 80) -> str:
            t = t.strip()
            # Additional check to further limit text length
            words = t.split()
            if len(words) > 10:  # Too many words
                t = ' '.join(words[:10])
                # Don't add "..." if it ends with an emoji
                if t and not any(unicodedata.category(c) == 'So' for c in t[-2:]):
                    t += "..."
            elif len(t) > max_chars:
                # Truncate at word boundary
                truncated = t[:max_chars - 3]
                last_space = truncated.rfind(' ')
                if last_space > max_chars * 0.7:
                    t = t[:last_space] + "..."
                else:
                    t = truncated + "..."
            return t

        # Use stricter length limits
        top_txt = shorten_for_overlay(top_txt, 80)
        bot_txt = shorten_for_overlay(bot_txt, 60)

        # Calculate font size BEFORE creating overlay assets
        is_vertical = target_h > target_w
        fs_base = int(target_h * (0.04 if is_vertical else 0.05))

        # Bars
        bar_height_pct = 0.2
        if tpl.use_gradient:
            c1, c2 = tpl.gradient_colors
            filters.append(
                f"drawbox=x=0:y=0:w=iw:h={int(target_h * bar_height_pct)}:color={c1}@0.9:t=fill"
            )
            filters.append(
                f"drawbox=x=0:y={int(target_h * (1 - bar_height_pct))}:w=iw:h={int(target_h * bar_height_pct)}:color={c2}@0.9:t=fill"
            )
        else:
            if tpl.top_bar_opacity > 0:
                filters.append(
                    f"drawbox=x=0:y=0:w=iw:h={int(target_h * bar_height_pct)}:color={tpl.top_bar_color}@{tpl.top_bar_opacity}:t=fill"
                )
            if tpl.bottom_bar_opacity > 0:
                filters.append(
                    f"drawbox=x=0:y={int(target_h * (1 - bar_height_pct))}:w=iw:h={int(target_h * bar_height_pct)}:color={tpl.bottom_bar_color}@{tpl.bottom_bar_opacity}:t=fill"
                )

        # --- Overlay backend selection ---
        # Prefer ASS (libass) when available in ffmpeg build; fallback to PNG overlay (Pillow)
        overlay_backend = getattr(cfg, "overlay_backend", "ass")

        if overlay_backend == "ass":
            ass_file = os.path.join(cfg.workdir, f"overlay_{idx}.ass")
            top_y_pos = int(target_h * bar_height_pct / 2)
            bottom_y_pos = int(target_h * (1 - bar_height_pct / 2))

            with open(ass_file, 'w', encoding='utf-8') as f:
                f.write('[Script Info]\n')
                f.write('ScriptType: v4.00+\n')
                f.write(f'PlayResX: {target_w}\n')
                f.write(f'PlayResY: {target_h}\n')
                f.write('WrapStyle: 2\n')
                f.write('ScaledBorderAndShadow: yes\n\n')

                f.write('[V4+ Styles]\n')
                f.write('Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n')
                # Alignment 8 = top center; 2 = bottom center
                f.write(f'Style: Top,Arial,{fs_base},&H00FFFFFF,&H000000FF,&H00000000,&H00000000,-1,0,0,0,100,100,0,0,1,4,2,8,30,30,{top_y_pos},1\n')
                f.write(f'Style: Bottom,Arial,{int(fs_base * 0.9)},&H00FFFFFF,&H000000FF,&H00000000,&H00000000,-1,0,0,0,100,100,0,0,1,4,2,2,30,30,{target_h - bottom_y_pos},1\n\n')

                f.write('[Events]\n')
                f.write('Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n')
                f.write(f'Dialogue: 0,0:00:00.00,9:59:59.99,Top,,0,0,0,,{top_txt}\n')
                f.write(f'Dialogue: 0,0:00:00.00,9:59:59.99,Bottom,,0,0,0,,{bot_txt}\n')

            # FFmpeg ass filter on Windows requires escaping drive colon (D:) as D\:
            ass_path = os.path.abspath(ass_file).replace('\\', '/')
            if len(ass_path) >= 2 and ass_path[1] == ':':
                ass_path = ass_path[0] + r"\:" + ass_path[2:]
            # Quote to preserve spaces
            filters.append(f"ass=filename='{ass_path}'")

        elif overlay_backend == "png":
            # Defer actual PNG overlay generation to export stage (needs extra input)
            # Marker placeholder; export stage will replace vf+inputs.
            filters.append("__PNG_OVERLAY__")

    return ",".join(filters)


# ==========================================
# 6. Main Logic
# ==========================================

def run_ffmpeg(args):
    cmd, log = args
    try:
        with open(log, "w", encoding='utf-8') as f:
            result = subprocess.run(cmd, stdout=f, stderr=f, check=False)

        # Check if FFmpeg actually succeeded
        if result.returncode != 0:
            with open(log, "r", encoding='utf-8') as f:
                error_content = f.read()
            raise RuntimeError(f"FFmpeg failed with return code {result.returncode}. Check log for details")

        # Verify the output file was actually created
        output_file = cmd[-1]
        if not os.path.exists(output_file):
            with open(log, "r", encoding='utf-8') as f:
                error_content = f.read()
            raise RuntimeError(f"FFmpeg completed but output file not created: {output_file}")

        return True
    except Exception as e:
        raise RuntimeError(f"FFmpeg error: {e}")


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
    parser.add_argument(
        "--target-length",
        type=int,
        default=180,
        help="Target clip length (PER CLIP) in seconds",
    )

    # Export Settings
    parser.add_argument("--target-res", default="1080x1920")
    parser.add_argument("--aspect-mode", default="fit", choices=["fit", "fill"])
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--encoder", default="auto")

    # Overlay
    parser.add_argument("--template-style", default="viral_shorts")
    parser.add_argument("--channel-name", default="")
    parser.add_argument("--overlay-font", default="Arial")
    parser.add_argument("--overlay-top-text", default="{hook}")
    parser.add_argument("--overlay-bottom-text", default="{punchline}")
    parser.add_argument("--overlay-backend", default="ass", choices=["ass", "png"], help="Overlay backend: 'ass' (libass) or 'png' (Pillow rendered).")

    args = parser.parse_args()

    try:
        global FFMPEG
        FFMPEG = resolve_exe("ffmpeg")
    except Exception:
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
        print("Anime content detected, using anime template")
        args.template_style = "anime"
        args.overlay_font = "Noto Sans CJK JP"  # Better for Japanese text

    # Auto-detect language from audio if set to auto
    if args.language == "auto":
        print("Analyzing audio to detect language...")
        detected_language = detect_audio_language(args.video)
        if detected_language != "auto":
            args.language = detected_language
            print(f"Auto-detected language from audio: {args.language}")

    # Validate and adjust target length (PER CLIP)
    if args.target_length:
        args.target_length = min(300, max(10, int(args.target_length)))
        print(f"Target clip length (per clip): {args.target_length} seconds")

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
                srt_file = transcribe_video(
                    video_path=args.video,
                    model_size=args.whisper_model,
                    language=args.language,
                    transcript_output=transcript_output,
                )

                print("Transcript saved to output directory")

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
            for i in range(min(args.num_clips * 3, int(duration // segment_duration)))
        ]

    # Skip intro if specified
    if args.skip_intro > 0:
        print(f"Skipping first {args.skip_intro} seconds (intro)")
        candidates = [c for c in candidates if c.start >= args.skip_intro]

    # Improved Selection: merge into clip-sized chunks, then score
    print("Building candidate clips from subtitles...")
    duration = get_video_duration(args.video)

    # Use target length to determine optimal clip length
    target_clip_duration = 20  # Default clip length (seconds)
    if args.target_length:
        # target_length is the desired duration for EACH clip, not total
        target_clip_duration = args.target_length
        # Constrain to reasonable bounds (8 seconds to 5 minutes)
        target_clip_duration = min(300.0, max(8.0, target_clip_duration))

    print(f"Target clip duration: ~{target_clip_duration:.1f} seconds per clip")
    if args.num_clips > 0:
        total_duration_estimate = target_clip_duration * args.num_clips
        print(
            f"Will generate {args.num_clips} clips × {target_clip_duration:.0f}s = ~{total_duration_estimate:.0f}s total")

    merged_candidates = merge_adjacent_candidates(
        candidates,
        max_clip_len=target_clip_duration * 1.5,  # Allow some flexibility
        min_clip_len=target_clip_duration * 0.5,  # Allow some flexibility
    )

    if not merged_candidates:
        print("No suitable merged segments found from transcript; using raw segments.")
        merged_candidates = candidates

    # Ensure we have enough candidates (at least 3x requested clips)
    if len(merged_candidates) < args.num_clips * 3:
        print(f"Warning: Only found {len(merged_candidates)} candidate clips. Creating additional candidates...")
        # Create more candidates by splitting existing ones or using time segments
        segment_duration = int(target_clip_duration)
        for i in range(args.num_clips * 3 - len(merged_candidates)):
            start_time = i * segment_duration
            if start_time + segment_duration < duration:
                merged_candidates.append(
                    Candidate(start_time, start_time + segment_duration, f"Additional clip {i + 1}")
                )

    # Compute scores
    print(f"Scoring {len(merged_candidates)} candidate clips...")
    for idx, cand in enumerate(merged_candidates, start=1):
        h_score = compute_heuristic_score(cand, video_duration=duration)
        l_score = llm_score_candidate(cand, clip_index=idx)
        cand.score = h_score + l_score

        # Boost scores for clips closer to target duration
        clip_duration = cand.end - cand.start
        duration_match = 1.0 - min(1.0, abs(clip_duration - target_clip_duration) / target_clip_duration)
        cand.score += duration_match * 1.5

    # Sort by score descending, avoid heavy overlaps
    merged_candidates.sort(key=lambda c: c.score, reverse=True)

    # Fix: Ensure we select exactly the requested number of clips
    num_clips_to_select = args.num_clips
    print(f"Attempting to select exactly {num_clips_to_select} clips...")

    selected: List[Candidate] = []
    used_intervals: List[Tuple[float, float]] = []

    def overlaps(a_start, a_end, b_start, b_end, max_overlap_ratio=0.5) -> bool:
        inter_start = max(a_start, b_start)
        inter_end = min(a_end, b_end)
        inter = max(0.0, inter_end - inter_start)
        len_a = a_end - a_start
        len_b = b_end - b_start
        return (inter / max(len_a, 0.1) > max_overlap_ratio) or \
            (inter / max(len_b, 0.1) > max_overlap_ratio)

    # First pass - select clips while avoiding overlaps
    for cand in merged_candidates:
        if len(selected) >= num_clips_to_select:
            break
        too_close = False
        for (us, ue) in used_intervals:
            if overlaps(cand.start, cand.end, us, ue):
                too_close = True
                break
        if not too_close:
            selected.append(cand)
            used_intervals.append((cand.start, cand.end))

    # If we still need more clips, relax the overlap constraints
    if len(selected) < num_clips_to_select:
        print(f"First pass only selected {len(selected)} clips. Relaxing overlap constraints...")
        for cand in merged_candidates:
            if cand in selected:
                continue
            selected.append(cand)
            used_intervals.append((cand.start, cand.end))
            if len(selected) >= num_clips_to_select:
                break

    # Final fallback - if still not enough clips, duplicate some existing clips from different parts
    if len(selected) < num_clips_to_select:
        print(f"Not enough unique clips. Creating additional clips...")
        # If we have some selected clips, duplicate them with slight offsets
        if selected:
            clip_count = len(selected)
            for i in range(num_clips_to_select - clip_count):
                idx = i % clip_count
                base_clip = selected[idx]
                offset = 3.0  # Offset by 3 seconds
                new_start = max(0, base_clip.start - offset)
                new_end = min(duration, base_clip.end + offset)
                selected.append(Candidate(new_start, new_end, f"{base_clip.text} (variant)"))
        else:
            # Last resort - create time-based clips
            segment_duration = min(30, duration / num_clips_to_select)
            for i in range(num_clips_to_select):
                start_time = i * segment_duration
                end_time = min(duration, start_time + segment_duration)
                selected.append(Candidate(start_time, end_time, f"Segment {i + 1}"))

    # Limit to exactly the requested number of clips
    selected = selected[:num_clips_to_select]

    # Sort by timestamp for sequential output
    selected.sort(key=lambda c: c.start)

    print(f"Selected exactly {len(selected)} clips for export.")

    # 3. Export
    tasks = []
    try:
        w, h = args.target_res.lower().split("x")
        res = (int(w), int(h))
    except Exception:
        res = (1080, 1920)

    for i, clip in enumerate(selected):
        try:
            # Create descriptive filename
            clip_start_time = format_timestamp(clip.start).replace(':', '_').replace(',', '_')
            out_name = f"{args.output_prefix}_clip_{i + 1:02d}_{clip_start_time}.mp4"
            out_path = os.path.join(args.outdir, out_name)
            log_path = os.path.join(args.workdir, f"log_{args.output_prefix}_{i + 1}.txt")

            print(f"Building filter chain for clip {i + 1}/{len(selected)}...")
            vf = build_filter_chain(clip, i + 1, args, res)

            # Debug: print filter chain
            print(f"  Filter: {vf[:100]}..." if len(vf) > 100 else f"  Filter: {vf}")

            # Hardware Accel Check - Optimize for RTX 3060
            v_codec = "libx264"
            enc_opts = ["-preset", "fast", "-crf", "23"]

            # For RTX 3060, use NVENC with optimized settings
            if args.encoder == "nvenc" or (args.encoder == "auto" and shutil.which("nvidia-smi")):
                v_codec = "h264_nvenc"
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
        except Exception as e:
            print(f"ERROR: Failed to build task for clip {i + 1}: {e}")
            import traceback
            traceback.print_exc()

    # Parallel Export - Optimize for RTX 3060
    if not tasks:
        print("ERROR: No tasks were created. Check errors above.")
        sys.exit(1)

    print(f"Successfully created {len(tasks)} export tasks")
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
                log_file = tasks[i][1]
                if os.path.exists(log_file):
                    try:
                        with open(log_file, 'r', encoding='utf-8', errors='ignore') as lf:
                            error_log = lf.read()
                            if error_log:
                                print(f"  FFmpeg error output:\n{error_log[:500]}")
                    except Exception:
                        pass

    # Calculate total output duration
    total_duration = sum(c.end - c.start for c in selected)

    # Create a summary file with clip information
    summary_path = os.path.join(args.outdir, f"{args.output_prefix}_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"Video: {video_basename}\n")
        f.write(f"Processed: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Transcript: {os.path.basename(srt_file)}\n")
        f.write(f"Model: {args.whisper_model}\n")
        f.write(f"Language: {args.language}\n")
        f.write(f"Resolution: {args.target_res}\n")
        f.write(f"Aspect Mode: {args.aspect_mode}\n")
        f.write(f"Requested Clips: {num_clips_to_select}\n")
        f.write(f"Generated Clips: {len(selected)}\n")
        f.write(f"Clip Length (per clip): {args.target_length}s\n")
        f.write(f"Total Duration: {total_duration:.1f} seconds ({total_duration / 60:.1f} minutes)\n\n")
        f.write("Clips:\n")

        for i, clip in enumerate(selected):
            clip_duration = clip.end - clip.start
            hook, punchline = get_hook_punchline(clip.text)
            f.write(f"Clip {i + 1}:\n")
            f.write(f"  Time: {format_timestamp(clip.start)} - {format_timestamp(clip.end)} ({clip_duration:.1f}s)\n")
            f.write(f"  Hook: {hook}\n")
            f.write(f"  Punchline: {punchline}\n")
            f.write(f"  Full text: {clip.text[:100]}{'...' if len(clip.text) > 100 else ''}\n\n")

    print(f"Processing complete. Output saved to: {args.outdir}")
    print(f"Total video duration: {total_duration:.1f} seconds ({total_duration / 60:.1f} minutes)")
    print(f"Summary file created: {summary_path}")

    # Clean up any temporary text files that might have been created
    for f_name in os.listdir(args.workdir):
        if (f_name.startswith("t_") and f_name.endswith(".txt")) or \
                (f_name.startswith("b_") and f_name.endswith(".txt")):
            try:
                os.remove(os.path.join(args.workdir, f_name))
            except Exception:
                pass

if __name__ == "__main__":
    # Ensure UTF-8 stdout
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        # Fallback for older Python versions
        pass
    main()