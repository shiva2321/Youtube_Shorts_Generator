import argparse
import json
import os
import re
import subprocess
import sys
import time
import textwrap
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


# Ensure console output is UTF-8 so emoji-sourced log lines no longer trigger Windows codepage errors.
try:
  sys.stdout.reconfigure(encoding="utf-8", errors="replace")
  sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except AttributeError:
  pass

# Make sure child-process output decoded by Python uses UTF-8 where possible.
os.environ.setdefault("PYTHONIOENCODING", "utf-8")


# ----------------------------
# Portable tool discovery (no hard-coded paths)
# ----------------------------
def _which(cmd: str) -> Optional[str]:
  from shutil import which

  return which(cmd)


def _env_path(name: str) -> Optional[str]:
  v = (os.environ.get(name) or "").strip().strip('"')
  return v or None


def resolve_exe(
    *,
    env_var: str,
    default_names: List[str],
    label: str,
    extra_candidates: Optional[List[str]] = None,
) -> str:
  """Resolve an executable path.

  Priority:
    1) explicit env var path
    2) PATH lookup (shutil.which)
    3) optional additional candidate paths
  """
  p = _env_path(env_var)
  if p:
    if os.path.exists(p):
      return p
    raise FileNotFoundError(f"{label} from ${env_var} not found: {p}")

  for nm in default_names:
    hit = _which(nm)
    if hit:
      return hit

  for cand in (extra_candidates or []):
    if cand and os.path.exists(cand):
      return cand

  raise FileNotFoundError(
    f"{label} not found. Set {env_var} to the full path, or install it and make it available on PATH."
  )


def resolve_file(*, env_var: str, default_path: Optional[str], label: str) -> str:
  p = _env_path(env_var) or (default_path.strip() if default_path else None)
  if not p:
    raise FileNotFoundError(f"{label} not configured. Set {env_var} to the full path.")
  if not os.path.exists(p):
    raise FileNotFoundError(f"{label} not found: {p}")
  return p


def _common_ffmpeg_candidates() -> List[str]:
  """A few common Windows installs (best effort; safe if missing)."""
  cands: List[str] = []
  lad = os.environ.get("LOCALAPPDATA")
  if lad:
    cands.append(os.path.join(lad, "Microsoft", "WinGet", "Packages"))
  # Donâ€™t walk directories here (too slow). Just keep this minimal.
  return []


# ----------------------------
# Tool paths (resolved at runtime)
# ----------------------------
# Env vars supported:
#  - ANIMECLIPS_FFMPEG
#  - ANIMECLIPS_FFPROBE
#  - ANIMECLIPS_WHISPER_CLI
#  - ANIMECLIPS_WHISPER_MODEL
#  - ANIMECLIPS_WHISPER_VAD_MODEL

# NOTE: We resolve these in main() so importing this module never fails.
FFMPEG = "ffmpeg"
FFPROBE = "ffprobe"

# whisper.cpp is optional if you use faster-whisper. Resolve lazily when used.
WHISPER_CLI = _env_path("ANIMECLIPS_WHISPER_CLI")
WHISPER_MODEL = _env_path("ANIMECLIPS_WHISPER_MODEL")
WHISPER_VAD_MODEL = _env_path("ANIMECLIPS_WHISPER_VAD_MODEL")


# ----------------------------
# Data structures
# ----------------------------
@dataclass
class Segment:
  start: float
  end: float
  text: str


@dataclass
class Candidate:
  start: float
  end: float
  text: str
  score: float = 0.0


@dataclass
class ASRConfig:
  backend: str = "auto"          # auto|whispercpp|faster-whisper
  model: str = "small"           # faster-whisper model name
  device: str = "auto"           # auto|cpu|cuda
  compute_type: str = "auto"     # auto|float16|int8|...
  threads: int = 6               # whisper.cpp threads


# ----------------------------
# Subprocess helpers (NO shell=True; Windows-safe)
# ----------------------------
def run_capture(args: List[str]) -> str:
  print("RUN:", " ".join(_pretty_args(args)))
  p = subprocess.run(args, check=True, capture_output=True, text=True, encoding="utf-8", errors="ignore")
  # Some ffmpeg/ffprobe info comes on stderr; callers pass args accordingly.
  return (p.stdout or "") + (p.stderr or "")


def run(args: List[str]) -> None:
  print("RUN:", " ".join(_pretty_args(args)))
  subprocess.run(args, check=True)


def _safe_display(s: str) -> str:
  """Best-effort safe string for console output on Windows codepages."""
  try:
    return s.encode("utf-8", "backslashreplace").decode("utf-8")
  except Exception:
    try:
      return s.encode("ascii", "backslashreplace").decode("ascii")
    except Exception:
      return str(s)


def _pretty_args(args: List[str]) -> List[str]:
  out: List[str] = []
  for a in args:
    a = "" if a is None else str(a)
    a = _safe_display(a)
    if not a:
      out.append('""')
    elif any(c.isspace() for c in a) or "'" in a or '"' in a:
      out.append('"' + a.replace('"', '\\"') + '"')
    else:
      out.append(a)
  return out


def _vf_escape_path(p: str) -> str:
  """Escape a filesystem path for ffmpeg filter args (drawtext textfile/fontfile).

  For filter arguments, ':' is a key/value delimiter, so Windows drive letters must be escaped.
  Also escape backslashes and single-quotes, and keep forward slashes.
  """
  p = os.path.abspath(p)
  p = p.replace("\\", "/")
  # escape for ffmpeg option parsing inside filter definitions
  p = p.replace("'", "\\'")
  p = p.replace(":", "\\:")
  return p


def _write_textfile_utf8(path: str, text: str) -> str:
  os.makedirs(os.path.dirname(path), exist_ok=True)
  with open(path, "w", encoding="utf-8", newline="\n") as f:
    f.write((text or "").strip() + "\n")
  return path


def _overlay_should_apply(*, aspect: str, mode: str, overlay_mode: str) -> bool:
  if overlay_mode == "off":
    return False
  if overlay_mode == "on":
    return True
  # auto -> only for padded outputs: aspect is 9:16 and using fit/smart (pad path)
  return aspect == "9:16" and mode in ("fit", "smart")


def _wrap_overlay_text(text: str, *, max_chars: int, max_lines: int) -> str:
  """Word-wrap overlay text into multiple lines.

  This keeps overlays readable on vertical video: we hard-wrap by
  character count so drawtext can render multi-line text from a
  textfile, since ffmpeg does not auto-wrap.
  """
  if not text:
    return ""
  text = re.sub(r"\s+", " ", text.strip())
  if not text:
    return ""
  # Basic greedy word wrap.
  wrapper = textwrap.TextWrapper(width=max_chars, break_long_words=True, break_on_hyphens=True)
  lines = wrapper.wrap(text)
  if not lines:
    return ""
  if len(lines) > max_lines:
    lines = lines[:max_lines]
    # Indicate truncation a bit.
    if not lines[-1].endswith("â€¦") and not lines[-1].endswith("..."):
      if len(lines[-1]) >= 3:
        lines[-1] = lines[-1][:-3] + "..."
      else:
        lines[-1] = lines[-1] + "..."
  return "\n".join(lines)


def _extract_sentences(text: str) -> List[str]:
  """Split text into sentences, preserving sentence structure."""
  if not text:
    return []
  # Basic sentence splitting on . ! ?
  sentences = re.split(r'(?<=[.!?])\s+', text.strip())
  return [s.strip() for s in sentences if s.strip()]


def _is_engaging_sentence(sentence: str) -> bool:
  """Heuristic to detect engaging/punchline-worthy sentences."""
  if not sentence or len(sentence) < 5:
    return False

  # Check for engagement markers
  engagement_words = [
    'wow', 'crazy', 'amazing', 'insane', 'incredible', 'unbelievable',
    'what', 'how', 'why', 'never', 'always', 'must', 'don\'t', 'can\'t',
    'wait', 'watch', 'look', 'see', 'hear', 'remember', 'think', 'know',
    'best', 'worst', 'better', 'worse', 'great', 'terrible', 'awesome',
    'epic', 'legendary', 'insane', 'shocked', 'surprised', 'shocked',
    'laugh', 'joke', 'funny', 'hilarious', 'ridiculous', 'absurd'
  ]

  sentence_lower = sentence.lower()
  engagement_score = sum(1 for word in engagement_words if f' {word} ' in f' {sentence_lower} ')

  # Exclamation marks and questions are good indicators
  if '!' in sentence or ('?' in sentence and len(sentence) < 50):
    engagement_score += 2

  return engagement_score > 0


def _extract_hook(text: str) -> str:
  """Extract an engaging hook from clip text.

  Strategy:
  1. Look for short questions (best hooks)
  2. Look for exclamations
  3. Look for commands/imperatives
  4. Fall back to first sentence
  """
  if not text:
    return "Amazing moment!"

  sentences = _extract_sentences(text)
  if not sentences:
    return "Amazing moment!"

  # Priority 1: Short questions (questions are great hooks!)
  for sent in sentences[:min(8, len(sentences))]:
    if '?' in sent and len(sent) <= 40:
      return sent.rstrip('?').strip()

  # Priority 2: Exclamations or imperatives
  for sent in sentences[:min(6, len(sentences))]:
    if '!' in sent and len(sent) <= 35:
      return sent.rstrip('!').strip()
    # Look for imperatives (commands)
    if sent.lower().startswith(('wait', 'watch', 'look', 'listen', 'come', 'hurry', 'go', 'try', 'stop', 'check')):
      if len(sent) <= 35:
        return sent.rstrip('!.?').strip()

  # Priority 3: Shocking words
  shock_words = ['what', 'how', 'never', 'impossible', 'unbelievable', 'crazy', 'insane', 'incredible']
  for sent in sentences[:min(6, len(sentences))]:
    if any(word in sent.lower() for word in shock_words) and len(sent) <= 40:
      return sent.rstrip('!.?').strip()

  # Priority 4: First sentence (usually good)
  first = sentences[0].strip().rstrip('!.?')
  if len(first) <= 40:
    return first

  # Priority 5: Shorten and use first sentence
  words = first.split()
  if len(words) > 8:
    return ' '.join(words[:8])

  return first[:40] if len(first) > 40 else first


def _extract_punchline(text: str) -> str:
  """Extract a punchline/payoff moment from clip text.

  Strategy:
  1. Look for answers to questions
  2. Look for the last interesting statement
  3. Look for strong conclusion statements
  4. Fall back to middle/end of text
  """
  if not text:
    return "Epic scene incoming"

  sentences = _extract_sentences(text)
  if not sentences:
    words = text.split()
    snippet = ' '.join(words[min(10, len(words)-5):min(20, len(words))])
    return snippet if len(snippet) <= 50 else snippet[:47]

  # Priority 1: Find answers to questions
  for i in range(len(sentences) - 1):
    if '?' in sentences[i]:
      answer = sentences[i+1].strip().rstrip('!.?')
      if 15 <= len(answer) <= 55:
        return answer
      if len(answer) > 55:
        words = answer.split()
        return ' '.join(words[:min(12, len(words))])

  # Priority 2: Look for last few sentences with strong words
  strong_words = ['will', 'must', 'should', 'can\'t', 'won\'t', 'never', 'always', 'have', 'remember', 'know', 'believe']
  for sent in reversed(sentences[-5:]):
    sent_clean = sent.rstrip('!.?').strip()
    if any(word in sent_clean.lower() for word in strong_words) and 15 <= len(sent_clean) <= 55:
      return sent_clean

  # Priority 3: Last non-empty sentence if reasonable length
  if len(sentences) >= 2:
    last = sentences[-1].rstrip('!.?').strip()
    if 15 <= len(last) <= 55:
      return last
    if len(last) > 55:
      words = last.split()
      return ' '.join(words[:min(12, len(words))])

  # Priority 4: Get middle/end segment
  mid_idx = max(1, len(sentences) // 2)
  for j in range(len(sentences) - 1, max(mid_idx - 2, -1), -1):
    sent_clean = sentences[j].rstrip('!.?').strip()
    if 15 <= len(sent_clean) <= 55:
      return sent_clean

  # Priority 5: Last resort - combine last few words
  words = text.split()
  if len(words) > 20:
    # Get from middle onwards
    start = max(0, len(words) - 20)
    snippet = ' '.join(words[start:start+12])
  else:
    snippet = ' '.join(words[-min(12, len(words)):])

  return snippet if len(snippet) <= 55 else snippet[:52]


def _call_ollama_generate(prompt: str, max_tokens: int = 100) -> str:
  """Call Ollama to generate smart text using a small model.

  Falls back gracefully if Ollama is not available.
  """
  try:
    import subprocess
    import json

    # Try to use ollama with mistral or similar small model
    result = subprocess.run(
      ["ollama", "run", "mistral", prompt],
      capture_output=True,
      text=True,
      timeout=5
    )

    if result.returncode == 0:
      return result.stdout.strip()
  except Exception:
    pass

  return None


def _generate_smart_hook_ollama(transcript: str, fallback_hook: str) -> str:
  """Use Ollama to generate a smart hook if available, else use fallback."""
  if not transcript:
    return fallback_hook

  # Truncate transcript to reasonable length for Ollama
  transcript_short = transcript[:300]

  prompt = f"""Based on this anime scene transcript, write a SHORT (max 5 words) punchy hook that makes viewers want to watch.

Transcript: "{transcript_short}"

Hook (SHORT and punchy):"""

  result = _call_ollama_generate(prompt, max_tokens=20)
  if result:
    # Clean up the result
    result = result.strip().rstrip('.!?')[:30]
    return result if result else fallback_hook

  return fallback_hook


def _generate_smart_punchline_ollama(transcript: str, fallback_punchline: str) -> str:
  """Use Ollama to generate a smart punchline if available, else use fallback."""
  if not transcript:
    return fallback_punchline

  # Truncate transcript to reasonable length
  transcript_short = transcript[:300]

  prompt = f"""Based on this anime scene transcript, write a SHORT (max 10 words) punchline that captures the cool/funny/shocking moment.

Transcript: "{transcript_short}"

Punchline (SHORT and impactful):"""

  result = _call_ollama_generate(prompt, max_tokens=30)
  if result:
    # Clean up the result
    result = result.strip().rstrip('.!?')[:50]
    return result if result else fallback_punchline

  return fallback_punchline




def build_overlay_vf(
    *,
    clip_index: int,
    clip: Candidate,
    work_dir: str,
    out_res: Optional[Tuple[int, int]],
    bar_pct: float,
    top_text_tpl: str,
    bottom_text_tpl: str,
    font: str,
    font_size: int,
) -> str:
  """Create vf filters that draw solid bars and top/bottom text.

  Uses drawtext textfile= to avoid escaping issues and allow UTF-8 text.
  Layout goals:
    - Top hook sits comfortably inside the top bar (not hugging the edge).
    - Bottom subtitle/punchline is fully visible above the bottom edge.
    - Text is wrapped to 1â€“3 short lines for readability.
  """
  # Select a sane reference size (after pad/scale). If unknown, rely on `h` (`ih`) in expressions.
  # Bar height in pixels (expression). Keep within [40px, 25%].
  bar_pct = float(bar_pct)
  if bar_pct <= 0:
    bar_pct = 0.16
  if bar_pct > 0.30:
    bar_pct = 0.30

  # Build dynamic text. Use transcript snippet as description.
  snippet = re.sub(r"\s+", " ", (clip.text or "").strip())
  if snippet:
    snippet = snippet[:160].rstrip() + ("..." if len(snippet) > 160 else "")
  else:
    snippet = "Epic moment incoming"

  # Generate intelligent hook and punchline from transcript
  hook = _extract_hook(clip.text or "")
  punchline = _extract_punchline(clip.text or "")

  # Try to enhance with Ollama if available, fall back to extracted versions
  ollama_hook = _generate_smart_hook_ollama(clip.text or "", hook)
  ollama_punchline = _generate_smart_punchline_ollama(clip.text or "", punchline)

  hook = ollama_hook if ollama_hook else hook
  punchline = ollama_punchline if ollama_punchline else punchline

  # Template variables: {desc}, {i}, {hook}, {punchline}
  def _fmt(tpl: str) -> str:
    tpl = (tpl or "").strip()
    if not tpl:
      return ""
    return tpl.format(desc=snippet, i=clip_index, hook=hook, punchline=punchline)

  raw_top = _fmt(top_text_tpl) or hook or "Wait for it..."
  raw_bottom = _fmt(bottom_text_tpl) or punchline or snippet

  # Wrap to keep within bars; hooks can be a bit shorter.
  top_text = _wrap_overlay_text(raw_top, max_chars=22, max_lines=2)
  bottom_text = _wrap_overlay_text(raw_bottom, max_chars=26, max_lines=3)

  top_file = _write_textfile_utf8(os.path.join(work_dir, f"overlay_{clip_index:02d}_top.txt"), top_text)
  bot_file = _write_textfile_utf8(os.path.join(work_dir, f"overlay_{clip_index:02d}_bottom.txt"), bottom_text)

  top_file_vf = _vf_escape_path(top_file)
  bot_file_vf = _vf_escape_path(bot_file)

  # Font handling:
  # - Prefer a font file if provided
  # - If a plain family name is provided, do NOT force font=... because some ffmpeg builds
  #   route that through fontconfig (and error on Windows). In that case we just omit it
  #   and let drawtext pick a default.
  font_arg = ""
  if font:
    f = (font or "").strip().strip('"')
    if os.path.exists(f):
      font_arg = f":fontfile='{_vf_escape_path(f)}'"
    else:
      font_arg = ""

  fs = int(font_size) if int(font_size) > 0 else 48
  if out_res is not None:
    _tw, th = out_res
    # 1920-height baseline -> fontsize ~ 54 (slightly smaller than before).
    fs = int(max(24, round(fs * (th / 1920.0))))

  bh = f"ih*{bar_pct:.4f}"

  # Place text vertically using simple percentages so ffmpeg's expression
  # parser on Windows doesn't choke on nested expressions. We keep the
  # text comfortably inside the bars.
  # Top: roughly centered in top bar (~half bar height from top).
  # Bottom: centered in bottom bar, a bit above the very bottom.
  top_y = "(h*{pct:.4f}-text_h)/2".format(pct=bar_pct)
  bot_y = "h-(h*{pct:.4f})+(h*{pct:.4f}-text_h)/2".format(pct=bar_pct)

  # Draw solid bars (black). Then draw text centered within bars.
  # Use alpha-safe colors; add subtle shadow for readability.
  return ",".join([
    f"drawbox=x=0:y=0:w=iw:h={bh}:color=black@1:t=fill",
    f"drawbox=x=0:y=ih-({bh}):w=iw:h={bh}:color=black@1:t=fill",
    (
      "drawtext="
      f"textfile='{top_file_vf}'{font_arg}:reload=0:"
      f"fontcolor=white:fontsize={fs}:line_spacing=4:shadowcolor=black:shadowx=2:shadowy=2:"
      "x=(w-text_w)/2:"
      f"y={top_y}"
    ),
    (
      "drawtext="
      f"textfile='{bot_file_vf}'{font_arg}:reload=0:"
      f"fontcolor=white:fontsize={max(22, int(fs*0.85))}:line_spacing=4:shadowcolor=black:shadowx=2:shadowy=2:"
      "x=(w-text_w)/2:"
      f"y={bot_y}"
    ),
  ])


def build_shorts_overlay_vf(
    *,
    clip_index: int,
    clip: Candidate,
    work_dir: str,
    out_res: Optional[Tuple[int, int]],
    top_text_tpl: str,
    bottom_text_tpl: str,
    font: str,
    font_size: int,
    top_pad_pct: float,
    bottom_pad_pct: float,
    border_w: int,
    border_color: str,
    shadow_x: int,
    shadow_y: int,
) -> str:
  """Shorts/TikTok-style overlay: large outlined text at top (+ optional bottom).

  Matches typical Shorts look:
    - no solid black bars
    - thick outline (borderw) for readability
    - centered text, multi-line wrapped

  Note: background blur/fill is handled by aspect/pad + target-res in `build_vf_chain`.
  """
  # Build dynamic text.
  snippet = re.sub(r"\s+", " ", (clip.text or "").strip())
  if snippet:
    snippet = snippet[:160].rstrip() + ("..." if len(snippet) > 160 else "")
  else:
    snippet = "Epic moment incoming"

  hook = _extract_hook(clip.text or "")
  punchline = _extract_punchline(clip.text or "")

  ollama_hook = _generate_smart_hook_ollama(clip.text or "", hook)
  ollama_punchline = _generate_smart_punchline_ollama(clip.text or "", punchline)
  hook = ollama_hook if ollama_hook else hook
  punchline = ollama_punchline if ollama_punchline else punchline

  def _fmt(tpl: str) -> str:
    tpl = (tpl or "").strip()
    if not tpl:
      return ""
    return tpl.format(desc=snippet, i=clip_index, hook=hook, punchline=punchline)

  raw_top = _fmt(top_text_tpl) or hook or "Wait for it..."
  raw_bottom = _fmt(bottom_text_tpl) or ""

  top_text = _wrap_overlay_text(raw_top, max_chars=24, max_lines=3)
  bottom_text = _wrap_overlay_text(raw_bottom, max_chars=28, max_lines=2) if raw_bottom.strip() else ""

  top_file = _write_textfile_utf8(os.path.join(work_dir, f"overlay_{clip_index:02d}_top.txt"), top_text)
  top_file_vf = _vf_escape_path(top_file)

  bot_file_vf = ""
  if bottom_text:
    bot_file = _write_textfile_utf8(os.path.join(work_dir, f"overlay_{clip_index:02d}_bottom.txt"), bottom_text)
    bot_file_vf = _vf_escape_path(bot_file)

  # Font handling (same policy as build_overlay_vf)
  font_arg = ""
  if font:
    f = (font or "").strip().strip('"')
    if os.path.exists(f):
      font_arg = f":fontfile='{_vf_escape_path(f)}'"

  fs = int(font_size) if int(font_size) > 0 else 56
  if out_res is not None:
    _tw, th = out_res
    fs = int(max(22, round(fs * (th / 1920.0))))

  # Place text with simple % padding from edges.
  # Use y=... as expression strings; keep it simple for Windows builds.
  top_pad_pct = float(top_pad_pct)
  bottom_pad_pct = float(bottom_pad_pct)
  if top_pad_pct <= 0:
    top_pad_pct = 0.06
  if bottom_pad_pct <= 0:
    bottom_pad_pct = 0.10

  top_y = f"h*{top_pad_pct:.4f}"
  bot_y = f"h-(text_h + h*{bottom_pad_pct:.4f})"

  # Normalize border args.
  bw = max(0, int(border_w))
  bx = int(shadow_x)
  by = int(shadow_y)
  bc = (border_color or "black").strip() or "black"

  parts: List[str] = []

  parts.append(
    (
      "drawtext="
      f"textfile='{top_file_vf}'{font_arg}:reload=0:"
      f"fontcolor=white:fontsize={fs}:line_spacing=6:"
      f"borderw={bw}:bordercolor={bc}:"
      f"shadowcolor=black@0.75:shadowx={bx}:shadowy={by}:"
      "x=(w-text_w)/2:"
      f"y={top_y}"
    )
  )

  if bot_file_vf:
    parts.append(
      (
        "drawtext="
        f"textfile='{bot_file_vf}'{font_arg}:reload=0:"
        f"fontcolor=white:fontsize={max(20, int(fs*0.72))}:line_spacing=6:"
        f"borderw={max(0, int(round(bw*0.85)))}:bordercolor={bc}:"
        f"shadowcolor=black@0.75:shadowx={bx}:shadowy={by}:"
        "x=(w-text_w)/2:"
        f"y={bot_y}"
      )
    )

  return ",".join(parts)


# ----------------------------
# ffprobe
# ----------------------------
def ffprobe_duration(video_path: str) -> float:
  """Return video duration in seconds.

  Primary method: ffprobe.
  Fallback: parse duration from `ffmpeg -i` output.
  """
  # Use args list so paths with spaces/apostrophes work on Windows
  probe_variants: List[List[str]] = [
    [
      FFPROBE,
      "-v", "error",
      "-show_entries", "format=duration",
      "-of", "default=nk=1:nw=1",
      video_path,
    ],
    [
      FFPROBE,
      "-hide_banner",
      "-v", "error",
      "-show_entries", "format=duration",
      "-of", "json",
      video_path,
    ],
  ]

  last_err = ""
  for args in probe_variants:
    try:
      p = subprocess.run(args, check=True, capture_output=True, text=True, encoding="utf-8", errors="ignore")
      out = ((p.stdout or "") + (p.stderr or "")).strip()
      if not out:
        continue
      # default output is a bare float; json output is {"format":{"duration":"..."}}
      m = re.search(r"([0-9]+(?:\.[0-9]+)?)", out)
      if m:
        return float(m.group(1))
    except subprocess.CalledProcessError as e:
      last_err = ((e.stdout or "") + (e.stderr or "")).strip()
      continue
    except Exception as e:
      last_err = str(e)
      continue

  # Fallback: parse duration from ffmpeg -i
  try:
    p = subprocess.run([FFMPEG, "-hide_banner", "-i", video_path], check=False, capture_output=True, text=True, encoding="utf-8", errors="ignore")
    txt = (p.stdout or "") + (p.stderr or "")
    # Example: Duration: 00:23:19.55,
    m = re.search(r"Duration:\s*(\d\d):(\d\d):(\d\d(?:\.\d+)?)", txt)
    if m:
      hh = int(m.group(1))
      mm = int(m.group(2))
      ss = float(m.group(3))
      return hh * 3600 + mm * 60 + ss
  except Exception:
    pass

  raise RuntimeError(
    "Could not determine video duration. ffprobe failed and ffmpeg fallback failed. "
    f"ffprobe error: {last_err[:500]}"
  )


def clamp(x: float, lo: float, hi: float) -> float:
  return max(lo, min(hi, x))


def tokenize(text: str) -> List[str]:
  text = re.sub(r"[^\w\s']+", " ", text.lower())
  return [t for t in text.split() if t]


# ----------------------------
# Naming + caching helpers
# ----------------------------
def safe_basename_no_ext(path: str) -> str:
  base = os.path.splitext(os.path.basename(path))[0]
  # Keep it Windows-friendly
  base = re.sub(r"[^A-Za-z0-9._ -]+", "_", base)
  return base.strip() or "video"


def cached_work_paths(video_path: str, work_dir: str) -> Tuple[str, str]:
  """Return (wav_path, srt_path) for a given input video, using stable names."""
  stem = safe_basename_no_ext(video_path)
  wav_path = os.path.join(work_dir, f"{stem}_audio_16k_mono.wav")
  srt_path = os.path.join(work_dir, f"{stem}.srt")
  return wav_path, srt_path


def _asr_cache_meta_path(video_path: str, work_dir: str) -> str:
  stem = safe_basename_no_ext(video_path)
  return os.path.join(work_dir, f"{stem}.asr.json")


def _load_json(path: str) -> Optional[dict]:
  try:
    with open(path, "r", encoding="utf-8") as f:
      return json.load(f)
  except Exception:
    return None


def _write_json(path: str, data: dict) -> None:
  try:
    with open(path, "w", encoding="utf-8") as f:
      json.dump(data, f, indent=2)
  except Exception:
    pass


# ----------------------------
# Transcript (SRT)
# ----------------------------
def srt_time_to_sec(t: str) -> float:
  hh, mm, rest = t.split(":")
  ss, ms = rest.split(",")
  return int(hh) * 3600 + int(mm) * 60 + int(ss) + int(ms) / 1000.0


def load_srt(path: str) -> List[Segment]:
  with open(path, "r", encoding="utf-8", errors="ignore") as f:
    data = f.read()
  blocks = re.split(r"\n\s*\n", data.strip())
  segs: List[Segment] = []
  for b in blocks:
    lines = [ln.strip() for ln in b.splitlines() if ln.strip()]
    if len(lines) < 3:
      continue
    m = re.match(
      r"(\d\d:\d\d:\d\d,\d\d\d)\s*-->\s*(\d\d:\d\d:\d\d,\d\d\d)",
      lines[1],
    )
    if not m:
      continue
    start = srt_time_to_sec(m.group(1))
    end = srt_time_to_sec(m.group(2))
    text = " ".join(lines[2:]).strip()
    text = re.sub(r"\s+", " ", text)
    if text:
      segs.append(Segment(start, end, text))
  return segs


def _format_srt_time(seconds: float) -> str:
  seconds = max(0.0, float(seconds))
  ms = int(round((seconds - int(seconds)) * 1000))
  s = int(seconds) % 60
  m = (int(seconds) // 60) % 60
  h = int(seconds) // 3600
  return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _write_srt(path: str, entries: List[Tuple[float, float, str]]) -> None:
  with open(path, "w", encoding="utf-8") as f:
    for i, (st, en, txt) in enumerate(entries, start=1):
      f.write(f"{i}\n")
      f.write(f"{_format_srt_time(st)} --> {_format_srt_time(en)}\n")
      f.write((txt or "").strip() + "\n\n")


def run_faster_whisper_to_srt(
    wav_path: str,
    srt_path: str,
    language: str = "auto",
    model: str = "small",
    device: str = "auto",
    compute_type: str = "auto",
) -> str:
  """Transcribe wav -> SRT using faster-whisper (ctranslate2).

  Important: On Windows, CUDA builds can fail hard if cuDNN DLLs aren't on PATH.
  This function avoids selecting CUDA when cuDNN isn't discoverable, and will
  fall back to CPU automatically if CUDA initialization fails.
  """
  try:
    from faster_whisper import WhisperModel
  except Exception as e:
    raise RuntimeError("faster-whisper is not installed. Install it or switch --asr-backend whispercpp") from e

  # Decide device deterministically.
  if device == "cuda":
    fw_device = "cuda"
  elif device == "cpu":
    fw_device = "cpu"
  else:
    # auto: prefer CUDA only if CUDA is available AND cuDNN DLLs are discoverable.
    fw_device = "cpu"
    can_try_cuda = _windows_has_cudnn_on_path()
    if not can_try_cuda and os.name == "nt":
      print("[ASR] cuDNN DLLs not found on PATH; skipping CUDA and using CPU.")
    try:
      if can_try_cuda:
        import ctranslate2  # type: ignore
        if hasattr(ctranslate2, "get_cuda_device_count") and ctranslate2.get_cuda_device_count() > 0:
          fw_device = "cuda"
    except Exception:
      fw_device = "cpu"

  # Pick compute type defaults.
  if compute_type != "auto":
    fw_compute = compute_type
  else:
    fw_compute = "float16" if fw_device == "cuda" else "int8"

  def _do(device_name: str, compute_name: str) -> str:
    print(f"[ASR] faster-whisper model={model} device={device_name} compute_type={compute_name}")
    wm = WhisperModel(model, device=device_name, compute_type=compute_name)
    segments, _info = wm.transcribe(
      wav_path,
      language=None if (not language or language == "auto") else language,
      beam_size=1,
      vad_filter=True,
    )

    entries: List[Tuple[float, float, str]] = []
    for s in segments:
      txt = (s.text or "").strip()
      if not txt:
        continue
      entries.append((float(s.start), float(s.end), txt))

    if not entries:
      raise RuntimeError("faster-whisper produced 0 segments")

    _write_srt(srt_path, entries)
    return srt_path

  # First attempt (maybe CUDA).
  try:
    return _do(fw_device, fw_compute)
  except Exception as e:
    msg = str(e)
    # Common Windows CUDA/cuDNN failures:
    cuda_like = any(
      k in msg.lower()
      for k in [
        "cudnn",
        "cublas",
        "cuda",
        "invalid handle",
        "cannot load symbol",
        "could not locate",
        "loadlibrary",
        "dll",
      ]
    )
    if fw_device == "cuda" and cuda_like:
      print(f"[ASR] CUDA transcription failed ({type(e).__name__}: {e}). Falling back to CPU (int8).")
      return _do("cpu", "int8")
    raise


# ----------------------------
# Whisper.cpp integration
# ----------------------------
def extract_audio_wav(video_path: str, wav_path: str) -> None:
  args = [
    FFMPEG, "-y",
    "-i", video_path,
    "-vn",
    "-ac", "1",
    "-ar", "16000",
    "-c:a", "pcm_s16le",
    wav_path,
  ]
  run(args)


def whisper_cli_help() -> str:
  # Resolve whisper-cli lazily so the rest of the script can run without it.
  global WHISPER_CLI
  if not WHISPER_CLI:
    try:
      WHISPER_CLI = resolve_exe(env_var="ANIMECLIPS_WHISPER_CLI", default_names=["whisper-cli", "whisper-cli.exe"], label="whisper-cli")
    except Exception:
      return ""

  try:
    p = subprocess.run([WHISPER_CLI, "-h"], check=True, capture_output=True, text=True, encoding="utf-8", errors="ignore")
    return (p.stdout or "") + (p.stderr or "")
  except Exception:
    return ""


def whisper_cli_supports_flag(flag: str) -> bool:
  out = whisper_cli_help()
  return flag in out if out else False


def run_whisper_to_srt(
    wav_path: str,
    out_dir: str,
    language: str = "auto",
    threads: int = 6,
    use_vad: bool = True,
    base_name: Optional[str] = None,
) -> str:
  """Run whisper.cpp and return an SRT path."""

  global WHISPER_CLI, WHISPER_MODEL, WHISPER_VAD_MODEL

  # Resolve whisper-cli + model paths.
  if not WHISPER_CLI:
    WHISPER_CLI = resolve_exe(env_var="ANIMECLIPS_WHISPER_CLI", default_names=["whisper-cli", "whisper-cli.exe"], label="whisper-cli")

  if not WHISPER_MODEL:
    raise FileNotFoundError(
      "whisper.cpp model not configured. Set ANIMECLIPS_WHISPER_MODEL to a ggml/gguf model path, "
      "or switch to --asr-backend faster-whisper."
    )

  exists_or_raise(WHISPER_CLI, "whisper-cli")
  exists_or_raise(WHISPER_MODEL, "whisper model")
  if use_vad:
    if not WHISPER_VAD_MODEL:
      raise FileNotFoundError(
        "whisper.cpp VAD model not configured. Set ANIMECLIPS_WHISPER_VAD_MODEL or disable VAD."
      )
    exists_or_raise(WHISPER_VAD_MODEL, "VAD model")

  os.makedirs(out_dir, exist_ok=True)

  # Use stable name when possible (cache-friendly)
  if base_name:
    base = os.path.join(out_dir, base_name)
  else:
    base = os.path.join(out_dir, f"whisper_{int(time.time())}")

  common = [
    WHISPER_CLI,
    "-m", WHISPER_MODEL,
    "-f", wav_path,
    "-osrt",
    "-of", base,
    "-t", str(int(threads)),
  ]
  if language and language != "auto":
    common += ["-l", language]

  vad_variants: List[List[str]] = []
  if use_vad:
    # Try likely VAD flags; fall back to no VAD if not supported.
    if whisper_cli_supports_flag("vad-model") or whisper_cli_supports_flag("--vad-model"):
      vad_variants.append(["--vad-model", WHISPER_VAD_MODEL])
    if whisper_cli_supports_flag("\n  -vm") or whisper_cli_supports_flag(" -vm"):
      vad_variants.append(["-vm", WHISPER_VAD_MODEL])
    if whisper_cli_supports_flag("--vad"):
      vad_variants.append(["--vad"])
    vad_variants.append([])  # fallback
  else:
    vad_variants = [[]]

  last_err = None
  for extra in vad_variants:
    args = common + extra
    try:
      run(args)
      srt_path = base + ".srt"
      if os.path.exists(srt_path):
        return srt_path
      # Fallback scan
      for fn in os.listdir(out_dir):
        if fn.lower().endswith(".srt") and fn.lower().startswith(os.path.basename(base).lower()):
          return os.path.join(out_dir, fn)
      raise RuntimeError("whisper ran but .srt not found in out_dir")
    except subprocess.CalledProcessError as e:
      last_err = e
      print("whisper failed with these args, retrying if possible:", extra)
      continue

  raise RuntimeError(f"whisper failed. Last error: {last_err}")


def ensure_transcript(
    video_path: str,
    transcript_path: Optional[str],
    work_dir: str,
    language: str,
    asr: Optional[ASRConfig] = None,
) -> str:
  """Return the SRT transcript path.

  Priority:
  1) user-provided --transcript
  2) cached {work_dir}/{video_stem}.srt (if ASR config matches)
  3) generate once
  """
  if transcript_path:
    exists_or_raise(transcript_path, "transcript")
    return transcript_path

  asr = asr or ASRConfig()
  os.makedirs(work_dir, exist_ok=True)

  wav_path, srt_cache_path = cached_work_paths(video_path, work_dir)
  meta_path = _asr_cache_meta_path(video_path, work_dir)

  meta = _load_json(meta_path) or {}
  want = {"backend": asr.backend, "model": asr.model, "device": asr.device, "compute_type": asr.compute_type, "threads": int(asr.threads)}
  meta_ok = meta.get("asr") == want

  # Reuse cached SRT if present AND ASR config matches
  if os.path.exists(srt_cache_path) and meta_ok:
    print(f"Reusing existing transcript: {srt_cache_path}")
    return srt_cache_path

  # Reuse cached WAV if present; otherwise extract once
  if os.path.exists(wav_path):
    print(f"Reusing existing WAV: {wav_path}")
  else:
    extract_audio_wav(video_path, wav_path)

  base_name = os.path.splitext(os.path.basename(srt_cache_path))[0]

  backend = asr.backend
  if backend == "auto":
    # Prefer faster-whisper if available
    try:
      import faster_whisper  # noqa: F401
      backend = "faster-whisper"
    except Exception:
      backend = "whispercpp"

  if backend == "faster-whisper":
    # Generate directly to cache path
    run_faster_whisper_to_srt(
      wav_path=wav_path,
      srt_path=srt_cache_path,
      language=language,
      model=asr.model,
      device=asr.device,
      compute_type=asr.compute_type,
    )
    _write_json(meta_path, {"video": video_path, "asr": want, "ts": int(time.time())})
    return srt_cache_path

  # whispercpp
  srt_path = run_whisper_to_srt(
    wav_path=wav_path,
    out_dir=work_dir,
    language=language,
    threads=int(asr.threads),
    use_vad=True,
    base_name=base_name,
  )

  if os.path.abspath(srt_path) != os.path.abspath(srt_cache_path):
    try:
      with open(srt_path, "r", encoding="utf-8", errors="ignore") as src, open(srt_cache_path, "w", encoding="utf-8") as dst:
        dst.write(src.read())
      srt_path = srt_cache_path
    except Exception:
      pass

  _write_json(meta_path, {"video": video_path, "asr": want, "ts": int(time.time())})
  return srt_path


# ----------------------------
# Engagement scoring
# ----------------------------
ENGAGEMENT_NAMES_BOOST = 1.0


def transcript_name_list(segs: List[Segment]) -> List[str]:
  """Heuristic extraction of important 'names' from transcript.

  Without NER, we approximate by:
  - tokens that appear often
  - length >= 3
  - not in stopwords
  """
  freq: Dict[str, int] = {}
  for s in segs:
    for t in tokenize(s.text):
      if t in STOPWORDS or len(t) < 3:
        continue
      freq[t] = freq.get(t, 0) + 1
  # keep top frequent terms, but not too many
  ranked = sorted(freq.items(), key=lambda kv: kv[1], reverse=True)
  top = [w for (w, c) in ranked[:25] if c >= 3]
  return top


def estimate_dialogue_score(text: str, names: List[str]) -> float:
  toks = tokenize(text)
  if not toks:
    return 0.0
  words = [t for t in toks if t not in STOPWORDS]
  # dialogue tends to have more words and more unique content
  richness = keyword_richness(text)
  density = content_density(text)
  # boost for important names/terms
  hits = 0
  if names:
    s = " " + " ".join(toks) + " "
    for nm in names:
      if f" {nm} " in s:
        hits += 1
  name_boost = clamp(hits / max(1, len(names) / 5), 0.0, 1.0)
  score = 0.45 * density + 0.35 * richness + 0.20 * name_boost
  # small boost for longer speech blocks (without going crazy)
  score *= clamp(len(words) / 120.0, 0.6, 1.2)
  return clamp(score, 0.0, 1.0)


def estimate_action_score(video_path: str, start: float, end: float, scene_cuts: List[float], silence_starts: List[float], silence_ends: List[float]) -> float:
  """Estimate action intensity from:
  - cut rate (cuts per second)
  - low silence ratio (more continuous loudness)
  - audio 'volume' proxy via ffmpeg astats (optional)

  Kept fairly light: only uses already computed cut/silence features.
  """
  dur = max(0.1, end - start)
  cuts_in = [t for t in scene_cuts if start <= t <= end]
  cut_rate = clamp((len(cuts_in) / dur) / 0.25, 0.0, 1.0)  # ~1 cut/4s -> 1.0

  # Estimate silence fraction from boundaries (approx)
  sil_spans: List[Tuple[float, float]] = []
  for ss, se in zip(silence_starts, silence_ends):
    if se <= ss:
      continue
    if se < start or ss > end:
      continue
    sil_spans.append((max(start, ss), min(end, se)))
  sil_total = sum(max(0.0, b - a) for a, b in sil_spans)
  non_sil = clamp(1.0 - (sil_total / dur), 0.0, 1.0)

  return clamp(0.65 * cut_rate + 0.35 * non_sil, 0.0, 1.0)


def score_candidate_engagement(
    c: Candidate,
    mode: str,
    names: List[str],
    video_path: str,
    scene_cuts: List[float],
    silence_starts: List[float],
    silence_ends: List[float],
) -> float:
  base_text = score_candidate_text_only(c)
  dialogue = estimate_dialogue_score(c.text, names)
  action = estimate_action_score(video_path, c.start, c.end, scene_cuts, silence_starts, silence_ends)

  if mode == "action":
    score = 0.20 * base_text + 0.30 * dialogue + 0.50 * action
  elif mode == "dialogue":
    score = 0.20 * base_text + 0.65 * dialogue + 0.15 * action
  else:  # balanced
    score = 0.20 * base_text + 0.45 * dialogue + 0.35 * action

  return clamp(score, 0.0, 1.0)


# ----------------------------
# Aspect ratio formatting (crop/pad)
# ----------------------------
ASPECT_PRESETS: Dict[str, Tuple[int, int]] = {
  "source": (0, 0),
  "16:9": (16, 9),
  "9:16": (9, 16),
  "1:1": (1, 1),
  "4:5": (4, 5),
  "21:9": (21, 9),
}


def _parse_target_res(s: str) -> Optional[Tuple[int, int]]:
  if not s:
    return None
  m = re.match(r"^\s*(\d{2,5})\s*[xX]\s*(\d{2,5})\s*$", s)
  if not m:
    raise ValueError("--target-res must look like 1080x1920")
  w = int(m.group(1))
  h = int(m.group(2))
  if w <= 0 or h <= 0:
    raise ValueError("--target-res must be positive")
  return w, h


def build_vf_chain(
    *,
    aspect: str,
    mode: str,
    target_res: Optional[Tuple[int, int]],
) -> Optional[str]:
  """Build a safe vf chain for aspect enforcement + optional final scaling.

  - aspect=source: no crop/pad; only scale if target_res is set.
  - mode=fit: pad to aspect (no crop)
  - mode=fill: center crop to aspect
  - mode=smart is handled externally per-clip (call with fit/fill)
  """
  vf_parts: List[str] = []

  if aspect != "source":
    if aspect not in ASPECT_PRESETS:
      return None
    aw, ah = ASPECT_PRESETS[aspect]
    if aw <= 0 or ah <= 0:
      return None

    target = f"{aw}/{ah}"

    # Use explicit intermediate vars so crop/pad math is correct.
    if mode == "fill":
      crop_w = f"if(gt(iw/ih,({target})),ih*({target}),iw)"
      crop_h = f"if(gt(iw/ih,({target})),ih,iw/({target}))"
      vf_parts.append(f"crop=w='{crop_w}':h='{crop_h}':x='(iw-{crop_w})/2':y='(ih-{crop_h})/2'")
    else:  # fit
      pad_w = f"if(gt(iw/ih,({target})),iw,ih*({target}))"
      pad_h = f"if(gt(iw/ih,({target})),iw/({target}),ih)"
      vf_parts.append(
        f"pad=w='{pad_w}':h='{pad_h}':x='(ow-iw)/2':y='(oh-ih)/2':color=black"
      )

  # Optional: scale to exact output size.
  if target_res is not None:
    tw, th = target_res
    vf_parts.append(f"scale={tw}:{th}")

  # Normalize SAR at end (avoids weird display aspect on some players).
  if vf_parts:
    vf_parts.append("setsar=1")
    return ",".join(vf_parts)

  return None


# ----------------------------
# Export (safe re-encode)
# ----------------------------
def _export_clip_once(
    video_path: str,
    start: float,
    end: float,
    out_path: str,
    v_encoder: str,
    preset: str,
    crf_or_cq: int,
    use_hw_decode: bool,
    vf: Optional[str],
) -> None:
  args: List[str] = [FFMPEG, "-y", "-hide_banner", "-v", "error"]

  if use_hw_decode:
    if ffmpeg_supports_hwaccel("cuda"):
      args += ["-hwaccel", "cuda"]

  args += [
    "-ss", f"{start:.3f}",
    "-to", f"{end:.3f}",
    "-i", video_path,
  ]

  if vf:
    args += ["-vf", vf]

  # Video
  if v_encoder == "h264_nvenc":
    args += [
      "-c:v", "h264_nvenc",
      "-preset", preset,
      "-rc", "vbr",
      "-cq", str(int(crf_or_cq)),
      "-b:v", "0",
      "-pix_fmt", "yuv420p",
    ]
  else:
    # High quality but still fast-ish. Use a sane GOP to help social platforms.
    args += [
      "-c:v", "libx264",
      "-preset", preset,
      "-crf", str(int(crf_or_cq)),
      "-pix_fmt", "yuv420p",
      "-movflags", "+faststart",
    ]

  args += [
    "-c:a", "aac",
    "-b:a", "192k",
    out_path,
  ]
  run(args)


def export_clip_reencode(
    video_path: str,
    start: float,
    end: float,
    out_path: str,
    prefer_gpu: bool = False,
    use_hw_decode: bool = False,
    vf: Optional[str] = None,
    quality: str = "high",
) -> None:
  """Export one clip.

  quality:
    - high: preserve quality (default)
    - fast: faster, slightly larger quality loss
  """
  if quality == "fast":
    cpu_preset = "veryfast"
    cpu_crf = 21
    nvenc_preset = "p4"
    nvenc_cq = 23
  else:
    cpu_preset = "faster"
    cpu_crf = 18
    nvenc_preset = "p5"
    nvenc_cq = 21

  if prefer_gpu and ffmpeg_supports_encoder("h264_nvenc"):
    try:
      _export_clip_once(
        video_path=video_path,
        start=start,
        end=end,
        out_path=out_path,
        v_encoder="h264_nvenc",
        preset=nvenc_preset,
        crf_or_cq=nvenc_cq,
        use_hw_decode=use_hw_decode,
        vf=vf,
      )
      return
    except subprocess.CalledProcessError as e:
      print(f"NVENC export failed for {os.path.basename(out_path)}; retrying on CPU. Error: {e}")

  _export_clip_once(
    video_path=video_path,
    start=start,
    end=end,
    out_path=out_path,
    v_encoder="libx264",
    preset=cpu_preset,
    crf_or_cq=cpu_crf,
    use_hw_decode=False,
    vf=vf,
  )


def _export_job(job: Tuple[str, float, float, str, bool, bool, Optional[str], str]) -> str:
  video_path, start, end, out_path, prefer_gpu, use_hw_decode, vf, quality = job
  export_clip_reencode(
    video_path,
    start,
    end,
    out_path,
    prefer_gpu=prefer_gpu,
    use_hw_decode=use_hw_decode,
    vf=vf,
    quality=quality,
  )
  return out_path


# ----------------------------
# Boundary detection (scene cuts + silence)
# ----------------------------
def detect_scene_cuts(video_path: str, start: float, end: float, scene_thresh: float = 0.35) -> List[float]:
  dur = max(0.1, end - start)
  vf = f"select=gt(scene\\,{scene_thresh}),showinfo"
  args = [
    FFMPEG,
    "-hide_banner",
    "-v", "info",
    "-ss", f"{start:.3f}",
    "-t", f"{dur:.3f}",
    "-i", video_path,
    "-vf", vf,
    "-f", "null",
    "-",
  ]

  out = run_capture(args)
  cuts: List[float] = []
  for line in out.splitlines():
    m = re.search(r"pts_time:([0-9.]+)", line)
    if m:
      cuts.append(start + float(m.group(1)))

  return sorted(set(round(t, 3) for t in cuts))


def detect_silence_boundaries(
    video_path: str,
    start: float,
    end: float,
    silence_db: float = -35.0,
    min_silence_sec: float = 0.6
) -> Tuple[List[float], List[float]]:
  dur = max(0.1, end - start)
  af = f"silencedetect=noise={silence_db}dB:d={min_silence_sec}"
  args = [
    FFMPEG,
    "-hide_banner",
    "-v", "error",
    "-ss", f"{start:.3f}",
    "-t", f"{dur:.3f}",
    "-i", video_path,
    "-af", af,
    "-f", "null",
    "-",
  ]

  out = run_capture(args)
  s_starts: List[float] = []
  s_ends: List[float] = []
  for line in out.splitlines():
    m1 = re.search(r"silence_start:\s*([0-9.]+)", line)
    if m1:
      s_starts.append(start + float(m1.group(1)))
    m2 = re.search(r"silence_end:\s*([0-9.]+)", line)
    if m2:
      s_ends.append(start + float(m2.group(1)))

  return (
    sorted(set(round(t, 3) for t in s_starts)),
    sorted(set(round(t, 3) for t in s_ends)),
  )


# ----------------------------
# Candidate generation + snapping
# ----------------------------
STOPWORDS = set("""
a an and are as at be but by for from has have he her hers him his i if in into is it its
me my of on or our ours she that the their them they this to was we were what when where
who why will with you your yours not do did done just so
""".split())


def content_density(text: str) -> float:
  toks = tokenize(text)
  if not toks:
    return 0.0
  content = [t for t in toks if t not in STOPWORDS and len(t) >= 4]
  return clamp(len(content) / len(toks), 0.0, 1.0)


def keyword_richness(text: str) -> float:
  toks = [t for t in tokenize(text) if t not in STOPWORDS]
  if not toks:
    return 0.0
  uniq = len(set(toks))
  return clamp(uniq / max(1, len(toks)), 0.0, 1.0)


def make_candidates_from_segments(
    segs: List[Segment],
    search_start: float,
    search_end: float,
    min_len: float,
    max_len: float
) -> List[Candidate]:
  usable = [s for s in segs if s.end >= search_start and s.start <= search_end]
  usable.sort(key=lambda s: s.start)

  cands: List[Candidate] = []
  n = len(usable)
  for i in range(n):
    start = max(usable[i].start, search_start)
    text_parts = []
    end = start

    for j in range(i, n):
      if usable[j].start < start:
        continue
      if usable[j].start > search_end:
        break
      end = usable[j].end
      text_parts.append(usable[j].text)

      dur = end - start
      if dur >= min_len:
        if dur <= max_len and end <= search_end:
          cands.append(Candidate(start=start, end=end, text=" ".join(text_parts)))
        if dur >= max_len:
          break

  seen = set()
  uniq: List[Candidate] = []
  for c in cands:
    k = (round(c.start, 2), round(c.end, 2))
    if k in seen:
      continue
    seen.add(k)
    uniq.append(c)
  return uniq


def nearest_within(ts: float, points: List[float], window: float) -> Optional[float]:
  best = None
  best_d = 1e18
  for p in points:
    d = abs(p - ts)
    if d <= window and d < best_d:
      best = p
      best_d = d
  return best


def snap_clip(
    start: float,
    end: float,
    scene_cuts: List[float],
    silence_starts: List[float],
    silence_ends: List[float],
    subtitle_bounds: List[float],
    snap_scene_sec: float = 1.5,
    snap_silence_sec: float = 2.0,
    snap_sub_sec: float = 1.0,
) -> Tuple[float, float]:
  s = start
  e = end

  s2 = nearest_within(s, scene_cuts, snap_scene_sec) or \
       nearest_within(s, silence_ends, snap_silence_sec) or \
       nearest_within(s, subtitle_bounds, snap_sub_sec)
  if s2 is not None:
    s = s2

  e2 = nearest_within(e, scene_cuts, snap_scene_sec) or \
       nearest_within(e, silence_starts, snap_silence_sec) or \
       nearest_within(e, subtitle_bounds, snap_sub_sec)
  if e2 is not None:
    e = e2

  if e <= s:
    return start, end
  return s, e


def score_candidate_text_only(c: Candidate) -> float:
  dens = content_density(c.text)
  rich = keyword_richness(c.text)
  return clamp(0.6 * dens + 0.4 * rich, 0.0, 1.0)


def overlap_ratio(a: Candidate, b: Candidate) -> float:
  inter = max(0.0, min(a.end, b.end) - max(a.start, b.start))
  union = max(a.end, b.end) - min(a.start, b.start)
  return 0.0 if union <= 0 else inter / union


def select_top_n(cands: List[Candidate], n: int, max_overlap_ratio: float = 0.50) -> List[Candidate]:
  cands = sorted(cands, key=lambda x: x.score, reverse=True)
  picked: List[Candidate] = []
  for c in cands:
    if len(picked) >= n:
      break
    if any(overlap_ratio(c, p) > max_overlap_ratio for p in picked):
      continue
    picked.append(c)
  return picked


# ----------------------------
# ffmpeg/ffprobe resolution helpers (must be defined before main)
# ----------------------------

def _find_gyan_winget_ffmpeg_bin() -> Optional[str]:
  """Best-effort: locate the Winget Gyan FFmpeg bin folder on Windows."""
  la = os.environ.get("LOCALAPPDATA")
  if not la:
    return None
  root = os.path.join(la, "Microsoft", "WinGet", "Packages")
  if not os.path.isdir(root):
    return None

  try:
    for name in os.listdir(root):
      if not name.lower().startswith("gyan.ffmpeg"):
        continue
      pkg_dir = os.path.join(root, name)
      if not os.path.isdir(pkg_dir):
        continue
      for child in os.listdir(pkg_dir):
        if not child.lower().startswith("ffmpeg-"):
          continue
        bin_dir = os.path.join(pkg_dir, child, "bin")
        if os.path.exists(os.path.join(bin_dir, "ffmpeg.exe")) and os.path.exists(os.path.join(bin_dir, "ffprobe.exe")):
          return bin_dir
  except Exception:
    return None

  return None


def resolve_ffmpeg_tools() -> Tuple[str, str]:
  """Resolve (ffmpeg, ffprobe) paths with env/PATH + common Windows installs."""
  extra: List[str] = []
  gyan_bin = _find_gyan_winget_ffmpeg_bin()
  if gyan_bin:
    extra = [os.path.join(gyan_bin, "ffmpeg.exe"), os.path.join(gyan_bin, "ffprobe.exe")]

  ffmpeg = resolve_exe(
    env_var="ANIMECLIPS_FFMPEG",
    default_names=["ffmpeg", "ffmpeg.exe"],
    label="ffmpeg",
    extra_candidates=extra,
  )
  ffprobe = resolve_exe(
    env_var="ANIMECLIPS_FFPROBE",
    default_names=["ffprobe", "ffprobe.exe"],
    label="ffprobe",
    extra_candidates=extra,
  )
  return ffmpeg, ffprobe


# ----------------------------
# FFmpeg capability probing
# ----------------------------

def _cmd_exists(path: str) -> bool:
  try:
    return bool(path) and os.path.exists(path)
  except Exception:
    return False


def ffmpeg_supports_encoder(encoder: str) -> bool:
  if not _cmd_exists(FFMPEG):
    return False
  try:
    out = run_capture([FFMPEG, "-hide_banner", "-encoders"])
    return re.search(rf"\b{re.escape(encoder)}\b", out) is not None
  except Exception:
    return False


def ffmpeg_supports_hwaccel(hwaccel: str) -> bool:
  if not _cmd_exists(FFMPEG):
    return False
  try:
    out = run_capture([FFMPEG, "-hide_banner", "-hwaccels"])
    return re.search(rf"\b{re.escape(hwaccel)}\b", out) is not None
  except Exception:
    return False


# ----------------------------
# CUDA/cuDNN helpers
# ----------------------------

def _windows_has_cudnn_on_path(dll_names: Optional[List[str]] = None) -> bool:
  """Return True if cuDNN DLLs appear discoverable via PATH on Windows.

  faster-whisper/ctranslate2 loads cuDNN dynamically; if the DLLs are not on PATH
  it can crash the process (not always a clean Python exception).
  """
  if os.name != "nt":
    return True
  dlls = dll_names or ["cudnn_ops64_9.dll", "cudnn_cnn64_9.dll", "cudnn64_9.dll"]
  path = os.environ.get("PATH") or ""
  parts = [p.strip().strip('"') for p in path.split(os.pathsep) if p.strip()]
  for d in dlls:
    found = False
    for p in parts:
      try:
        if os.path.exists(os.path.join(p, d)):
          found = True
          break
      except Exception:
        continue
    if not found:
      return False
  return True


def exists_or_raise(path: str, label: str) -> None:
  if not os.path.exists(path):
    raise FileNotFoundError(f"{label} not found: {path}")


# ----------------------------
# Main
# ----------------------------

def main():
  ap = argparse.ArgumentParser(description="Generate N important, sensible clips from a video (auto transcript if missing).")
  ap.add_argument("--video", required=True, help="Input video path (mkv/mp4).")
  ap.add_argument("--num-clips", type=int, required=True, help="How many clips to generate.")
  ap.add_argument("--transcript", default=None, help="Optional SRT transcript path.")

  ap.add_argument("--skip-intro-sec", type=float, default=0.0, help="Seconds to skip from the start.")
  ap.add_argument("--skip-outro-sec", type=float, default=0.0, help="Seconds to skip from the end.")

  ap.add_argument("--len-preset", choices=["60-90", "60-120"], default="60-90")
  ap.add_argument("--outdir", default="out_clips", help="Output folder.")
  ap.add_argument("--workdir", default="work", help="Working folder (audio + generated transcripts).")

  ap.add_argument("--language", default="auto", help="Whisper language: auto/en/ja/etc.")
  ap.add_argument("--scene-thresh", type=float, default=0.35, help="Scene cut threshold; lower => more cuts.")
  ap.add_argument("--silence-db", type=float, default=-35.0)
  ap.add_argument("--min-silence-sec", type=float, default=0.6)

  ap.add_argument(
    "--engagement",
    choices=["balanced", "action", "dialogue"],
    default="balanced",
    help="How to prioritize highlights: action, dialogue, or balanced mix.",
  )

  ap.add_argument(
    "--aspect",
    choices=list(ASPECT_PRESETS.keys()),
    default="source",
    help="Output aspect ratio (source keeps original).",
  )
  ap.add_argument(
    "--aspect-mode",
    choices=["fit", "fill", "smart"],
    default="smart",
    help="fit=pad (no crop), fill=center crop, smart=auto per-clip.",
  )
  ap.add_argument(
    "--target-res",
    default="",
    help="Optional output resolution like 1080x1920. If set, clips will be scaled to exactly this size.",
  )

  # --- Overlays for padded vertical clips ---
  ap.add_argument(
    "--overlay",
    choices=["off", "auto", "on"],
    default="auto",
    help="Add attention text/bars to padded clips: off/auto/on.",
  )
  ap.add_argument(
    "--overlay-style",
    choices=["bars", "shorts"],
    default="bars",
    help="Overlay style when --overlay is active: bars=black bars top/bottom, shorts=outlined caption (like Shorts/TikTok).",
  )
  ap.add_argument(
    "--overlay-bar-pct",
    type=float,
    default=0.14,
    help="Bar height as fraction of output height (e.g., 0.14).",
  )
  ap.add_argument(
    "--overlay-top-pad-pct",
    type=float,
    default=0.06,
    help="For overlay-style=shorts: top padding as fraction of height.",
  )
  ap.add_argument(
    "--overlay-bottom-pad-pct",
    type=float,
    default=0.10,
    help="For overlay-style=shorts: bottom padding as fraction of height.",
  )
  ap.add_argument(
    "--overlay-top-text",
    default="{hook}",
    help="Top overlay text template. Supports {desc}, {hook}, {punchline}, and {i}.",
  )
  ap.add_argument(
    "--overlay-bottom-text",
    default="{punchline}",
    help="Bottom overlay text template. Supports {desc}, {hook}, {punchline}, and {i}.",
  )
  ap.add_argument(
    "--overlay-font",
    default=os.environ.get("ANIMECLIPS_OVERLAY_FONT", "Arial"),
    help="Font family name or path to a .ttf/.otf file.",
  )
  ap.add_argument(
    "--overlay-fontsize",
    type=int,
    default=56,
    help="Base font size (scaled with target-res height if set).",
  )
  ap.add_argument(
    "--overlay-borderw",
    type=int,
    default=10,
    help="For overlay-style=shorts: outline thickness (drawtext borderw).",
  )
  ap.add_argument(
    "--overlay-bordercolor",
    default="black",
    help="For overlay-style=shorts: outline color (drawtext bordercolor).",
  )
  ap.add_argument(
    "--overlay-shadowx",
    type=int,
    default=2,
    help="For overlay-style=shorts: shadow offset x.",
  )
  ap.add_argument(
    "--overlay-shadowy",
    type=int,
    default=2,
    help="For overlay-style=shorts: shadow offset y.",
  )

  ap.add_argument(
    "--quality",
    choices=["high", "fast"],
    default="high",
    help="Encoding quality target.",
  )

  ap.add_argument(
    "--jobs",
    type=int,
    default=max(1, min(4, (os.cpu_count() or 4))),
    help="Parallel clip export workers (default: up to 4).",
  )
  ap.add_argument(
    "--encoder",
    choices=["cpu", "nvenc", "auto"],
    default="auto",
    help="Video encoder: cpu=libx264, nvenc=h264_nvenc, auto=nvenc if available else cpu.",
  )
  ap.add_argument(
    "--hw-decode",
    action="store_true",
    help="Try hardware decode (CUDA) when using NVENC.",
  )

  ap.add_argument(
    "--asr-backend",
    choices=["auto", "whispercpp", "faster-whisper"],
    default="auto",
    help="ASR backend. auto prefers faster-whisper if installed, else whisper.cpp.",
  )
  ap.add_argument(
    "--asr-model",
    default="small",
    help="ASR model name (faster-whisper). Examples: tiny/base/small/medium/large-v3.",
  )
  ap.add_argument(
    "--asr-device",
    choices=["auto", "cpu", "cuda"],
    default="auto",
    help="ASR device for faster-whisper (cpu/cuda).",
  )
  ap.add_argument(
    "--asr-compute-type",
    default="auto",
    help="ASR compute type for faster-whisper (auto/float16/int8/int8_float16/etc.).",
  )
  ap.add_argument(
    "--asr-threads",
    type=int,
    default=6,
    help="Threads for whisper.cpp backend.",
  )

  args = ap.parse_args()

  # Resolve tools now (portable)
  global FFMPEG, FFPROBE
  FFMPEG, FFPROBE = resolve_ffmpeg_tools()

  exists_or_raise(args.video, "video")
  exists_or_raise(FFMPEG, "ffmpeg")
  exists_or_raise(FFPROBE, "ffprobe")

  os.makedirs(args.outdir, exist_ok=True)
  video_out_dir = os.path.join(args.outdir, safe_basename_no_ext(args.video))
  os.makedirs(video_out_dir, exist_ok=True)

  dur = ffprobe_duration(args.video)
  search_start = clamp(args.skip_intro_sec, 0.0, dur)
  search_end = clamp(dur - args.skip_outro_sec, search_start + 1.0, dur)

  target_res = _parse_target_res(args.target_res)

  asr_cfg = ASRConfig(
    backend=args.asr_backend,
    model=args.asr_model,
    device=args.asr_device,
    compute_type=args.asr_compute_type,
    threads=int(args.asr_threads),
  )

  transcript_path = ensure_transcript(args.video, args.transcript, args.workdir, args.language, asr=asr_cfg)
  segs = load_srt(transcript_path)
  if not segs:
    raise RuntimeError(f"Transcript parsed 0 segments: {transcript_path}")

  if args.len_preset == "60-90":
    min_len, max_len = 60.0, 90.0
  else:
    min_len, max_len = 60.0, 120.0

  scene_cuts = detect_scene_cuts(args.video, search_start, search_end, scene_thresh=args.scene_thresh)
  silence_starts, silence_ends = detect_silence_boundaries(
    args.video, search_start, search_end,
    silence_db=args.silence_db, min_silence_sec=args.min_silence_sec
  )
  subtitle_bounds = sorted(set(round(s.end, 3) for s in segs if search_start <= s.end <= search_end))

  cands = make_candidates_from_segments(segs, search_start, search_end, min_len=min_len, max_len=max_len)
  if not cands:
    raise RuntimeError("No candidates generated. Try len preset 60-120 or reduce intro/outro skip.")

  names = transcript_name_list(segs)
  for c in cands:
    c.score = score_candidate_engagement(
      c,
      mode=args.engagement,
      names=names,
      video_path=args.video,
      scene_cuts=scene_cuts,
      silence_starts=silence_starts,
      silence_ends=silence_ends,
    )

  snapped = []  # type: List[Candidate]
  for c in cands:
    s, e = snap_clip(
      c.start, c.end,
      scene_cuts=scene_cuts,
      silence_starts=silence_starts,
      silence_ends=silence_ends,
      subtitle_bounds=subtitle_bounds,
    )
    if (e - s) < (min_len * 0.80) or (e - s) > (max_len * 1.20):
      continue
    snapped.append(Candidate(start=s, end=e, text=c.text, score=c.score))

  if not snapped:
    raise RuntimeError("All candidates filtered out after snapping. Try 60-120 preset or relax thresholds.")

  picked = select_top_n(snapped, n=args.num_clips, max_overlap_ratio=0.50)

  debug_path = os.path.join(video_out_dir, "picked.json")
  with open(debug_path, "w", encoding="utf-8") as f:
    json.dump(
      {
        "video": args.video,
        "duration": dur,
        "search_start": search_start,
        "search_end": search_end,
        "transcript": transcript_path,
        "picked": [c.__dict__ for c in picked],
      },
      f,
      indent=2
    )

  if args.encoder == "cpu":
    prefer_gpu = False
  elif args.encoder == "nvenc":
    prefer_gpu = True
  else:
    prefer_gpu = ffmpeg_supports_encoder("h264_nvenc")

  jobs = max(1, int(args.jobs))
  if prefer_gpu:
    jobs = min(jobs, 3)

  vf_fit = build_vf_chain(aspect=args.aspect, mode="fit", target_res=target_res)
  vf_fill = build_vf_chain(aspect=args.aspect, mode="fill", target_res=target_res)

  export_jobs = []  # type: List[Tuple[str, float, float, str, bool, bool, Optional[str], str]]
  for i, c in enumerate(picked, start=1):
    clip_name = f"clip_{i:02d}_{safe_basename_no_ext(args.video)}_{c.start:.2f}-{c.end:.2f}.mp4"
    out_path = os.path.join(video_out_dir, clip_name)

    vf = None
    if args.aspect == "source" and target_res is None:
      vf = None
    elif args.aspect == "source":
      vf = build_vf_chain(aspect="source", mode="fit", target_res=target_res)
    elif args.aspect_mode == "fit":
      vf = vf_fit
    elif args.aspect_mode == "fill":
      vf = vf_fill
    else:
      vf = vf_fit

    # Append overlay filters (text + bars) after aspect padding/cropping.
    if _overlay_should_apply(aspect=args.aspect, mode=args.aspect_mode, overlay_mode=args.overlay):
      if str(args.overlay_style) == "shorts":
        overlay_vf = build_shorts_overlay_vf(
          clip_index=i,
          clip=c,
          work_dir=args.workdir,
          out_res=target_res,
          top_text_tpl=str(args.overlay_top_text),
          bottom_text_tpl=str(args.overlay_bottom_text),
          font=str(args.overlay_font),
          font_size=int(args.overlay_fontsize),
          top_pad_pct=float(args.overlay_top_pad_pct),
          bottom_pad_pct=float(args.overlay_bottom_pad_pct),
          border_w=int(args.overlay_borderw),
          border_color=str(args.overlay_bordercolor),
          shadow_x=int(args.overlay_shadowx),
          shadow_y=int(args.overlay_shadowy),
        )
      else:
        overlay_vf = build_overlay_vf(
          clip_index=i,
          clip=c,
          work_dir=args.workdir,
          out_res=target_res,
          bar_pct=float(args.overlay_bar_pct),
          top_text_tpl=str(args.overlay_top_text),
          bottom_text_tpl=str(args.overlay_bottom_text),
          font=str(args.overlay_font),
          font_size=int(args.overlay_fontsize),
        )
      vf = overlay_vf if not vf else f"{vf},{overlay_vf}"

    export_jobs.append((args.video, c.start, c.end, out_path, prefer_gpu, bool(args.hw_decode and prefer_gpu), vf, args.quality))

  if jobs == 1 or len(export_jobs) <= 1:
    for j in export_jobs:
      _export_job(j)
  else:
    print(f"Exporting {len(export_jobs)} clips with {jobs} parallel workers...")
    from concurrent.futures import ThreadPoolExecutor, as_completed
    with ThreadPoolExecutor(max_workers=jobs) as ex:
      futs = [ex.submit(_export_job, j) for j in export_jobs]
      for fut in as_completed(futs):
        _ = fut.result()

  print(f"\nDone. Wrote {len(picked)} clips to: {args.outdir}")
  print(f"Debug: {debug_path}")
  print(f"Transcript used: {transcript_path}")


if __name__ == "__main__":
  main()

