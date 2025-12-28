"""Tests for core functionality of auto_important_clips.py."""

import pytest
import os
import sys
from unittest.mock import Mock, patch, MagicMock
import datetime

# Import functions to test
from auto_important_clips import (
    Candidate,
    OverlayTemplate,
    OVERLAY_TEMPLATES,
    format_timestamp,
    parse_srt,
    sanitize_filename,
    escape_text_for_ffmpeg,
    get_video_hash,
)


class TestCandidateDataclass:
    """Test the Candidate dataclass."""

    @pytest.mark.unit
    def test_candidate_creation(self):
        """Test creating a Candidate object."""
        candidate = Candidate(start=10.0, end=20.0, text="Test text", score=5.0)
        assert candidate.start == 10.0
        assert candidate.end == 20.0
        assert candidate.text == "Test text"
        assert candidate.score == 5.0

    @pytest.mark.unit
    def test_candidate_default_score(self):
        """Test Candidate with default score."""
        candidate = Candidate(start=5.0, end=15.0, text="No score")
        assert candidate.score == 0.0


class TestOverlayTemplate:
    """Test the OverlayTemplate dataclass."""

    @pytest.mark.unit
    def test_overlay_template_creation(self):
        """Test creating an OverlayTemplate."""
        template = OverlayTemplate(
            name="test",
            top_bar_color="blue",
            bottom_bar_color="red"
        )
        assert template.name == "test"
        assert template.top_bar_color == "blue"
        assert template.bottom_bar_color == "red"
        assert template.top_bar_opacity == 1.0  # default
        assert template.text_style == "outline"  # default

    @pytest.mark.unit
    def test_overlay_templates_exist(self):
        """Test that predefined overlay templates exist."""
        assert "simple" in OVERLAY_TEMPLATES
        assert "viral_shorts" in OVERLAY_TEMPLATES
        assert "neon_vibes" in OVERLAY_TEMPLATES
        assert "anime" in OVERLAY_TEMPLATES
        
        # Test a specific template
        viral = OVERLAY_TEMPLATES["viral_shorts"]
        assert viral.name == "viral_shorts"
        assert viral.show_branding is True
        assert viral.show_subscribe is True

    @pytest.mark.unit
    def test_overlay_template_with_gradient(self):
        """Test OverlayTemplate with gradient settings."""
        template = OverlayTemplate(
            name="gradient_test",
            use_gradient=True,
            gradient_colors=["#ff0000", "#00ff00"]
        )
        assert template.use_gradient is True
        assert len(template.gradient_colors) == 2


class TestFormatTimestamp:
    """Test the format_timestamp function."""

    @pytest.mark.unit
    def test_format_timestamp_zero(self):
        """Test formatting zero seconds."""
        result = format_timestamp(0.0)
        assert result == "00:00:00,000"

    @pytest.mark.unit
    def test_format_timestamp_simple(self):
        """Test formatting simple timestamps."""
        result = format_timestamp(65.5)
        assert result == "00:01:05,500"

    @pytest.mark.unit
    def test_format_timestamp_hours(self):
        """Test formatting timestamps with hours."""
        result = format_timestamp(3661.123)
        assert result == "01:01:01,123"

    @pytest.mark.unit
    def test_format_timestamp_milliseconds(self):
        """Test milliseconds are correctly formatted."""
        result = format_timestamp(10.999)
        assert result == "00:00:10,999"


class TestParseSrt:
    """Test the parse_srt function."""

    @pytest.mark.unit
    def test_parse_srt_basic(self, temp_srt_file):
        """Test parsing a basic SRT file."""
        candidates = parse_srt(temp_srt_file)
        assert len(candidates) == 3
        assert candidates[0].start == 0.0
        assert candidates[0].end == 5.0
        assert "first subtitle" in candidates[0].text.lower()

    @pytest.mark.unit
    def test_parse_srt_nonexistent_file(self):
        """Test parsing a nonexistent file returns empty list."""
        candidates = parse_srt("/nonexistent/file.srt")
        assert candidates == []

    @pytest.mark.unit
    def test_parse_srt_timing(self, temp_srt_file):
        """Test that SRT timing is correctly parsed."""
        candidates = parse_srt(temp_srt_file)
        # Second subtitle: 00:00:05,000 --> 00:00:10,000
        assert candidates[1].start == 5.0
        assert candidates[1].end == 10.0

    @pytest.mark.unit
    def test_parse_srt_multiline_text(self, temp_dir):
        """Test parsing SRT with multiline text."""
        multiline_srt = os.path.join(temp_dir, "multiline.srt")
        with open(multiline_srt, "w", encoding="utf-8") as f:
            f.write("""1
00:00:00,000 --> 00:00:05,000
Line one
Line two
Line three

""")
        candidates = parse_srt(multiline_srt)
        assert len(candidates) == 1
        assert "Line one Line two Line three" in candidates[0].text


class TestSanitizeFilename:
    """Test the sanitize_filename function."""

    @pytest.mark.unit
    def test_sanitize_basic(self):
        """Test basic filename sanitization."""
        result = sanitize_filename("My Video File.mp4")
        assert result == "My_Video_File.mp4"

    @pytest.mark.unit
    def test_sanitize_special_chars(self):
        """Test removing special characters."""
        result = sanitize_filename('Test: "video" <name>')
        assert '"' not in result
        assert '<' not in result
        assert '>' not in result
        assert ':' not in result

    @pytest.mark.unit
    def test_sanitize_slashes(self):
        """Test removing path separators."""
        result = sanitize_filename("path/to\\file.mp4")
        assert '/' not in result
        assert '\\' not in result

    @pytest.mark.unit
    def test_sanitize_length_limit(self):
        """Test filename length is limited to 100 characters."""
        long_name = "a" * 150 + ".mp4"
        result = sanitize_filename(long_name)
        assert len(result) <= 100


class TestEscapeTextForFfmpeg:
    """Test the escape_text_for_ffmpeg function."""

    @pytest.mark.unit
    def test_escape_single_quote(self):
        """Test escaping single quotes."""
        result = escape_text_for_ffmpeg("Don't stop")
        assert "\\'" in result

    @pytest.mark.unit
    def test_escape_colon(self):
        """Test escaping colons."""
        result = escape_text_for_ffmpeg("Time: 10:30")
        assert "\\:" in result

    @pytest.mark.unit
    def test_escape_comma(self):
        """Test escaping commas."""
        result = escape_text_for_ffmpeg("Red, Green, Blue")
        assert "\\," in result

    @pytest.mark.unit
    def test_escape_multiple(self):
        """Test escaping multiple special characters."""
        result = escape_text_for_ffmpeg("It's 10:30, isn't it?")
        assert "\\'" in result
        assert "\\:" in result
        assert "\\," in result


class TestGetVideoHash:
    """Test the get_video_hash function."""

    @pytest.mark.unit
    def test_video_hash_nonexistent(self):
        """Test hash for nonexistent video returns 'unknown'."""
        result = get_video_hash("/nonexistent/video.mp4")
        assert result == "unknown"

    @pytest.mark.unit
    def test_video_hash_consistency(self, temp_dir):
        """Test that hash is consistent for the same file."""
        video_path = os.path.join(temp_dir, "test_video.mp4")
        with open(video_path, "wb") as f:
            f.write(b"fake video content")
        
        hash1 = get_video_hash(video_path)
        hash2 = get_video_hash(video_path)
        assert hash1 == hash2
        assert len(hash1) == 10  # Should be 10 chars from MD5

    @pytest.mark.unit
    def test_video_hash_different_files(self, temp_dir):
        """Test that different files produce different hashes."""
        video1 = os.path.join(temp_dir, "video1.mp4")
        video2 = os.path.join(temp_dir, "video2.mp4")
        
        with open(video1, "wb") as f:
            f.write(b"content1")
        with open(video2, "wb") as f:
            f.write(b"content2")
        
        hash1 = get_video_hash(video1)
        hash2 = get_video_hash(video2)
        assert hash1 != hash2
