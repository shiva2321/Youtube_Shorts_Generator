"""Pytest fixtures and configuration for the test suite."""

import os
import sys
import tempfile
import pytest
from unittest.mock import MagicMock, Mock, patch
from dataclasses import dataclass
from typing import List

# Add parent directory to path to import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the modules we need to test
from auto_important_clips import (
    Candidate, OverlayTemplate, OVERLAY_TEMPLATES
)


@pytest.fixture
def sample_candidate():
    """Create a sample Candidate object for testing."""
    return Candidate(start=10.0, end=20.0, text="Sample text for testing", score=5.0)


@pytest.fixture
def sample_candidates():
    """Create a list of sample Candidate objects."""
    return [
        Candidate(start=0.0, end=5.0, text="First segment", score=3.0),
        Candidate(start=5.0, end=12.0, text="Second segment with more text", score=7.5),
        Candidate(start=15.0, end=25.0, text="Third segment", score=4.0),
        Candidate(start=30.0, end=40.0, text="Fourth segment", score=6.0),
    ]


@pytest.fixture
def sample_srt_content():
    """Sample SRT file content for testing."""
    return """1
00:00:00,000 --> 00:00:05,000
This is the first subtitle.

2
00:00:05,000 --> 00:00:10,000
This is the second subtitle.

3
00:00:10,000 --> 00:00:15,000
This is the third subtitle with more text!

"""


@pytest.fixture
def sample_overlay_template():
    """Create a sample OverlayTemplate for testing."""
    return OverlayTemplate(
        name="test_template",
        top_bar_color="blue",
        bottom_bar_color="red",
        top_bar_opacity=0.8,
        bottom_bar_opacity=0.7,
        text_style="outline",
        emoji_prefix="🔥 ",
        emoji_suffix=" 👀"
    )


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def temp_srt_file(temp_dir, sample_srt_content):
    """Create a temporary SRT file for testing."""
    srt_path = os.path.join(temp_dir, "test.srt")
    with open(srt_path, "w", encoding="utf-8") as f:
        f.write(sample_srt_content)
    return srt_path


@pytest.fixture
def mock_ffmpeg():
    """Mock FFmpeg subprocess calls."""
    with patch('subprocess.run') as mock_run, \
         patch('subprocess.check_output') as mock_output:
        mock_run.return_value = Mock(returncode=0)
        mock_output.return_value = "Duration: 00:23:45.67"
        yield {"run": mock_run, "output": mock_output}


@pytest.fixture
def mock_whisper():
    """Mock Whisper model to avoid loading it during tests."""
    with patch('auto_important_clips.whisper') as mock:
        mock_model = MagicMock()
        mock_model.transcribe.return_value = {
            "segments": [
                {"start": 0.0, "end": 5.0, "text": "Test segment 1"},
                {"start": 5.0, "end": 10.0, "text": "Test segment 2"},
            ],
            "language": "en"
        }
        mock.load_model.return_value = mock_model
        mock.load_audio.return_value = MagicMock()
        mock.pad_or_trim.return_value = MagicMock()
        mock.log_mel_spectrogram.return_value = MagicMock()
        yield mock


@pytest.fixture
def mock_config(temp_dir):
    """Create a mock configuration object similar to argparse.Namespace."""
    config = MagicMock()
    config.workdir = temp_dir  # Use actual temp_dir that exists
    config.template_style = "viral_shorts"
    config.channel_name = "TestChannel"
    config.overlay_top_text = "{hook}"
    config.overlay_bottom_text = "{punchline}"
    config.overlay_font = "Arial"
    config.aspect_mode = "fit"
    config.target_res = "1080x1920"
    config.encoder = "auto"
    config.catchy_hooks = False
    config.hook_seed = 0
    config.overlay_backend = "ass"
    return config


@pytest.fixture(autouse=True)
def reset_globals():
    """Reset global variables before each test."""
    # This ensures tests don't interfere with each other
    yield
    # Cleanup after test if needed
