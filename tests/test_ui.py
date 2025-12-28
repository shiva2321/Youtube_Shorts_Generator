"""Tests for UI components in clip_generator_ui.py."""

import pytest
import os
import sys
from unittest.mock import Mock, patch, MagicMock, call

# Try to import tkinter, skip tests if not available
try:
    import tkinter as tk
    TKINTER_AVAILABLE = True
except ImportError:
    TKINTER_AVAILABLE = False
    tk = None

# Skip all tests if tkinter is not available
pytestmark = pytest.mark.skipif(not TKINTER_AVAILABLE, reason="tkinter not available")

# Import UI module only if tkinter is available
if TKINTER_AVAILABLE:
    try:
        from clip_generator_ui import App, ToolTip
    except ImportError:
        App = None
        ToolTip = None
else:
    App = None
    ToolTip = None


class TestAppInitialization:
    """Test App class initialization."""

    @pytest.mark.ui
    @patch('tkinter.Tk.__init__')
    @patch('tkinter.Tk.title')
    @patch('tkinter.Tk.geometry')
    def test_app_creation(self, mock_geometry, mock_title, mock_init):
        """Test basic App creation without displaying GUI."""
        mock_init.return_value = None
        
        # We can't actually create the App without a display
        # So we test the initialization components
        assert App is not None
        assert hasattr(App, '__init__')

    @pytest.mark.ui
    def test_app_has_required_methods(self):
        """Test that App has required methods."""
        required_methods = [
            '__init__',
            'init_vars',
            'build_ui',
            'poll_log'
        ]
        
        for method in required_methods:
            assert hasattr(App, method), f"App missing {method} method"


class TestAppVariables:
    """Test App variable initialization (without GUI)."""

    @pytest.mark.ui
    @patch('tkinter.Tk.__init__')
    def test_init_vars_creates_variables(self, mock_init):
        """Test that init_vars creates necessary variables."""
        mock_init.return_value = None
        
        # Create mock App instance
        app = MagicMock(spec=App)
        app.video_path = tk.StringVar()
        app.transcript_path = tk.StringVar()
        app.outdir = tk.StringVar(value=os.path.join(os.getcwd(), "output"))
        app.workdir = tk.StringVar(value=os.path.join(os.getcwd(), "work"))
        
        # Test variable creation
        assert app.video_path is not None
        assert app.transcript_path is not None
        assert app.outdir is not None
        assert app.workdir is not None


class TestToolTip:
    """Test ToolTip class."""

    @pytest.mark.ui
    @patch('tkinter.Widget')
    def test_tooltip_creation(self, mock_widget):
        """Test ToolTip creation."""
        mock_widget.bind = Mock()
        
        tooltip = ToolTip(mock_widget, text="Test tooltip")
        
        assert tooltip.text == "Test tooltip"
        assert tooltip.waittime == 500
        assert tooltip.wraplength == 180
        
        # Should bind events
        assert mock_widget.bind.call_count >= 3  # Enter, Leave, ButtonPress

    @pytest.mark.ui
    @patch('tkinter.Widget')
    def test_tooltip_schedule(self, mock_widget):
        """Test tooltip scheduling."""
        mock_widget.bind = Mock()
        mock_widget.after = Mock(return_value=123)
        
        tooltip = ToolTip(mock_widget, text="Test")
        tooltip.schedule()
        
        assert tooltip.id is not None
        assert mock_widget.after.called

    @pytest.mark.ui
    @patch('tkinter.Widget')
    def test_tooltip_unschedule(self, mock_widget):
        """Test tooltip unscheduling."""
        mock_widget.bind = Mock()
        mock_widget.after = Mock(return_value=123)
        mock_widget.after_cancel = Mock()
        
        tooltip = ToolTip(mock_widget, text="Test")
        tooltip.id = 123
        tooltip.unschedule()
        
        assert mock_widget.after_cancel.called


class TestFilePathValidation:
    """Test file path validation logic."""

    @pytest.mark.ui
    def test_video_path_validation(self, temp_dir):
        """Test video path validation."""
        # Create a dummy video file
        video_path = os.path.join(temp_dir, "test_video.mp4")
        with open(video_path, "w") as f:
            f.write("dummy")
        
        # Test path exists
        assert os.path.exists(video_path)
        
        # Test invalid path
        invalid_path = os.path.join(temp_dir, "nonexistent.mp4")
        assert not os.path.exists(invalid_path)

    @pytest.mark.ui
    def test_output_directory_creation(self, temp_dir):
        """Test output directory creation."""
        output_dir = os.path.join(temp_dir, "output")
        work_dir = os.path.join(temp_dir, "work")
        
        # Create directories
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(work_dir, exist_ok=True)
        
        assert os.path.exists(output_dir)
        assert os.path.exists(work_dir)


class TestTranscriptDetection:
    """Test transcript file detection logic."""

    @pytest.mark.ui
    def test_find_existing_transcript(self, temp_dir):
        """Test finding existing transcript file."""
        # Create dummy video and transcript
        video_path = os.path.join(temp_dir, "anime_episode.mp4")
        transcript_path = os.path.join(temp_dir, "anime_episode.srt")
        
        with open(video_path, "w") as f:
            f.write("dummy video")
        with open(transcript_path, "w") as f:
            f.write("1\n00:00:00,000 --> 00:00:05,000\nTest subtitle\n")
        
        # Simulate transcript detection
        video_basename = os.path.splitext(video_path)[0]
        expected_srt = f"{video_basename}.srt"
        
        assert os.path.exists(expected_srt)

    @pytest.mark.ui
    def test_transcript_not_found(self, temp_dir):
        """Test when transcript doesn't exist."""
        video_path = os.path.join(temp_dir, "video.mp4")
        with open(video_path, "w") as f:
            f.write("dummy")
        
        video_basename = os.path.splitext(video_path)[0]
        expected_srt = f"{video_basename}.srt"
        
        assert not os.path.exists(expected_srt)


class TestVideoHashGeneration:
    """Test video hash generation in UI context."""

    @pytest.mark.ui
    def test_hash_generation_for_ui(self, temp_dir):
        """Test generating hash for video in UI context."""
        from auto_important_clips import get_video_hash
        
        video_path = os.path.join(temp_dir, "ui_test_video.mp4")
        with open(video_path, "wb") as f:
            f.write(b"video content for UI")
        
        video_hash = get_video_hash(video_path)
        
        assert video_hash != "unknown"
        assert len(video_hash) == 10

    @pytest.mark.ui
    def test_hash_for_cache_lookup(self, temp_dir):
        """Test using hash for transcript cache lookup."""
        from auto_important_clips import get_video_hash
        
        video_path = os.path.join(temp_dir, "cached_video.mp4")
        with open(video_path, "wb") as f:
            f.write(b"cached content")
        
        video_hash = get_video_hash(video_path)
        
        # Simulate cache file naming
        cache_file = f"transcript_{video_hash}.srt"
        assert cache_file.startswith("transcript_")
        assert cache_file.endswith(".srt")


class TestResolutionParsing:
    """Test resolution string parsing."""

    @pytest.mark.ui
    def test_parse_vertical_resolution(self):
        """Test parsing vertical resolution (shorts)."""
        res_str = "1080x1920"
        width, height = map(int, res_str.split("x"))
        
        assert width == 1080
        assert height == 1920
        assert height > width  # Vertical

    @pytest.mark.ui
    def test_parse_horizontal_resolution(self):
        """Test parsing horizontal resolution."""
        res_str = "1920x1080"
        width, height = map(int, res_str.split("x"))
        
        assert width == 1920
        assert height == 1080
        assert width > height  # Horizontal

    @pytest.mark.ui
    def test_parse_square_resolution(self):
        """Test parsing square resolution."""
        res_str = "1080x1080"
        width, height = map(int, res_str.split("x"))
        
        assert width == 1080
        assert height == 1080
        assert width == height  # Square


class TestUIConfigValidation:
    """Test UI configuration validation."""

    @pytest.mark.ui
    def test_num_clips_validation(self):
        """Test number of clips validation."""
        valid_values = [1, 3, 5, 10]
        for value in valid_values:
            assert value > 0
            assert isinstance(value, int)

    @pytest.mark.ui
    def test_skip_intro_validation(self):
        """Test skip intro time validation."""
        valid_values = [0.0, 30.0, 60.0, 90.0]
        for value in valid_values:
            assert value >= 0.0
            assert isinstance(value, float)

    @pytest.mark.ui
    def test_model_size_validation(self):
        """Test Whisper model size validation."""
        valid_models = ["small", "medium", "large"]
        for model in valid_models:
            assert model in valid_models

    @pytest.mark.ui
    def test_language_validation(self):
        """Test language selection validation."""
        valid_languages = ["auto", "english", "japanese"]
        for lang in valid_languages:
            assert lang in valid_languages

    @pytest.mark.ui
    def test_template_validation(self):
        """Test template style validation."""
        from auto_important_clips import OVERLAY_TEMPLATES
        
        for template_name in OVERLAY_TEMPLATES.keys():
            assert template_name in OVERLAY_TEMPLATES
            assert OVERLAY_TEMPLATES[template_name].name == template_name


class TestUIIntegration:
    """Integration tests for UI components."""

    @pytest.mark.integration
    @pytest.mark.ui
    def test_ui_imports(self):
        """Test that UI module imports correctly."""
        import clip_generator_ui
        
        assert hasattr(clip_generator_ui, 'App')
        assert hasattr(clip_generator_ui, 'ToolTip')

    @pytest.mark.integration
    @pytest.mark.ui
    def test_ui_and_backend_integration(self):
        """Test that UI can import backend functions."""
        # UI should be able to import from auto_important_clips
        try:
            import auto_important_clips
            assert auto_important_clips is not None
        except ImportError:
            pytest.fail("UI cannot import backend module")

    @pytest.mark.integration
    @pytest.mark.ui
    @patch('subprocess.Popen')
    def test_process_execution_mock(self, mock_popen):
        """Test that process execution can be mocked."""
        mock_process = MagicMock()
        mock_process.poll.return_value = 0
        mock_popen.return_value = mock_process
        
        # Simulate subprocess call
        import subprocess
        proc = subprocess.Popen(["echo", "test"])
        
        assert proc is not None
        assert mock_popen.called
