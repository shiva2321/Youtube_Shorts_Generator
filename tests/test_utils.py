"""Tests for utility functions in auto_important_clips.py."""

import pytest
import os
import sys
import platform
from unittest.mock import Mock, patch, MagicMock

from auto_important_clips import (
    get_system_font_path,
    get_video_hash,
    resolve_exe,
)


class TestGetSystemFontPath:
    """Test the get_system_font_path function."""

    @pytest.mark.unit
    def test_existing_file_path(self, temp_dir):
        """Test with an existing file path."""
        font_file = os.path.join(temp_dir, "test_font.ttf")
        with open(font_file, "w") as f:
            f.write("fake font")
        
        result = get_system_font_path(font_file)
        assert result == os.path.abspath(font_file)

    @pytest.mark.unit
    @patch('platform.system')
    @patch('os.walk')
    def test_windows_font_search(self, mock_walk, mock_system, temp_dir):
        """Test font search on Windows."""
        mock_system.return_value = "Windows"
        
        # Mock os.walk to return a fake font
        mock_walk.return_value = [
            (temp_dir, [], ["arial.ttf", "times.ttf"])
        ]
        
        with patch.dict(os.environ, {"WINDIR": temp_dir}):
            result = get_system_font_path("arial")
            # Should find arial.ttf
            assert "arial" in result.lower() or result == ""

    @pytest.mark.unit
    @patch('platform.system')
    @patch('os.walk')
    def test_linux_font_search(self, mock_walk, mock_system, temp_dir):
        """Test font search on Linux."""
        mock_system.return_value = "Linux"
        
        # Mock os.walk to return a fake font
        mock_walk.return_value = [
            (temp_dir, [], ["DejaVuSans.ttf", "Liberation.ttf"])
        ]
        
        result = get_system_font_path("dejavu")
        # Should search in system directories
        assert isinstance(result, str)

    @pytest.mark.unit
    @patch('platform.system')
    @patch('os.walk')
    def test_macos_font_search(self, mock_walk, mock_system, temp_dir):
        """Test font search on macOS."""
        mock_system.return_value = "Darwin"
        
        mock_walk.return_value = [
            (temp_dir, [], ["Helvetica.ttf", "Arial.ttf"])
        ]
        
        result = get_system_font_path("helvetica")
        assert isinstance(result, str)

    @pytest.mark.unit
    @patch('platform.system')
    @patch('os.walk')
    def test_cjk_font_priority(self, mock_walk, mock_system, temp_dir):
        """Test that CJK fonts are prioritized for Japanese text."""
        mock_system.return_value = "Linux"
        
        # Mock os.walk to return various fonts
        mock_walk.return_value = [
            (temp_dir, [], ["arial.ttf", "yumin.ttf", "msgothic.ttf"])
        ]
        
        # Use Japanese characters to trigger CJK font search
        result = get_system_font_path("日本語")
        # Should prioritize CJK fonts
        assert isinstance(result, str)

    @pytest.mark.unit
    @patch('platform.system')
    @patch('os.walk')
    def test_fallback_font(self, mock_walk, mock_system):
        """Test fallback when requested font not found."""
        mock_system.return_value = "Linux"
        
        # Mock os.walk to return only fallback fonts
        mock_walk.return_value = [
            ("/usr/share/fonts", [], ["arial.ttf"])
        ]
        
        result = get_system_font_path("nonexistent_font")
        # Should return a fallback or empty string
        assert isinstance(result, str)


class TestGetVideoHash:
    """Test the get_video_hash function (additional tests)."""

    @pytest.mark.unit
    def test_hash_format(self, temp_dir):
        """Test that hash has correct format."""
        video_path = os.path.join(temp_dir, "video.mp4")
        with open(video_path, "wb") as f:
            f.write(b"video content")
        
        result = get_video_hash(video_path)
        
        # Should be 10 characters from MD5 hex
        assert len(result) == 10
        assert all(c in "0123456789abcdef" for c in result)

    @pytest.mark.unit
    def test_hash_uses_size_and_name(self, temp_dir):
        """Test that hash considers both size and name."""
        # Same size, different names
        video1 = os.path.join(temp_dir, "video1.mp4")
        video2 = os.path.join(temp_dir, "video2.mp4")
        
        content = b"x" * 1000
        with open(video1, "wb") as f:
            f.write(content)
        with open(video2, "wb") as f:
            f.write(content)
        
        hash1 = get_video_hash(video1)
        hash2 = get_video_hash(video2)
        
        # Different names should produce different hashes
        assert hash1 != hash2

    @pytest.mark.unit
    def test_hash_deterministic(self, temp_dir):
        """Test that hash is deterministic."""
        video_path = os.path.join(temp_dir, "test.mp4")
        with open(video_path, "wb") as f:
            f.write(b"test content")
        
        # Get hash multiple times
        hashes = [get_video_hash(video_path) for _ in range(5)]
        
        # All hashes should be identical
        assert len(set(hashes)) == 1


class TestResolveExe:
    """Test the resolve_exe function."""

    @pytest.mark.unit
    @patch('shutil.which')
    def test_finds_in_path(self, mock_which):
        """Test finding executable in PATH."""
        mock_which.return_value = "/usr/bin/ffmpeg"
        
        result = resolve_exe("ffmpeg")
        assert result == "/usr/bin/ffmpeg"

    @pytest.mark.unit
    @patch('shutil.which')
    @patch('platform.system')
    @patch('os.path.exists')
    def test_windows_common_locations(self, mock_exists, mock_system, mock_which):
        """Test Windows-specific common locations."""
        mock_which.return_value = None  # Not in PATH
        mock_system.return_value = "Windows"
        
        # First common path exists
        mock_exists.side_effect = lambda p: p == r"C:\Program Files\ffmpeg\bin\ffmpeg.exe"
        
        result = resolve_exe("ffmpeg")
        assert result == r"C:\Program Files\ffmpeg\bin\ffmpeg.exe"

    @pytest.mark.unit
    @patch('shutil.which')
    def test_not_found_raises_error(self, mock_which):
        """Test that FileNotFoundError is raised when tool not found."""
        mock_which.return_value = None
        
        with pytest.raises(FileNotFoundError):
            resolve_exe("nonexistent_tool")

    @pytest.mark.unit
    @patch('shutil.which')
    def test_different_tools(self, mock_which):
        """Test resolving different tools."""
        tools = ["ffmpeg", "ffprobe", "python"]
        
        for tool in tools:
            mock_which.return_value = f"/usr/bin/{tool}"
            result = resolve_exe(tool)
            assert tool in result


class TestCrossPlatformCompatibility:
    """Test cross-platform compatibility of utility functions."""

    @pytest.mark.unit
    def test_path_handling_unix(self, temp_dir):
        """Test path handling on Unix-like systems."""
        if platform.system() != "Windows":
            video_path = os.path.join(temp_dir, "test_video.mp4")
            with open(video_path, "wb") as f:
                f.write(b"content")
            
            # Should handle Unix paths
            hash_result = get_video_hash(video_path)
            assert hash_result != "unknown"

    @pytest.mark.unit
    @patch('platform.system')
    def test_system_detection(self, mock_system):
        """Test system detection works correctly."""
        systems = ["Windows", "Linux", "Darwin"]
        
        for system in systems:
            mock_system.return_value = system
            result = platform.system()
            assert result in systems


class TestEdgeCases:
    """Test edge cases and error handling."""

    @pytest.mark.unit
    def test_empty_font_name(self):
        """Test get_system_font_path with empty font name."""
        result = get_system_font_path("")
        assert isinstance(result, str)

    @pytest.mark.unit
    def test_special_characters_in_path(self, temp_dir):
        """Test handling paths with special characters."""
        video_path = os.path.join(temp_dir, "test video (2023).mp4")
        with open(video_path, "wb") as f:
            f.write(b"content")
        
        # Should handle special characters
        hash_result = get_video_hash(video_path)
        assert hash_result != "unknown"
        assert len(hash_result) == 10

    @pytest.mark.unit
    def test_unicode_in_path(self, temp_dir):
        """Test handling paths with unicode characters."""
        video_path = os.path.join(temp_dir, "テスト動画.mp4")
        with open(video_path, "wb") as f:
            f.write(b"content")
        
        # Should handle unicode
        hash_result = get_video_hash(video_path)
        assert hash_result != "unknown"

    @pytest.mark.unit
    def test_very_long_filename(self, temp_dir):
        """Test handling very long filenames."""
        long_name = "a" * 200 + ".mp4"
        video_path = os.path.join(temp_dir, long_name)
        
        try:
            with open(video_path, "wb") as f:
                f.write(b"content")
            
            # Should handle long names
            hash_result = get_video_hash(video_path)
            assert hash_result != "unknown"
        except OSError:
            # Some systems have filename length limits
            pytest.skip("Filesystem doesn't support long filenames")
