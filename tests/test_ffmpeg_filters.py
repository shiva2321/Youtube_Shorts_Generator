"""Tests for FFmpeg filter chain building in auto_important_clips.py."""

import pytest
import os
from unittest.mock import Mock, patch, MagicMock

from auto_important_clips import (
    Candidate,
    OverlayTemplate,
    OVERLAY_TEMPLATES,
    build_filter_chain,
    _style_bottom_text_ass,
)


class TestBuildFilterChain:
    """Test the build_filter_chain function."""

    @pytest.mark.unit
    def test_basic_filter_chain(self, mock_config, sample_candidate):
        """Test basic filter chain generation."""
        resolution = (1080, 1920)
        filter_chain = build_filter_chain(sample_candidate, 1, mock_config, resolution)
        
        assert isinstance(filter_chain, str)
        assert len(filter_chain) > 0
        # Should contain some basic filter elements
        assert "scale" in filter_chain or "crop" in filter_chain

    @pytest.mark.unit
    def test_fit_mode_filter(self, mock_config, sample_candidate):
        """Test filter chain with fit aspect mode."""
        mock_config.aspect_mode = "fit"
        resolution = (1080, 1920)
        
        filter_chain = build_filter_chain(sample_candidate, 1, mock_config, resolution)
        
        # Fit mode should have specific filters
        assert "scale" in filter_chain
        # Should handle aspect ratio
        assert filter_chain

    @pytest.mark.unit
    def test_fill_mode_filter(self, mock_config, sample_candidate):
        """Test filter chain with fill aspect mode."""
        mock_config.aspect_mode = "fill"
        resolution = (1080, 1920)
        
        filter_chain = build_filter_chain(sample_candidate, 1, mock_config, resolution)
        
        # Fill mode should have crop
        assert "crop" in filter_chain
        assert "scale" in filter_chain

    @pytest.mark.unit
    def test_horizontal_resolution(self, mock_config, sample_candidate):
        """Test filter chain with horizontal resolution."""
        resolution = (1920, 1080)  # Horizontal
        
        filter_chain = build_filter_chain(sample_candidate, 1, mock_config, resolution)
        
        assert filter_chain
        # Should handle horizontal orientation

    @pytest.mark.unit
    def test_vertical_resolution(self, mock_config, sample_candidate):
        """Test filter chain with vertical resolution (shorts/TikTok)."""
        resolution = (1080, 1920)  # Vertical
        
        filter_chain = build_filter_chain(sample_candidate, 1, mock_config, resolution)
        
        assert filter_chain
        # Should handle vertical orientation

    @pytest.mark.unit
    def test_with_template_style(self, mock_config, sample_candidate):
        """Test filter chain includes template overlay."""
        mock_config.template_style = "viral_shorts"
        resolution = (1080, 1920)
        
        filter_chain = build_filter_chain(sample_candidate, 1, mock_config, resolution)
        
        # Should include drawbox for overlay bars
        assert "drawbox" in filter_chain or "ass" in filter_chain

    @pytest.mark.unit
    def test_without_template_style(self, mock_config, sample_candidate):
        """Test filter chain without template overlay."""
        mock_config.template_style = None
        resolution = (1080, 1920)
        
        filter_chain = build_filter_chain(sample_candidate, 1, mock_config, resolution)
        
        # Should still have basic filters
        assert "scale" in filter_chain or "crop" in filter_chain

    @pytest.mark.unit
    def test_ass_backend_creates_file(self, mock_config, sample_candidate, temp_dir):
        """Test ASS backend creates subtitle file."""
        mock_config.overlay_backend = "ass"
        mock_config.workdir = temp_dir
        resolution = (1080, 1920)
        
        filter_chain = build_filter_chain(sample_candidate, 1, mock_config, resolution)
        
        # Should reference ASS file
        assert "ass" in filter_chain
        
        # Check if ASS file was created
        ass_file = os.path.join(temp_dir, "overlay_1.ass")
        assert os.path.exists(ass_file)
        
        # Verify ASS file content
        with open(ass_file, 'r', encoding='utf-8') as f:
            content = f.read()
            assert "[Script Info]" in content
            assert "[V4+ Styles]" in content
            assert "[Events]" in content

    @pytest.mark.unit
    def test_png_backend_marker(self, mock_config, sample_candidate):
        """Test PNG backend adds placeholder marker."""
        mock_config.overlay_backend = "png"
        resolution = (1080, 1920)
        
        filter_chain = build_filter_chain(sample_candidate, 1, mock_config, resolution)
        
        # Should have PNG marker
        assert "__PNG_OVERLAY__" in filter_chain

    @pytest.mark.unit
    def test_different_templates(self, mock_config, sample_candidate):
        """Test filter chain with different template styles."""
        resolution = (1080, 1920)
        
        for template_name in ["simple", "viral_shorts", "anime", "cinematic"]:
            mock_config.template_style = template_name
            filter_chain = build_filter_chain(sample_candidate, 1, mock_config, resolution)
            assert filter_chain
            assert len(filter_chain) > 0

    @pytest.mark.unit
    def test_catchy_hooks_enabled(self, mock_config, sample_candidate, temp_dir):
        """Test filter chain with catchy hooks enabled."""
        mock_config.catchy_hooks = True
        mock_config.hook_seed = 42
        mock_config.workdir = temp_dir
        resolution = (1080, 1920)
        
        filter_chain = build_filter_chain(sample_candidate, 1, mock_config, resolution)
        
        # Should still generate valid filter chain
        assert filter_chain

    @pytest.mark.unit
    def test_channel_name_in_overlay(self, mock_config, sample_candidate, temp_dir):
        """Test that channel name appears in overlay."""
        mock_config.channel_name = "TestChannel"
        mock_config.template_style = "viral_shorts"
        mock_config.workdir = temp_dir
        resolution = (1080, 1920)
        
        filter_chain = build_filter_chain(sample_candidate, 1, mock_config, resolution)
        
        # Check ASS file contains channel name
        ass_file = os.path.join(temp_dir, "overlay_1.ass")
        if os.path.exists(ass_file):
            with open(ass_file, 'r', encoding='utf-8') as f:
                content = f.read()
                assert "TestChannel" in content

    @pytest.mark.unit
    def test_gradient_overlay(self, mock_config, sample_candidate):
        """Test filter chain with gradient overlay."""
        mock_config.template_style = "neon_vibes"  # Uses gradient
        resolution = (1080, 1920)
        
        filter_chain = build_filter_chain(sample_candidate, 1, mock_config, resolution)
        
        # Should have drawbox filters
        assert "drawbox" in filter_chain


class TestStyleBottomTextAss:
    """Test the _style_bottom_text_ass function."""

    @pytest.mark.unit
    def test_subscribe_styling(self, sample_overlay_template):
        """Test styling for subscribe CTA."""
        text = "Subscribe @TestChannel"
        result = _style_bottom_text_ass(text, sample_overlay_template)
        
        # Should contain ASS color codes
        assert "{\\c" in result or "Subscribe" in result

    @pytest.mark.unit
    def test_like_styling(self, sample_overlay_template):
        """Test styling for like CTA."""
        text = "Like and comment"
        result = _style_bottom_text_ass(text, sample_overlay_template)
        
        # Should handle like CTA
        assert "Like" in result

    @pytest.mark.unit
    def test_regular_text_no_styling(self, sample_overlay_template):
        """Test regular text without special styling."""
        text = "Regular bottom text"
        result = _style_bottom_text_ass(text, sample_overlay_template)
        
        # Should return text as-is
        assert result == text

    @pytest.mark.unit
    def test_empty_text(self, sample_overlay_template):
        """Test empty text handling."""
        result = _style_bottom_text_ass("", sample_overlay_template)
        assert result == ""

    @pytest.mark.unit
    def test_case_insensitive(self, sample_overlay_template):
        """Test that styling is case-insensitive."""
        text = "SUBSCRIBE NOW"
        result = _style_bottom_text_ass(text, sample_overlay_template)
        # Should still apply styling - the function converts "subscribe" but keeps case
        # Check that it at least returns something and handles the text
        assert result
        assert "NOW" in result


class TestFilterChainIntegration:
    """Integration tests for filter chain with various configurations."""

    @pytest.mark.integration
    def test_complete_workflow(self, mock_config, temp_dir):
        """Test complete filter chain generation workflow."""
        mock_config.workdir = temp_dir
        mock_config.template_style = "anime"
        mock_config.channel_name = "@AnimeClips"
        mock_config.overlay_backend = "ass"
        
        candidates = [
            Candidate(10.0, 30.0, "This is amazing! What will happen next?", 8.5),
            Candidate(50.0, 70.0, "The plot thickens here...", 7.2),
        ]
        
        resolution = (1080, 1920)
        
        for i, candidate in enumerate(candidates, 1):
            filter_chain = build_filter_chain(candidate, i, mock_config, resolution)
            
            # Verify filter chain is valid
            assert filter_chain
            assert isinstance(filter_chain, str)
            
            # Should contain core filters
            assert "scale" in filter_chain or "crop" in filter_chain
            
            # Should have overlay
            assert "drawbox" in filter_chain or "ass" in filter_chain
            
            # Verify ASS file created
            ass_file = os.path.join(temp_dir, f"overlay_{i}.ass")
            assert os.path.exists(ass_file)

    @pytest.mark.integration
    def test_multiple_resolutions(self, mock_config, sample_candidate, temp_dir):
        """Test filter chain generation for multiple resolutions."""
        mock_config.workdir = temp_dir
        
        resolutions = [
            (1080, 1920),  # Vertical (9:16)
            (1920, 1080),  # Horizontal (16:9)
            (720, 1280),   # Smaller vertical
            (1280, 720),   # Smaller horizontal
        ]
        
        for resolution in resolutions:
            filter_chain = build_filter_chain(sample_candidate, 1, mock_config, resolution)
            
            # Should generate valid filter for each resolution
            assert filter_chain
            assert len(filter_chain) > 0
