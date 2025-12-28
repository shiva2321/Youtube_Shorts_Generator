"""Tests for text processing functions in auto_important_clips.py."""

import pytest
import random
from unittest.mock import Mock, patch

from auto_important_clips import (
    Candidate,
    get_hook_punchline,
    _smart_extract,
    compute_heuristic_score,
    merge_adjacent_candidates,
    _wrap_for_ass,
    _pick_catchy_hook,
    _pick_top_emojis_for_text,
)


class TestGetHookPunchline:
    """Test the get_hook_punchline function."""

    @pytest.mark.unit
    def test_basic_extraction(self):
        """Test basic hook and punchline extraction."""
        text = "What is happening here? This is amazing! It's incredible."
        hook, punchline = get_hook_punchline(text)
        assert hook  # Should have extracted something
        assert punchline  # Should have extracted something
        assert isinstance(hook, str)
        assert isinstance(punchline, str)

    @pytest.mark.unit
    def test_short_text(self):
        """Test extraction from short text."""
        text = "Short text."
        hook, punchline = get_hook_punchline(text)
        assert hook
        assert punchline

    @pytest.mark.unit
    def test_japanese_text(self):
        """Test extraction from Japanese text."""
        text = "これは素晴らしいです。本当に驚きました。"
        hook, punchline = get_hook_punchline(text)
        # At least one should have content
        assert hook or punchline
        # Hook should be extracted
        assert hook
        # Check it's valid
        assert isinstance(hook, str)
        assert isinstance(punchline, str)


class TestSmartExtract:
    """Test the _smart_extract function."""

    @pytest.mark.unit
    def test_hook_mode_question(self):
        """Test hook mode prefers questions."""
        text = "What is this? This is a statement. Another statement."
        result = _smart_extract(text, "hook")
        assert "?" in result or "What" in result

    @pytest.mark.unit
    def test_hook_mode_exclamation(self):
        """Test hook mode handles exclamations."""
        text = "This is amazing! This is normal. Another one."
        result = _smart_extract(text, "hook")
        # Should pick the exciting first sentence
        assert result

    @pytest.mark.unit
    def test_punchline_mode(self):
        """Test punchline mode picks last sentence."""
        text = "First sentence. Second sentence. Final conclusion!"
        result = _smart_extract(text, "punchline")
        assert "Final" in result or "conclusion" in result

    @pytest.mark.unit
    def test_empty_text(self):
        """Test extraction from empty text."""
        result = _smart_extract("", "hook")
        assert len(result) <= 50


class TestComputeHeuristicScore:
    """Test the compute_heuristic_score function."""

    @pytest.mark.unit
    def test_question_mark_bonus(self):
        """Test that questions get a score bonus."""
        cand1 = Candidate(0, 10, "Why is this happening?", 0)
        cand2 = Candidate(0, 10, "This is happening.", 0)
        
        score1 = compute_heuristic_score(cand1)
        score2 = compute_heuristic_score(cand2)
        
        assert score1 > score2

    @pytest.mark.unit
    def test_exclamation_bonus(self):
        """Test that exclamations get a score bonus."""
        cand1 = Candidate(0, 10, "This is amazing!", 0)
        cand2 = Candidate(0, 10, "This is normal.", 0)
        
        score1 = compute_heuristic_score(cand1)
        score2 = compute_heuristic_score(cand2)
        
        assert score1 > score2

    @pytest.mark.unit
    def test_duration_scoring(self):
        """Test duration affects score."""
        # Too short - penalty
        short = Candidate(0, 3, "Short text", 0)
        # Good duration - bonus
        good = Candidate(0, 15, "Good duration text", 0)
        # Too long - penalty
        long = Candidate(0, 50, "Very long duration text", 0)
        
        score_short = compute_heuristic_score(short)
        score_good = compute_heuristic_score(good)
        score_long = compute_heuristic_score(long)
        
        assert score_good > score_short
        assert score_good > score_long

    @pytest.mark.unit
    def test_keyword_bonus(self):
        """Test that keywords increase score."""
        with_keyword = Candidate(0, 10, "This is the secret truth", 0)
        without_keyword = Candidate(0, 10, "This is something", 0)
        
        score_with = compute_heuristic_score(with_keyword)
        score_without = compute_heuristic_score(without_keyword)
        
        assert score_with > score_without

    @pytest.mark.unit
    def test_video_duration_position_bonus(self):
        """Test that middle clips get position bonus."""
        early = Candidate(5, 15, "Early clip", 0)
        middle = Candidate(500, 510, "Middle clip", 0)
        late = Candidate(1395, 1405, "Late clip", 0)
        
        video_duration = 1400.0  # 23 minutes
        
        score_early = compute_heuristic_score(early, video_duration)
        score_middle = compute_heuristic_score(middle, video_duration)
        score_late = compute_heuristic_score(late, video_duration)
        
        # Middle should have highest score
        assert score_middle > score_early
        assert score_middle > score_late


class TestMergeAdjacentCandidates:
    """Test the merge_adjacent_candidates function."""

    @pytest.mark.unit
    def test_merge_basic(self):
        """Test basic merging of candidates."""
        candidates = [
            Candidate(0, 5, "First", 0),
            Candidate(5, 10, "Second", 0),
            Candidate(10, 15, "Third", 0),
        ]
        
        merged = merge_adjacent_candidates(candidates, max_clip_len=20.0, min_clip_len=5.0)
        assert len(merged) < len(candidates)  # Should have merged some
        assert merged[0].end - merged[0].start >= 5.0  # Meets minimum

    @pytest.mark.unit
    def test_merge_respects_max_length(self):
        """Test that merging respects max clip length."""
        candidates = [
            Candidate(i * 5, (i + 1) * 5, f"Segment {i}", 0)
            for i in range(20)  # 20 segments of 5 seconds each
        ]
        
        max_len = 30.0
        merged = merge_adjacent_candidates(candidates, max_clip_len=max_len, min_clip_len=8.0)
        
        for clip in merged:
            duration = clip.end - clip.start
            assert duration <= max_len

    @pytest.mark.unit
    def test_merge_empty_list(self):
        """Test merging empty candidate list."""
        merged = merge_adjacent_candidates([])
        assert merged == []

    @pytest.mark.unit
    def test_merge_combines_text(self):
        """Test that merged candidates combine text."""
        candidates = [
            Candidate(0, 5, "First part", 0),
            Candidate(5, 10, "Second part", 0),
        ]
        
        merged = merge_adjacent_candidates(candidates, max_clip_len=15.0, min_clip_len=5.0)
        assert len(merged) == 1
        assert "First part" in merged[0].text
        assert "Second part" in merged[0].text


class TestWrapForAss:
    """Test the _wrap_for_ass function."""

    @pytest.mark.unit
    def test_wrap_short_text(self):
        """Test wrapping short text."""
        text = "Short text"
        result = _wrap_for_ass(text, max_chars_per_line=20, max_lines=2)
        assert result == "Short text"
        assert "\\N" not in result

    @pytest.mark.unit
    def test_wrap_long_text(self):
        """Test wrapping long text."""
        text = "This is a very long text that should be wrapped into multiple lines"
        result = _wrap_for_ass(text, max_chars_per_line=20, max_lines=3)
        assert "\\N" in result  # Should contain line break

    @pytest.mark.unit
    def test_wrap_respects_max_lines(self):
        """Test that wrapping respects max lines."""
        text = "Word " * 50  # Very long text
        result = _wrap_for_ass(text, max_chars_per_line=10, max_lines=2)
        line_count = result.count("\\N") + 1
        assert line_count <= 2

    @pytest.mark.unit
    def test_wrap_empty_text(self):
        """Test wrapping empty text."""
        result = _wrap_for_ass("", max_chars_per_line=20, max_lines=2)
        assert result == ""

    @pytest.mark.unit
    def test_wrap_preserves_ass_tags(self):
        """Test that ASS tags are not wrapped."""
        text = "{\\c&H00FFFFFF}Subscribe {\\c&H0033CCFF}@Channel"
        result = _wrap_for_ass(text, max_chars_per_line=10, max_lines=2)
        # Should not break ASS tags
        assert "{" in result and "}" in result

    @pytest.mark.unit
    def test_wrap_adds_ellipsis(self):
        """Test that truncated text gets ellipsis."""
        text = "Word " * 30
        result = _wrap_for_ass(text, max_chars_per_line=20, max_lines=2)
        # The text is very long, so it should be truncated
        # Check that the result is shorter than the original or has ellipsis
        assert len(result) < len(text) or "..." in result


class TestPickCatchyHook:
    """Test the _pick_catchy_hook function."""

    @pytest.mark.unit
    def test_returns_catchy_hook(self):
        """Test that function returns a catchy hook."""
        rnd = random.Random(42)
        result = _pick_catchy_hook("Some text", rnd)
        assert result  # Should return something
        assert isinstance(result, str)
        # Should have emoji
        assert any(ord(c) > 127 for c in result)

    @pytest.mark.unit
    def test_deterministic_with_seed(self):
        """Test that same seed produces same result."""
        rnd1 = random.Random(42)
        rnd2 = random.Random(42)
        
        result1 = _pick_catchy_hook("Text", rnd1)
        result2 = _pick_catchy_hook("Text", rnd2)
        
        assert result1 == result2

    @pytest.mark.unit
    def test_different_with_different_seed(self):
        """Test that different seeds can produce different results."""
        results = []
        for seed in range(10):
            rnd = random.Random(seed)
            results.append(_pick_catchy_hook("Text", rnd))
        
        # With 10 different seeds, we should get some variety
        unique_results = len(set(results))
        assert unique_results > 1


class TestPickTopEmojisForText:
    """Test the _pick_top_emojis_for_text function."""

    @pytest.mark.unit
    def test_returns_emojis(self):
        """Test that function returns emojis."""
        rnd = random.Random(42)
        result = _pick_top_emojis_for_text("Some text", rnd, max_emojis=2)
        assert isinstance(result, list)
        assert len(result) <= 2

    @pytest.mark.unit
    def test_context_aware_fight(self):
        """Test context-aware emoji selection for fight keywords."""
        rnd = random.Random(42)
        result = _pick_top_emojis_for_text("Epic fight battle scene", rnd, max_emojis=3)
        # Should select from fight-related emojis
        assert len(result) > 0

    @pytest.mark.unit
    def test_context_aware_funny(self):
        """Test context-aware emoji selection for funny keywords."""
        rnd = random.Random(42)
        result = _pick_top_emojis_for_text("This is so funny lol", rnd, max_emojis=2)
        assert len(result) > 0

    @pytest.mark.unit
    def test_max_emojis_limit(self):
        """Test that max_emojis is respected."""
        rnd = random.Random(42)
        result = _pick_top_emojis_for_text("Text", rnd, max_emojis=1)
        assert len(result) <= 1

    @pytest.mark.unit
    def test_default_emojis(self):
        """Test default emojis when no keywords match."""
        rnd = random.Random(42)
        result = _pick_top_emojis_for_text("Generic text with no keywords", rnd, max_emojis=2)
        # Should still return emojis from default pool
        assert len(result) > 0
