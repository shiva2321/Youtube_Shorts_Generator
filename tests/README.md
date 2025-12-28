# Test Suite for Anime Shorts Generator

This directory contains the comprehensive pytest test suite for the Anime Shorts Generator project.

## Overview

The test suite covers:
- Core functionality (data structures, timestamp formatting, file parsing)
- Text processing (hook/punchline extraction, scoring, merging)
- FFmpeg filter chain building
- UI components
- Utility functions

## Directory Structure

```
tests/
├── __init__.py                 # Test package initialization
├── conftest.py                 # Pytest fixtures and configuration
├── test_main_functionality.py  # Core functionality tests
├── test_text_processing.py     # Text processing tests
├── test_ffmpeg_filters.py      # FFmpeg filter tests
├── test_ui.py                  # UI component tests
├── test_utils.py               # Utility function tests
└── README.md                   # This file
```

## Running Tests

### Run All Tests

```bash
pytest
```

### Run Specific Test File

```bash
pytest tests/test_main_functionality.py
```

### Run Specific Test Class

```bash
pytest tests/test_main_functionality.py::TestCandidateDataclass
```

### Run Specific Test

```bash
pytest tests/test_main_functionality.py::TestCandidateDataclass::test_candidate_creation
```

### Run with Verbose Output

```bash
pytest -v
```

### Run with Coverage Report

```bash
pytest --cov=auto_important_clips --cov=clip_generator_ui --cov-report=html
```

This generates an HTML coverage report in `htmlcov/index.html`.

## Test Markers

The test suite uses markers to categorize tests:

- `@pytest.mark.unit` - Unit tests for individual functions
- `@pytest.mark.integration` - Integration tests for multiple components
- `@pytest.mark.slow` - Tests that take more than 1 second
- `@pytest.mark.ui` - UI-related tests

### Run Only Unit Tests

```bash
pytest -m unit
```

### Run Only Integration Tests

```bash
pytest -m integration
```

### Skip Slow Tests

```bash
pytest -m "not slow"
```

### Skip UI Tests

```bash
pytest -m "not ui"
```

## Test Configuration

Test configuration is in `pytest.ini` at the project root:

- Test discovery patterns
- Output formatting
- Markers definition
- Paths to ignore

## Fixtures

Common test fixtures are defined in `conftest.py`:

- `sample_candidate` - A single Candidate object
- `sample_candidates` - List of Candidate objects
- `sample_srt_content` - Sample SRT file content
- `sample_overlay_template` - Sample OverlayTemplate
- `temp_dir` - Temporary directory for test files
- `temp_srt_file` - Temporary SRT file
- `mock_ffmpeg` - Mocked FFmpeg subprocess calls
- `mock_whisper` - Mocked Whisper model
- `mock_config` - Mock configuration object

## Mocking

The tests use extensive mocking to avoid:
- Downloading Whisper models
- Running FFmpeg commands
- Requiring actual video files
- Displaying GUI windows

Key mocked components:
- `subprocess.run` and `subprocess.check_output` for FFmpeg calls
- `whisper.load_model` for Whisper model loading
- `tkinter` components for GUI tests

## Coverage Goals

The test suite aims for:
- At least 50% overall code coverage
- High coverage of core functionality (>70%)
- Comprehensive edge case testing
- Cross-platform compatibility testing

## Writing New Tests

When adding new tests:

1. Follow the naming convention: `test_*.py`
2. Use descriptive test names: `test_<what>_<condition>`
3. Add appropriate markers (`@pytest.mark.unit`, etc.)
4. Use fixtures from `conftest.py` when possible
5. Mock external dependencies (FFmpeg, Whisper, etc.)
6. Add docstrings explaining what the test does

### Example Test

```python
@pytest.mark.unit
def test_format_timestamp_hours(self):
    """Test formatting timestamps with hours."""
    result = format_timestamp(3661.123)
    assert result == "01:01:01,123"
```

## Continuous Integration

Tests are automatically run by GitHub Actions on:
- Every push to the master branch
- Every pull request
- Python versions: 3.9, 3.10, 3.11

See `.github/workflows/python-package.yml` for CI configuration.

## Troubleshooting

### Tests Not Discovered

If pytest doesn't find tests:
- Ensure test files start with `test_`
- Ensure test functions start with `test_`
- Check that `tests/` is in `testpaths` in `pytest.ini`

### Import Errors

If you get import errors:
- Ensure you're running pytest from the project root
- Check that the project directory is in Python path
- Install test dependencies: `pip install -r requirements-dev.txt`

### Tkinter Display Errors

If UI tests fail with display errors:
- Tests are designed to avoid actually displaying windows
- If issues persist, skip UI tests: `pytest -m "not ui"`

## Dependencies

Test dependencies are in `requirements-dev.txt`:
- pytest>=7.0.0
- pytest-mock>=3.10.0
- pytest-cov>=4.0.0

Install with:
```bash
pip install -r requirements-dev.txt
```

## Contributing

When contributing tests:
1. Ensure all tests pass before submitting PR
2. Add tests for new functionality
3. Maintain or improve code coverage
4. Follow existing test patterns and style
5. Update this README if adding new test categories

## License

Same as the main project.
