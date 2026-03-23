from __future__ import annotations

import logging
from unittest.mock import MagicMock

import pytest

from scanner.analysis.vision import LLaVAAnalyzer, _subsample

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_MODEL = "llava:latest"


def _make_client(response_text: str = "a description") -> MagicMock:
    """Return an ollama.Client mock whose generate() returns *response_text*."""
    client = MagicMock()
    client.generate.return_value = {"response": response_text}
    return client


def _make_analyzer(client: MagicMock | None = None, model: str = _MODEL) -> LLaVAAnalyzer:
    if client is None:
        client = _make_client()
    return LLaVAAnalyzer(client=client, model=model)


# ---------------------------------------------------------------------------
# _subsample unit tests
# ---------------------------------------------------------------------------

class TestSubsample:
    def test_returns_all_frames_when_under_limit(self):
        frames = ["a", "b", "c"]
        assert _subsample(frames, 5) == ["a", "b", "c"]

    def test_returns_all_frames_when_equal_to_limit(self):
        frames = ["a", "b", "c", "d", "e"]
        assert _subsample(frames, 5) == ["a", "b", "c", "d", "e"]

    def test_returns_max_count_frames_when_over_limit(self):
        frames = [str(i) for i in range(10)]
        result = _subsample(frames, 5)
        assert len(result) == 5

    def test_first_and_last_frame_always_included(self):
        frames = [str(i) for i in range(10)]
        result = _subsample(frames, 5)
        assert result[0] == frames[0]
        assert result[-1] == frames[-1]

    def test_single_frame_list(self):
        assert _subsample(["only"], 5) == ["only"]

    def test_empty_list(self):
        assert _subsample([], 5) == []

    def test_exactly_two_frames_with_limit_five(self):
        frames = ["first", "last"]
        result = _subsample(frames, 5)
        assert result == ["first", "last"]

    def test_subsample_count_max_five_for_large_input(self):
        frames = [str(i) for i in range(100)]
        result = _subsample(frames, 5)
        assert len(result) == 5

    def test_subsample_returns_original_objects(self):
        """Items in the result must be references into the original list."""
        frames = [object() for _ in range(20)]
        result = _subsample(frames, 5)
        for item in result:
            assert item in frames


# ---------------------------------------------------------------------------
# LLaVAAnalyzer.describe_frame
# ---------------------------------------------------------------------------

class TestDescribeFrame:
    def test_returns_model_response(self):
        client = _make_client("A red car on a highway.")
        analyzer = _make_analyzer(client)

        result = analyzer.describe_frame("base64data", "Describe this frame.")

        assert result == "A red car on a highway."

    def test_appends_frame_suffix_to_prompt(self):
        client = _make_client()
        analyzer = _make_analyzer(client)

        analyzer.describe_frame("base64data", "My prompt.")

        call_kwargs = client.generate.call_args[1]
        assert "Focus on: objects visible, people and actions, text on screen, scene setting." in call_kwargs["prompt"]
        assert call_kwargs["prompt"].startswith("My prompt.")

    def test_passes_correct_model_name(self):
        client = _make_client()
        analyzer = _make_analyzer(client, model="llava:13b")

        analyzer.describe_frame("base64data", "prompt")

        assert client.generate.call_args[1]["model"] == "llava:13b"

    def test_passes_frame_in_images_list(self):
        client = _make_client()
        analyzer = _make_analyzer(client)

        analyzer.describe_frame("myframe==", "prompt")

        call_kwargs = client.generate.call_args[1]
        assert call_kwargs["images"] == ["myframe=="]

    def test_returns_analysis_unavailable_on_exception(self):
        client = MagicMock()
        client.generate.side_effect = RuntimeError("connection refused")
        analyzer = _make_analyzer(client)

        result = analyzer.describe_frame("base64data", "prompt")

        assert result == "[analysis unavailable]"

    def test_logs_warning_on_exception(self, caplog):
        client = MagicMock()
        client.generate.side_effect = ValueError("model not found")
        analyzer = _make_analyzer(client)

        with caplog.at_level(logging.WARNING, logger="scanner.analysis.vision"):
            analyzer.describe_frame("base64data", "prompt")

        assert any("Frame description failed" in r.message for r in caplog.records)

    def test_does_not_raise_on_exception(self):
        client = MagicMock()
        client.generate.side_effect = Exception("any error")
        analyzer = _make_analyzer(client)

        # Must not propagate
        result = analyzer.describe_frame("base64data", "prompt")
        assert result == "[analysis unavailable]"


# ---------------------------------------------------------------------------
# LLaVAAnalyzer.describe_video_overall
# ---------------------------------------------------------------------------

class TestDescribeVideoOverall:
    def test_returns_model_response(self):
        client = _make_client("A person cooking in a kitchen.")
        analyzer = _make_analyzer(client)

        result = analyzer.describe_video_overall(["f1", "f2"], "Describe the video.")

        assert result == "A person cooking in a kitchen."

    def test_appends_video_suffix_to_prompt(self):
        client = _make_client()
        analyzer = _make_analyzer(client)

        analyzer.describe_video_overall(["f1"], "My overall prompt.")

        call_kwargs = client.generate.call_args[1]
        assert (
            "You are seeing multiple frames from a video in sequence. "
            "Provide a single coherent overall description."
        ) in call_kwargs["prompt"]
        assert call_kwargs["prompt"].startswith("My overall prompt.")

    def test_passes_correct_model_name(self):
        client = _make_client()
        analyzer = _make_analyzer(client, model="llava:34b")

        analyzer.describe_video_overall(["f1", "f2"], "prompt")

        assert client.generate.call_args[1]["model"] == "llava:34b"

    def test_passes_all_frames_when_five_or_fewer(self):
        client = _make_client()
        analyzer = _make_analyzer(client)
        frames = ["a", "b", "c", "d", "e"]

        analyzer.describe_video_overall(frames, "prompt")

        call_kwargs = client.generate.call_args[1]
        assert call_kwargs["images"] == frames

    def test_subsamples_to_five_frames_when_more_than_five(self):
        client = _make_client()
        analyzer = _make_analyzer(client)
        frames = [f"frame{i}" for i in range(20)]

        analyzer.describe_video_overall(frames, "prompt")

        call_kwargs = client.generate.call_args[1]
        assert len(call_kwargs["images"]) == 5

    def test_subsampled_frames_are_from_original_list(self):
        client = _make_client()
        analyzer = _make_analyzer(client)
        frames = [f"frame{i}" for i in range(20)]

        analyzer.describe_video_overall(frames, "prompt")

        call_kwargs = client.generate.call_args[1]
        for img in call_kwargs["images"]:
            assert img in frames

    def test_returns_analysis_unavailable_on_exception(self):
        client = MagicMock()
        client.generate.side_effect = RuntimeError("timeout")
        analyzer = _make_analyzer(client)

        result = analyzer.describe_video_overall(["f1", "f2"], "prompt")

        assert result == "[analysis unavailable]"

    def test_logs_warning_on_exception(self, caplog):
        client = MagicMock()
        client.generate.side_effect = OSError("network error")
        analyzer = _make_analyzer(client)

        with caplog.at_level(logging.WARNING, logger="scanner.analysis.vision"):
            analyzer.describe_video_overall(["f1"], "prompt")

        assert any("Overall video description failed" in r.message for r in caplog.records)

    def test_does_not_raise_on_exception(self):
        client = MagicMock()
        client.generate.side_effect = Exception("any error")
        analyzer = _make_analyzer(client)

        result = analyzer.describe_video_overall(["f1"], "prompt")
        assert result == "[analysis unavailable]"

    def test_handles_empty_frame_list(self):
        client = _make_client("empty video")
        analyzer = _make_analyzer(client)

        result = analyzer.describe_video_overall([], "prompt")

        assert result == "empty video"
        call_kwargs = client.generate.call_args[1]
        assert call_kwargs["images"] == []

    def test_single_frame_list(self):
        client = _make_client("one frame")
        analyzer = _make_analyzer(client)

        result = analyzer.describe_video_overall(["only_frame"], "prompt")

        assert result == "one frame"
        call_kwargs = client.generate.call_args[1]
        assert call_kwargs["images"] == ["only_frame"]

    def test_first_and_last_frame_included_in_subsample(self):
        client = _make_client()
        analyzer = _make_analyzer(client)
        frames = [f"frame{i}" for i in range(10)]

        analyzer.describe_video_overall(frames, "prompt")

        call_kwargs = client.generate.call_args[1]
        images = call_kwargs["images"]
        assert images[0] == frames[0]
        assert images[-1] == frames[-1]
