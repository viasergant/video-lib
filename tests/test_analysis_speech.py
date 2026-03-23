from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

from scanner.analysis.speech import transcribe_speech
from scanner.models import Event, TranscriptSegment

# ---------------------------------------------------------------------------
# Patch targets
# ---------------------------------------------------------------------------
_PATCH_SUBPROCESS = "scanner.analysis.speech.subprocess.run"
_PATCH_MKDTEMP = "scanner.analysis.speech.tempfile.mkdtemp"
_PATCH_RMTREE = "scanner.analysis.speech.shutil.rmtree"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_segment(start: float, end: float, text: str) -> MagicMock:
    seg = MagicMock()
    seg.start = start
    seg.end = end
    seg.text = text
    return seg


def _make_model(segments: list[MagicMock]) -> MagicMock:
    """Return a WhisperModel mock whose transcribe() yields *segments*."""
    model = MagicMock()
    info = MagicMock()
    model.transcribe.return_value = (iter(segments), info)
    return model


def _ffmpeg_ok() -> MagicMock:
    """Return a subprocess.CompletedProcess mock with returncode=0."""
    result = MagicMock()
    result.returncode = 0
    return result


def _ffmpeg_fail(stderr: bytes = b"error") -> MagicMock:
    result = MagicMock()
    result.returncode = 1
    result.stderr = stderr
    return result


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestTranscribeSpeech:
    def test_returns_transcript_and_events_for_single_segment(self):
        seg = _make_segment(1.0, 3.5, " Hello world ")
        model = _make_model([seg])

        with (
            patch(_PATCH_SUBPROCESS, return_value=_ffmpeg_ok()),
            patch(_PATCH_MKDTEMP, return_value="/tmp/vl_fake"),
            patch(_PATCH_RMTREE),
        ):
            transcript, events = transcribe_speech("video.mp4", model)

        assert len(transcript) == 1
        assert transcript[0] == TranscriptSegment(start=1.0, end=3.5, text="Hello world")

        assert len(events) == 1
        assert events[0].type == "speech"
        assert events[0].timestamp_seconds == 1.0
        assert events[0].timestamp_formatted == "00:00:01"
        assert events[0].description == "Hello world"

    def test_returns_multiple_segments(self):
        segments = [
            _make_segment(0.0, 2.0, "First"),
            _make_segment(5.0, 7.0, "Second"),
        ]
        model = _make_model(segments)

        with (
            patch(_PATCH_SUBPROCESS, return_value=_ffmpeg_ok()),
            patch(_PATCH_MKDTEMP, return_value="/tmp/vl_fake"),
            patch(_PATCH_RMTREE),
        ):
            transcript, events = transcribe_speech("video.mp4", model)

        assert len(transcript) == 2
        assert len(events) == 2
        assert transcript[0].text == "First"
        assert transcript[1].text == "Second"

    def test_skips_whitespace_only_segments(self):
        segments = [
            _make_segment(0.0, 1.0, "   "),
            _make_segment(1.0, 2.0, "Real text"),
        ]
        model = _make_model(segments)

        with (
            patch(_PATCH_SUBPROCESS, return_value=_ffmpeg_ok()),
            patch(_PATCH_MKDTEMP, return_value="/tmp/vl_fake"),
            patch(_PATCH_RMTREE),
        ):
            transcript, events = transcribe_speech("video.mp4", model)

        assert len(transcript) == 1
        assert transcript[0].text == "Real text"

    def test_returns_empty_lists_when_no_segments(self):
        model = _make_model([])

        with (
            patch(_PATCH_SUBPROCESS, return_value=_ffmpeg_ok()),
            patch(_PATCH_MKDTEMP, return_value="/tmp/vl_fake"),
            patch(_PATCH_RMTREE),
        ):
            transcript, events = transcribe_speech("video.mp4", model)

        assert transcript == []
        assert events == []

    def test_cleans_up_temp_dir_on_success(self):
        model = _make_model([])

        with (
            patch(_PATCH_SUBPROCESS, return_value=_ffmpeg_ok()),
            patch(_PATCH_MKDTEMP, return_value="/tmp/vl_fake"),
            patch(_PATCH_RMTREE) as mock_rmtree,
        ):
            transcribe_speech("video.mp4", model)

        mock_rmtree.assert_called_once_with("/tmp/vl_fake", ignore_errors=True)

    def test_cleans_up_temp_dir_on_ffmpeg_failure(self):
        model = MagicMock()

        with (
            patch(_PATCH_SUBPROCESS, return_value=_ffmpeg_fail()),
            patch(_PATCH_MKDTEMP, return_value="/tmp/vl_fake"),
            patch(_PATCH_RMTREE) as mock_rmtree,
        ):
            transcript, events = transcribe_speech("video.mp4", model)

        # Graceful degradation: empty results
        assert transcript == []
        assert events == []
        # Temp dir still cleaned up
        mock_rmtree.assert_called_once_with("/tmp/vl_fake", ignore_errors=True)

    def test_cleans_up_temp_dir_on_transcription_exception(self):
        model = MagicMock()
        model.transcribe.side_effect = RuntimeError("cuda OOM")

        with (
            patch(_PATCH_SUBPROCESS, return_value=_ffmpeg_ok()),
            patch(_PATCH_MKDTEMP, return_value="/tmp/vl_fake"),
            patch(_PATCH_RMTREE) as mock_rmtree,
        ):
            transcript, events = transcribe_speech("video.mp4", model)

        assert transcript == []
        assert events == []
        mock_rmtree.assert_called_once_with("/tmp/vl_fake", ignore_errors=True)

    def test_logs_warning_on_ffmpeg_failure(self, caplog):
        model = MagicMock()

        with (
            patch(_PATCH_SUBPROCESS, return_value=_ffmpeg_fail(b"No such file")),
            patch(_PATCH_MKDTEMP, return_value="/tmp/vl_fake"),
            patch(_PATCH_RMTREE),
            caplog.at_level(logging.WARNING, logger="scanner.analysis.speech"),
        ):
            transcribe_speech("video.mp4", model)

        assert any("Speech transcription failed" in r.message for r in caplog.records)

    def test_logs_warning_on_transcription_error(self, caplog):
        model = MagicMock()
        model.transcribe.side_effect = ValueError("model broken")

        with (
            patch(_PATCH_SUBPROCESS, return_value=_ffmpeg_ok()),
            patch(_PATCH_MKDTEMP, return_value="/tmp/vl_fake"),
            patch(_PATCH_RMTREE),
            caplog.at_level(logging.WARNING, logger="scanner.analysis.speech"),
        ):
            transcribe_speech("video.mp4", model)

        assert any("Speech transcription failed" in r.message for r in caplog.records)

    def test_passes_correct_transcription_parameters(self):
        model = _make_model([])

        with (
            patch(_PATCH_SUBPROCESS, return_value=_ffmpeg_ok()),
            patch(_PATCH_MKDTEMP, return_value="/tmp/vl_fake"),
            patch(_PATCH_RMTREE),
        ):
            transcribe_speech("video.mp4", model)

        model.transcribe.assert_called_once()
        _, kwargs = model.transcribe.call_args
        assert kwargs["beam_size"] == 5
        assert kwargs["word_timestamps"] is True
        assert kwargs["vad_filter"] is True
        assert kwargs["vad_parameters"]["min_silence_duration_ms"] == 500

    def test_passes_wav_path_to_transcribe(self):
        model = _make_model([])

        with (
            patch(_PATCH_SUBPROCESS, return_value=_ffmpeg_ok()),
            patch(_PATCH_MKDTEMP, return_value="/tmp/vl_fake"),
            patch(_PATCH_RMTREE),
        ):
            transcribe_speech("video.mp4", model)

        positional_args = model.transcribe.call_args[0]
        assert positional_args[0] == "/tmp/vl_fake/audio.wav"

    def test_ffmpeg_command_uses_mono_16khz(self):
        model = _make_model([])

        with (
            patch(_PATCH_SUBPROCESS, return_value=_ffmpeg_ok()) as mock_run,
            patch(_PATCH_MKDTEMP, return_value="/tmp/vl_fake"),
            patch(_PATCH_RMTREE),
        ):
            transcribe_speech("video.mp4", model)

        cmd = mock_run.call_args[0][0]
        assert "-ac" in cmd and cmd[cmd.index("-ac") + 1] == "1"
        assert "-ar" in cmd and cmd[cmd.index("-ar") + 1] == "16000"
        assert "-f" in cmd and cmd[cmd.index("-f") + 1] == "wav"

    def test_accepts_pathlib_path(self):
        model = _make_model([])

        with (
            patch(_PATCH_SUBPROCESS, return_value=_ffmpeg_ok()) as mock_run,
            patch(_PATCH_MKDTEMP, return_value="/tmp/vl_fake"),
            patch(_PATCH_RMTREE),
        ):
            transcribe_speech(Path("/some/video.mp4"), model)

        cmd = mock_run.call_args[0][0]
        assert "/some/video.mp4" in cmd

    def test_event_timestamp_formatted_matches_start_time(self):
        seg = _make_segment(3661.0, 3665.0, "Long video speech")
        model = _make_model([seg])

        with (
            patch(_PATCH_SUBPROCESS, return_value=_ffmpeg_ok()),
            patch(_PATCH_MKDTEMP, return_value="/tmp/vl_fake"),
            patch(_PATCH_RMTREE),
        ):
            _transcript, events = transcribe_speech("video.mp4", model)

        assert events[0].timestamp_formatted == "01:01:01"

    def test_transcript_segment_text_is_stripped(self):
        seg = _make_segment(0.0, 1.0, "  padded text  ")
        model = _make_model([seg])

        with (
            patch(_PATCH_SUBPROCESS, return_value=_ffmpeg_ok()),
            patch(_PATCH_MKDTEMP, return_value="/tmp/vl_fake"),
            patch(_PATCH_RMTREE),
        ):
            transcript, _events = transcribe_speech("video.mp4", model)

        assert transcript[0].text == "padded text"
