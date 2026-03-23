from __future__ import annotations

import json
import subprocess
from unittest.mock import MagicMock, patch

import pytest

from scanner.video.probe import VideoMeta, probe_video

FFPROBE_OUTPUT = {
    "streams": [
        {
            "codec_type": "video",
            "codec_name": "h264",
            "width": 1920,
            "height": 1080,
            "r_frame_rate": "30/1",
        }
    ],
    "format": {
        "duration": "125.5",
    },
}


def _mock_run(output: dict):
    mock = MagicMock()
    mock.stdout = json.dumps(output)
    return mock


class TestProbeVideo:
    def test_returns_video_meta(self):
        with patch("subprocess.run", return_value=_mock_run(FFPROBE_OUTPUT)):
            meta = probe_video("video.mp4")

        assert isinstance(meta, VideoMeta)
        assert meta.path == "video.mp4"
        assert meta.duration == 125.5
        assert meta.fps == 30.0
        assert meta.width == 1920
        assert meta.height == 1080
        assert meta.codec == "h264"

    def test_path_converted_to_str(self, tmp_path):
        fake = tmp_path / "clip.mp4"
        with patch("subprocess.run", return_value=_mock_run(FFPROBE_OUTPUT)):
            meta = probe_video(fake)

        assert meta.path == str(fake)

    def test_fractional_fps(self):
        output = {**FFPROBE_OUTPUT, "streams": [{**FFPROBE_OUTPUT["streams"][0], "r_frame_rate": "30000/1001"}]}
        with patch("subprocess.run", return_value=_mock_run(output)):
            meta = probe_video("video.mp4")

        assert abs(meta.fps - 29.97) < 0.01

    def test_duration_falls_back_to_stream(self):
        output = {
            "streams": [{**FFPROBE_OUTPUT["streams"][0], "duration": "60.0"}],
            "format": {},
        }
        with patch("subprocess.run", return_value=_mock_run(output)):
            meta = probe_video("video.mp4")

        assert meta.duration == 60.0

    def test_raises_on_ffprobe_error(self):
        with patch(
            "subprocess.run",
            side_effect=subprocess.CalledProcessError(1, "ffprobe"),
        ):
            with pytest.raises(RuntimeError, match="ffprobe failed"):
                probe_video("bad.mp4")

    def test_raises_when_ffprobe_not_found(self):
        with patch("subprocess.run", side_effect=FileNotFoundError):
            with pytest.raises(RuntimeError, match="ffprobe failed"):
                probe_video("video.mp4")

    def test_raises_on_missing_video_stream(self):
        output = {"streams": [], "format": {"duration": "10.0"}}
        with patch("subprocess.run", return_value=_mock_run(output)):
            with pytest.raises(RuntimeError, match="Failed to parse"):
                probe_video("video.mp4")

    def test_raises_on_invalid_json(self):
        mock = MagicMock()
        mock.stdout = "not json"
        with patch("subprocess.run", return_value=mock):
            with pytest.raises(RuntimeError, match="Failed to parse"):
                probe_video("video.mp4")

    def test_logs_error_on_ffprobe_failure(self, caplog):
        import logging

        with patch(
            "subprocess.run",
            side_effect=subprocess.CalledProcessError(1, "ffprobe"),
        ):
            with caplog.at_level(logging.ERROR, logger="scanner.video.probe"):
                with pytest.raises(RuntimeError):
                    probe_video("bad.mp4")

        assert any("ffprobe failed" in r.message for r in caplog.records)

    def test_logs_error_on_parse_failure(self, caplog):
        import logging

        output = {"streams": [], "format": {}}
        with patch("subprocess.run", return_value=_mock_run(output)):
            with caplog.at_level(logging.ERROR, logger="scanner.video.probe"):
                with pytest.raises(RuntimeError):
                    probe_video("video.mp4")

        assert any("Failed to parse" in r.message for r in caplog.records)
