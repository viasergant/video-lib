from __future__ import annotations

import base64
from unittest.mock import MagicMock, call, patch

import numpy as np
import pytest

from scanner.video.frames import extract_frames, frames_to_base64_jpeg

_PATCH_CAP = "scanner.video.frames.cv2.VideoCapture"
_PATCH_RESIZE = "scanner.video.frames.cv2.resize"
_PATCH_IMENCODE = "scanner.video.frames.cv2.imencode"


def _make_frame(h: int = 480, w: int = 640) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)


def _make_cap(frames: list[tuple[bool, np.ndarray | None]]) -> MagicMock:
    """Build a mock VideoCapture whose read() returns each (ok, frame) in turn."""
    cap = MagicMock()
    cap.read.side_effect = frames
    return cap


class TestExtractFrames:
    def test_yields_timestamp_and_frame(self):
        frame = _make_frame(240, 320)
        cap = _make_cap([(True, frame)])

        with patch(_PATCH_CAP, return_value=cap):
            results = list(extract_frames("video.mp4", [1.0]))

        assert len(results) == 1
        ts, f = results[0]
        assert ts == 1.0
        assert f is frame  # no resize needed: longest side 320 < 640

    def test_seeks_to_correct_milliseconds(self):
        frame = _make_frame(240, 320)
        cap = _make_cap([(True, frame), (True, frame)])

        with patch(_PATCH_CAP, return_value=cap):
            list(extract_frames("video.mp4", [1.5, 3.0]))

        import cv2
        cap.set.assert_any_call(cv2.CAP_PROP_POS_MSEC, 1500.0)
        cap.set.assert_any_call(cv2.CAP_PROP_POS_MSEC, 3000.0)

    def test_resizes_when_longest_side_exceeds_max_dimension(self):
        frame = _make_frame(1080, 1920)  # longest side 1920 > 640
        cap = _make_cap([(True, frame)])
        small_frame = _make_frame(180, 320)

        with patch(_PATCH_CAP, return_value=cap), \
                patch(_PATCH_RESIZE, return_value=small_frame) as mock_resize:
            results = list(extract_frames("video.mp4", [0.0]))

        mock_resize.assert_called_once()
        _, f = results[0]
        assert f is small_frame

    def test_resize_preserves_aspect_ratio(self):
        frame = _make_frame(1080, 1920)  # longest side 1920
        cap = _make_cap([(True, frame)])

        import cv2 as _cv2

        with patch(_PATCH_CAP, return_value=cap):
            with patch(_PATCH_RESIZE, wraps=_cv2.resize) as mock_resize:
                list(extract_frames("video.mp4", [0.0], max_dimension=640))

        # scale = 640/1920 ≈ 0.333 → new_w=640, new_h=360
        _, (new_w, new_h), *_ = mock_resize.call_args[0]
        assert new_w == 640
        assert new_h == 360

    def test_no_resize_when_within_max_dimension(self):
        frame = _make_frame(480, 640)
        cap = _make_cap([(True, frame)])

        with patch(_PATCH_CAP, return_value=cap), \
                patch(_PATCH_RESIZE) as mock_resize:
            list(extract_frames("video.mp4", [0.0], max_dimension=640))

        mock_resize.assert_not_called()

    def test_skips_unreadable_frame_with_warning(self, caplog):
        import logging
        frame = _make_frame()
        cap = _make_cap([(False, None), (True, frame)])

        with patch(_PATCH_CAP, return_value=cap):
            with caplog.at_level(logging.WARNING, logger="scanner.video.frames"):
                results = list(extract_frames("video.mp4", [0.5, 1.0]))

        assert len(results) == 1
        assert results[0][0] == 1.0
        assert any("Could not read frame" in r.message for r in caplog.records)

    def test_releases_capture_after_iteration(self):
        frame = _make_frame()
        cap = _make_cap([(True, frame)])

        with patch(_PATCH_CAP, return_value=cap):
            list(extract_frames("video.mp4", [0.0]))

        cap.release.assert_called_once()

    def test_releases_capture_on_exception(self):
        cap = MagicMock()
        cap.read.side_effect = RuntimeError("boom")

        with patch(_PATCH_CAP, return_value=cap):
            with pytest.raises(RuntimeError):
                list(extract_frames("video.mp4", [0.0]))

        cap.release.assert_called_once()

    def test_empty_timestamps_yields_nothing(self):
        cap = MagicMock()

        with patch(_PATCH_CAP, return_value=cap):
            results = list(extract_frames("video.mp4", []))

        cap.read.assert_not_called()
        assert results == []

    def test_accepts_pathlib_path(self):
        from pathlib import Path

        frame = _make_frame()
        cap = _make_cap([(True, frame)])

        with patch(_PATCH_CAP, return_value=cap) as mock_vc:
            list(extract_frames(Path("/some/video.mp4"), [0.0]))

        mock_vc.assert_called_once_with("/some/video.mp4")


class TestFramesToBase64Jpeg:
    def test_returns_valid_base64_string(self):
        frame = _make_frame()
        buf = np.array([0xFF, 0xD8, 0xFF], dtype=np.uint8)  # minimal JPEG-like bytes

        with patch(_PATCH_IMENCODE, return_value=(True, buf)):
            result = frames_to_base64_jpeg(frame)

        assert isinstance(result, str)
        decoded = base64.b64decode(result)
        assert decoded == buf.tobytes()

    def test_raises_on_encode_failure(self):
        frame = _make_frame()

        with patch(_PATCH_IMENCODE, return_value=(False, None)):
            with pytest.raises(RuntimeError, match="cv2.imencode failed"):
                frames_to_base64_jpeg(frame)

    def test_encodes_as_jpeg(self):
        frame = _make_frame()
        buf = np.zeros(10, dtype=np.uint8)

        with patch(_PATCH_IMENCODE, return_value=(True, buf)) as mock_enc:
            frames_to_base64_jpeg(frame)

        mock_enc.assert_called_once_with(".jpg", frame)
