from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from scanner.analysis.scene import detect_scene_changes

_PATCH_CAP = "scanner.analysis.scene.cv2.VideoCapture"
_PATCH_COMPARE = "scanner.analysis.scene.cv2.compareHist"
_PATCH_CALC_HIST = "scanner.analysis.scene.cv2.calcHist"
_PATCH_NORMALIZE = "scanner.analysis.scene.cv2.normalize"
_PATCH_CVT = "scanner.analysis.scene.cv2.cvtColor"


def _make_frame(h: int = 480, w: int = 640) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)


def _make_cap(frames: list[np.ndarray | None]) -> MagicMock:
    """Build a mock VideoCapture that reads the given frames in order.

    None entries produce (False, None); ndarray entries produce (True, frame).
    """
    cap = MagicMock()
    side_effects = [(f is not None, f) for f in frames] + [(False, None)]
    cap.read.side_effect = side_effects
    return cap


def _stub_hist_pipeline(
    calc_hist_mock: MagicMock,
    normalize_mock: MagicMock,
    cvt_mock: MagicMock,
    hist_value: np.ndarray | None = None,
) -> np.ndarray:
    """Wire the cv2 histogram mocks to return *hist_value* (or zeros)."""
    h = hist_value if hist_value is not None else np.zeros((50, 60), dtype=np.float32)
    calc_hist_mock.return_value = h
    normalize_mock.side_effect = None  # no-op
    cvt_mock.return_value = _make_frame()
    return h


class TestDetectSceneChanges:
    def test_returns_empty_list_for_single_frame_video(self):
        cap = _make_cap([_make_frame()])

        with (
            patch(_PATCH_CAP, return_value=cap),
            patch(_PATCH_CVT, return_value=_make_frame()),
            patch(_PATCH_CALC_HIST, return_value=np.zeros((50, 60), np.float32)),
            patch(_PATCH_NORMALIZE),
        ):
            result = detect_scene_changes("video.mp4", fps=25.0)

        assert result == []

    def test_detects_single_scene_change(self):
        frames = [_make_frame(), _make_frame()]
        cap = _make_cap(frames)
        h1 = np.ones((50, 60), dtype=np.float32)
        h2 = np.zeros((50, 60), dtype=np.float32)

        with (
            patch(_PATCH_CAP, return_value=cap),
            patch(_PATCH_CVT, return_value=_make_frame()),
            patch(_PATCH_CALC_HIST, side_effect=[h1, h2]),
            patch(_PATCH_NORMALIZE),
            # HISTCMP_CORREL returns 0.0 → well below threshold (0.65)
            patch(_PATCH_COMPARE, return_value=0.0),
        ):
            result = detect_scene_changes("video.mp4", fps=1.0)

        assert len(result) == 1
        assert result[0] == pytest.approx(1.0)

    def test_no_scene_change_when_correlation_above_threshold(self):
        frames = [_make_frame(), _make_frame()]
        cap = _make_cap(frames)
        h = np.ones((50, 60), dtype=np.float32)

        with (
            patch(_PATCH_CAP, return_value=cap),
            patch(_PATCH_CVT, return_value=_make_frame()),
            patch(_PATCH_CALC_HIST, return_value=h),
            patch(_PATCH_NORMALIZE),
            # HISTCMP_CORREL returns 1.0 → identical frames, no change
            patch(_PATCH_COMPARE, return_value=1.0),
        ):
            result = detect_scene_changes("video.mp4", fps=1.0)

        assert result == []

    def test_suppresses_flash_within_min_gap(self):
        """Two changes less than 0.5 s apart — only the first is recorded."""
        frames = [_make_frame(), _make_frame(), _make_frame()]
        cap = _make_cap(frames)
        h = np.ones((50, 60), dtype=np.float32)

        with (
            patch(_PATCH_CAP, return_value=cap),
            patch(_PATCH_CVT, return_value=_make_frame()),
            patch(_PATCH_CALC_HIST, return_value=h),
            patch(_PATCH_NORMALIZE),
            # Low correlation on both consecutive pairs
            patch(_PATCH_COMPARE, return_value=0.0),
        ):
            # fps=5 → frame timestamps 0.0, 0.2, 0.4 — gap is only 0.2 s
            result = detect_scene_changes("video.mp4", fps=5.0)

        assert len(result) == 1
        assert result[0] == pytest.approx(0.2)

    def test_accepts_two_changes_separated_by_min_gap(self):
        """Two changes exactly 0.5 s apart — both recorded."""
        frames = [_make_frame(), _make_frame(), _make_frame()]
        cap = _make_cap(frames)
        h = np.ones((50, 60), dtype=np.float32)

        with (
            patch(_PATCH_CAP, return_value=cap),
            patch(_PATCH_CVT, return_value=_make_frame()),
            patch(_PATCH_CALC_HIST, return_value=h),
            patch(_PATCH_NORMALIZE),
            patch(_PATCH_COMPARE, return_value=0.0),
        ):
            # fps=2 → timestamps 0.0, 0.5, 1.0 — gap is exactly 0.5 s
            result = detect_scene_changes("video.mp4", fps=2.0)

        assert len(result) == 2
        assert result[0] == pytest.approx(0.5)
        assert result[1] == pytest.approx(1.0)

    def test_samples_at_up_to_5_fps(self):
        """At fps=25 only every 5th frame is sampled."""
        # 10 frames → 2 sampled (index 0 and 5)
        frames = [_make_frame()] * 10
        cap = _make_cap(frames)
        h = np.ones((50, 60), dtype=np.float32)

        compare_calls: list[float] = []

        def _cmp(*_args, **_kwargs) -> float:
            compare_calls.append(1.0)
            return 1.0

        with (
            patch(_PATCH_CAP, return_value=cap),
            patch(_PATCH_CVT, return_value=_make_frame()),
            patch(_PATCH_CALC_HIST, return_value=h),
            patch(_PATCH_NORMALIZE),
            patch(_PATCH_COMPARE, side_effect=_cmp),
        ):
            detect_scene_changes("video.mp4", fps=25.0)

        # 10 frames at stride 5 → sampled indices 0, 5 → 1 comparison
        assert len(compare_calls) == 1

    def test_returns_sorted_timestamps(self):
        """Result is sorted even when detections happen out of order internally."""
        frames = [_make_frame()] * 5
        cap = _make_cap(frames)
        h = np.ones((50, 60), dtype=np.float32)
        # Alternating low/high correlation to produce multiple detections
        correlations = [0.0, 1.0, 0.0, 1.0]

        with (
            patch(_PATCH_CAP, return_value=cap),
            patch(_PATCH_CVT, return_value=_make_frame()),
            patch(_PATCH_CALC_HIST, return_value=h),
            patch(_PATCH_NORMALIZE),
            patch(_PATCH_COMPARE, side_effect=correlations),
        ):
            result = detect_scene_changes("video.mp4", fps=1.0)

        assert result == sorted(result)

    def test_returns_empty_list_on_exception(self, caplog):
        with (
            patch(_PATCH_CAP, side_effect=RuntimeError("disk failure")),
            caplog.at_level(logging.WARNING, logger="scanner.analysis.scene"),
        ):
            result = detect_scene_changes("bad.mp4", fps=25.0)

        assert result == []
        assert any("Scene detection failed" in r.message for r in caplog.records)

    def test_releases_capture_after_iteration(self):
        cap = _make_cap([_make_frame()])

        with (
            patch(_PATCH_CAP, return_value=cap),
            patch(_PATCH_CVT, return_value=_make_frame()),
            patch(_PATCH_CALC_HIST, return_value=np.zeros((50, 60), np.float32)),
            patch(_PATCH_NORMALIZE),
        ):
            detect_scene_changes("video.mp4", fps=25.0)

        cap.release.assert_called_once()

    def test_accepts_pathlib_path(self):
        from pathlib import Path

        cap = _make_cap([_make_frame()])

        with (
            patch(_PATCH_CAP, return_value=cap) as mock_vc,
            patch(_PATCH_CVT, return_value=_make_frame()),
            patch(_PATCH_CALC_HIST, return_value=np.zeros((50, 60), np.float32)),
            patch(_PATCH_NORMALIZE),
        ):
            detect_scene_changes(Path("/some/video.mp4"), fps=25.0)

        mock_vc.assert_called_once_with("/some/video.mp4")
