from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

from scanner.config import SCENE_THRESHOLD

logger = logging.getLogger(__name__)

_MIN_SCENE_GAP_S: float = 0.5
_HIST_H_BINS: int = 50
_HIST_S_BINS: int = 60


def detect_scene_changes(
    video_path: str | Path,
    fps: float,
) -> list[float]:
    """Detect scene changes using OpenCV HSV histogram correlation.

    Samples the video at up to 5 fps, computes a 2D H+S histogram for each
    frame, and records a scene change whenever the correlation between
    consecutive histograms drops below ``1 - SCENE_THRESHOLD``.

    Args:
        video_path: Path to the video file.
        fps: Frame rate of the video (used to compute sampling stride).

    Returns:
        Sorted list of scene-change timestamps in seconds.
        Returns an empty list on any error.
    """
    try:
        return _detect(str(video_path), fps)
    except Exception:
        logger.warning("Scene detection failed for %s", video_path, exc_info=True)
        return []


def _detect(video_path: str, fps: float) -> list[float]:
    sample_every_n_frames = max(1, int(fps / 5))
    correlation_threshold = 1.0 - SCENE_THRESHOLD

    cap = cv2.VideoCapture(video_path)
    try:
        scene_timestamps: list[float] = []
        prev_hist: np.ndarray | None = None
        last_scene_ts: float = -_MIN_SCENE_GAP_S  # allow detection at t=0

        frame_index = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if frame_index % sample_every_n_frames != 0:
                frame_index += 1
                continue

            timestamp = frame_index / fps
            hist = _hsv_histogram(frame)

            if prev_hist is not None:
                correlation = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL)
                if (
                    correlation < correlation_threshold
                    and (timestamp - last_scene_ts) >= _MIN_SCENE_GAP_S
                ):
                    scene_timestamps.append(timestamp)
                    last_scene_ts = timestamp

            prev_hist = hist
            frame_index += 1
    finally:
        cap.release()

    return sorted(scene_timestamps)


def _hsv_histogram(frame: np.ndarray) -> np.ndarray:
    """Compute a normalised 2D H+S histogram (illumination-invariant)."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist(
        [hsv],
        [0, 1],
        None,
        [_HIST_H_BINS, _HIST_S_BINS],
        [0, 180, 0, 256],
    )
    cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    return hist
