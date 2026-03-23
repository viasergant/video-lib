from __future__ import annotations

import base64
import logging
from collections.abc import Generator
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def extract_frames(
    video_path: str | Path,
    timestamps: list[float],
    max_dimension: int = 640,
) -> Generator[tuple[float, np.ndarray], None, None]:
    """Seek to each timestamp and yield (timestamp, resized_frame).

    Frames are resized so the longest side is at most *max_dimension*,
    preserving the original aspect ratio.

    Args:
        video_path: Path to the video file.
        timestamps: Timestamps in seconds to extract.
        max_dimension: Maximum size (px) of the longest side after resize.

    Yields:
        (timestamp_seconds, frame_ndarray) pairs in the order requested.
    """
    cap = cv2.VideoCapture(str(video_path))
    try:
        for ts in timestamps:
            cap.set(cv2.CAP_PROP_POS_MSEC, ts * 1000)
            ok, frame = cap.read()
            if not ok:
                logger.warning("Could not read frame at %.3fs from %s", ts, video_path)
                continue

            h, w = frame.shape[:2]
            longest = max(h, w)
            if longest > max_dimension:
                scale = max_dimension / longest
                new_w = int(w * scale)
                new_h = int(h * scale)
                frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

            yield ts, frame
    finally:
        cap.release()


def frames_to_base64_jpeg(frame: np.ndarray) -> str:
    """Encode an ndarray frame as a base64 JPEG string for Ollama payloads.

    Args:
        frame: BGR ndarray as returned by cv2.

    Returns:
        Base64-encoded JPEG string.

    Raises:
        RuntimeError: if JPEG encoding fails.
    """
    ok, buf = cv2.imencode(".jpg", frame)
    if not ok:
        raise RuntimeError("cv2.imencode failed to encode frame as JPEG")
    return base64.b64encode(buf.tobytes()).decode("utf-8")
