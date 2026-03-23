from __future__ import annotations

import json
import logging
import subprocess
from fractions import Fraction
from pathlib import Path

from scanner.models import VideoMeta

logger = logging.getLogger(__name__)


def probe_video(path: str | Path) -> VideoMeta:
    """Run ffprobe on *path* and return a VideoMeta.

    Raises:
        RuntimeError: if ffprobe fails or the output cannot be parsed.
            Callers should catch this, log at ERROR, and skip the file.
    """
    path = str(path)
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v", "quiet",
                "-print_format", "json",
                "-show_streams",
                "-show_format",
                path,
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        )
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired) as exc:
        logger.error("ffprobe failed for %s: %s", path, exc)
        raise RuntimeError(f"ffprobe failed for {path}") from exc

    try:
        data = json.loads(result.stdout)
        fmt = data.get("format", {})
        video_stream = next(
            s for s in data.get("streams", []) if s.get("codec_type") == "video"
        )

        raw_duration = fmt.get("duration") or video_stream.get("duration")
        if raw_duration is None:
            raise ValueError("duration not found in format or video stream")
        duration = float(raw_duration)
        fps = float(Fraction(video_stream["r_frame_rate"]))
        width = int(video_stream["width"])
        height = int(video_stream["height"])
        codec = video_stream["codec_name"]
    except (KeyError, StopIteration, ValueError, json.JSONDecodeError) as exc:
        logger.error("Failed to parse ffprobe output for %s: %s", path, exc)
        raise RuntimeError(f"Failed to parse ffprobe output for {path}") from exc

    return VideoMeta(
        path=path,
        duration=duration,
        fps=fps,
        width=width,
        height=height,
        codec=codec,
    )
