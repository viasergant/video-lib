from __future__ import annotations

import json
import logging
import os
from pathlib import Path

from scanner.models import Event, TranscriptSegment, VideoAnalysis

logger = logging.getLogger(__name__)


def sidecar_path(video_path: str | Path) -> Path:
    """Return the sidecar path for a video file.

    The sidecar is stored next to the video file with a ``.json`` suffix
    appended to the full filename (e.g. ``video.mp4`` → ``video.mp4.json``).
    """
    return Path(str(video_path) + ".json")


def sidecar_exists(video_path: str | Path) -> bool:
    """Return ``True`` if a sidecar file already exists for *video_path*."""
    return sidecar_path(video_path).exists()


def write_sidecar(video_path: str | Path, analysis: VideoAnalysis) -> None:
    """Atomically write *analysis* as a JSON sidecar next to *video_path*.

    Writes to a ``.json.tmp`` file first, then uses :func:`os.replace` to
    move it into the final ``.json`` path so that a crash mid-write never
    leaves a corrupt sidecar on disk.
    """
    final = sidecar_path(video_path)
    tmp = Path(str(final) + ".tmp")
    data = json.dumps(analysis.to_dict(), indent=2, ensure_ascii=False)
    tmp.write_text(data, encoding="utf-8")
    os.replace(tmp, final)


def read_sidecar(video_path: str | Path) -> VideoAnalysis | None:
    """Read and deserialize the JSON sidecar for *video_path*.

    Returns ``None`` if the sidecar file does not exist or its contents are
    malformed (missing keys, wrong types, etc.).
    """
    path = sidecar_path(video_path)
    try:
        text = path.read_text(encoding="utf-8")
        data = json.loads(text)
        return _from_dict(data)
    except FileNotFoundError:
        return None
    except Exception:
        logger.warning("Failed to read sidecar %s", path, exc_info=True)
        return None


def _from_dict(data: dict) -> VideoAnalysis:
    """Reconstruct a :class:`~scanner.models.VideoAnalysis` from a plain dict."""
    events = [
        Event(
            timestamp_seconds=e["timestamp_seconds"],
            timestamp_formatted=e["timestamp_formatted"],
            type=e["type"],
            description=e["description"],
        )
        for e in data.get("events", [])
    ]
    transcript = [
        TranscriptSegment(
            start=s["start"],
            end=s["end"],
            text=s["text"],
        )
        for s in data.get("transcript", [])
    ]
    return VideoAnalysis(
        file=data["file"],
        duration_seconds=data["duration_seconds"],
        analyzed_at=data["analyzed_at"],
        model=data["model"],
        description=data["description"],
        events=events,
        transcript=transcript,
    )
