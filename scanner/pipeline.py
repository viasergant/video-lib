from __future__ import annotations

import datetime
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from scanner.analysis.scene import detect_scene_changes
from scanner.analysis.speech import transcribe_speech
from scanner.analysis.vision import LLaVAAnalyzer
from scanner.models import Event, VideoAnalysis
from scanner.output.sidecar import write_sidecar
from scanner.utils.time_format import format_time
from scanner.video.frames import extract_frames, frames_to_base64_jpeg
from scanner.video.probe import probe_video

if TYPE_CHECKING:
    import ollama
    from faster_whisper import WhisperModel

logger = logging.getLogger(__name__)

_MIN_SCENE_CHANGES: int = 3
_GRID_INTERVAL_S: float = 2.0
_MAX_GRID_FRAMES: int = 30


class VideoProcessor:
    """Orchestrates all pipeline stages to analyze a single video file.

    Args:
        ollama_client: Pre-created ``ollama.Client`` instance.
        whisper_model: Pre-loaded ``WhisperModel`` instance (shared across videos).
        model: Ollama model name to use for vision inference.
        prompt: Prompt passed to the vision model for each frame and overall.
    """

    def __init__(
        self,
        ollama_client: "ollama.Client",
        whisper_model: "WhisperModel",
        model: str,
        prompt: str,
    ) -> None:
        self._analyzer = LLaVAAnalyzer(client=ollama_client, model=model)
        self._whisper_model = whisper_model
        self._model = model
        self._prompt = prompt

    def process(self, video_path: str | Path) -> VideoAnalysis:
        """Run the full analysis pipeline for *video_path*.

        Each stage after ``probe_video`` catches its own exceptions so that
        partial results are still written to the sidecar on failure.

        Args:
            video_path: Path to the video file to analyze.

        Returns:
            A :class:`~scanner.models.VideoAnalysis` with the analysis results.

        Raises:
            RuntimeError: if ``probe_video`` fails — caller should skip the file.
        """
        video_path = Path(video_path)

        # Stage 1: probe — failure skips video entirely (exception propagates)
        meta = probe_video(video_path)

        # Stage 2: scene detection (already catches internally; returns [] on error)
        scene_timestamps = detect_scene_changes(video_path, meta.fps)

        # Stage 3: speech transcription (already catches internally; returns [], [] on error)
        transcript, speech_events = transcribe_speech(video_path, self._whisper_model)

        # Stage 4: build frame timestamps
        frame_timestamps = _build_frame_timestamps(scene_timestamps, meta.duration)

        # Stage 5: extract frames and describe each
        frames_b64: list[str] = []
        frame_events: list[Event] = []
        try:
            for ts, frame in extract_frames(video_path, frame_timestamps):
                try:
                    b64 = frames_to_base64_jpeg(frame)
                except RuntimeError:
                    logger.warning("JPEG encoding failed at %.3fs in %s", ts, video_path)
                    continue
                frames_b64.append(b64)
                description = self._analyzer.describe_frame(b64, self._prompt)
                frame_events.append(
                    Event(
                        timestamp_seconds=ts,
                        timestamp_formatted=format_time(ts),
                        type="custom",
                        description=description,
                    )
                )
        except Exception:
            logger.warning("Frame extraction failed for %s", video_path, exc_info=True)

        # Stage 6: overall video description (uses up to 5 frames internally)
        overall_description = self._analyzer.describe_video_overall(frames_b64, self._prompt)

        # Stage 7: build scene-change events from detected timestamps
        scene_events = [
            Event(
                timestamp_seconds=ts,
                timestamp_formatted=format_time(ts),
                type="scene_change",
                description="Scene change detected",
            )
            for ts in scene_timestamps
        ]

        # Stage 8: assemble VideoAnalysis — events sorted by timestamp
        all_events = sorted(
            scene_events + speech_events + frame_events,
            key=lambda e: e.timestamp_seconds,
        )
        analysis = VideoAnalysis(
            file=str(video_path),
            duration_seconds=meta.duration,
            analyzed_at=datetime.datetime.now(datetime.timezone.utc).strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            ),
            model=self._model,
            description=overall_description,
            events=all_events,
            transcript=transcript,
        )

        # Stage 9: write sidecar atomically
        write_sidecar(video_path, analysis)
        logger.info("Analysis complete for %s", video_path)
        return analysis


def _build_frame_timestamps(scene_timestamps: list[float], duration: float) -> list[float]:
    """Return the frame timestamps to extract for scene analysis.

    Uses *scene_timestamps* when 3 or more scene changes are detected.
    Falls back to a uniform grid at :data:`_GRID_INTERVAL_S`-second intervals
    (capped at :data:`_MAX_GRID_FRAMES` frames) when fewer are available.

    Args:
        scene_timestamps: Detected scene-change timestamps in seconds.
        duration: Total video duration in seconds.

    Returns:
        Sorted list of timestamps in seconds.
    """
    if len(scene_timestamps) >= _MIN_SCENE_CHANGES:
        return sorted(scene_timestamps)

    if duration <= 0:
        return []

    n_frames = min(_MAX_GRID_FRAMES, max(1, int(duration / _GRID_INTERVAL_S)))
    return [float(t) for t in np.linspace(0, duration, n_frames, endpoint=False)]
