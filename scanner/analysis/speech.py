from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

from scanner.models import Event, TranscriptSegment
from scanner.utils.time_format import format_time

if TYPE_CHECKING:
    from faster_whisper import WhisperModel

logger = logging.getLogger(__name__)

_WHISPER_BEAM_SIZE: int = 5
_WHISPER_MIN_SILENCE_MS: int = 500


def transcribe_speech(
    video_path: str | Path,
    model: "WhisperModel",
) -> tuple[list[TranscriptSegment], list[Event]]:
    """Transcribe speech from a video file using faster-whisper.

    Audio is extracted to a temporary mono 16 kHz PCM WAV file via ffmpeg,
    transcribed, and then the temp directory is removed.

    Args:
        video_path: Path to the video file.
        model: A pre-loaded ``WhisperModel`` instance (created once at CLI
            startup and shared across all videos).

    Returns:
        A 2-tuple of:
        - ``list[TranscriptSegment]`` — one entry per speech segment.
        - ``list[Event]`` — one ``Event(type="speech")`` per segment for
          the unified timeline.

        Both lists are empty when the video contains no speech or when an
        error occurs.
    """
    try:
        return _transcribe(str(video_path), model)
    except Exception:
        logger.warning("Speech transcription failed for %s", video_path, exc_info=True)
        return [], []


def _transcribe(
    video_path: str,
    model: "WhisperModel",
) -> tuple[list[TranscriptSegment], list[Event]]:
    tmp_dir = tempfile.mkdtemp(prefix="vl_speech_")
    try:
        wav_path = str(Path(tmp_dir) / "audio.wav")
        _extract_audio(video_path, wav_path)

        segments_iter, _info = model.transcribe(
            wav_path,
            beam_size=_WHISPER_BEAM_SIZE,
            word_timestamps=True,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": _WHISPER_MIN_SILENCE_MS},
        )

        transcript: list[TranscriptSegment] = []
        events: list[Event] = []

        for seg in segments_iter:
            text = seg.text.strip()
            if not text:
                continue

            transcript.append(
                TranscriptSegment(start=seg.start, end=seg.end, text=text)
            )
            events.append(
                Event(
                    timestamp_seconds=seg.start,
                    timestamp_formatted=format_time(seg.start),
                    type="speech",
                    description=text,
                )
            )

        return transcript, events
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def _extract_audio(video_path: str, wav_path: str) -> None:
    """Run ffmpeg to extract mono 16 kHz PCM WAV audio from *video_path*."""
    cmd = [
        "ffmpeg",
        "-y",
        "-i", video_path,
        "-ac", "1",          # mono
        "-ar", "16000",      # 16 kHz sample rate
        "-f", "wav",
        wav_path,
    ]
    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg audio extraction failed (exit {result.returncode}): "
            f"{result.stderr.decode(errors='replace')}"
        )
