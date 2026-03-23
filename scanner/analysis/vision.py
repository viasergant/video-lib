from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import ollama

logger = logging.getLogger(__name__)

_FRAME_SUFFIX = "Focus on: objects visible, people and actions, text on screen, scene setting."
_VIDEO_SUFFIX = (
    "You are seeing multiple frames from a video in sequence. "
    "Provide a single coherent overall description."
)
_MAX_OVERALL_FRAMES: int = 5
_ANALYSIS_UNAVAILABLE: str = "[analysis unavailable]"


class LLaVAAnalyzer:
    """Describe video frames using a LLaVA vision model via Ollama.

    Args:
        client: An ``ollama.Client`` instance (injected; not created here).
        model: The Ollama model name to use for inference.
    """

    def __init__(self, client: "ollama.Client", model: str) -> None:
        self._client = client
        self._model = model

    def describe_frame(self, frame_b64: str, prompt: str) -> str:
        """Describe a single frame using the LLaVA model.

        The ``_FRAME_SUFFIX`` is appended to *prompt* before calling the model.

        Args:
            frame_b64: Base-64-encoded JPEG/PNG frame.
            prompt: Caller-supplied prompt prefix.

        Returns:
            The model's text response, or ``"[analysis unavailable]"`` on error.
        """
        full_prompt = f"{prompt} {_FRAME_SUFFIX}"
        try:
            response = self._client.generate(
                model=self._model,
                prompt=full_prompt,
                images=[frame_b64],
            )
            return response["response"]
        except Exception:
            logger.warning("Frame description failed", exc_info=True)
            return _ANALYSIS_UNAVAILABLE

    def describe_video_overall(self, sample_frames_b64: list[str], prompt: str) -> str:
        """Describe a video by sampling up to 5 evenly-spaced frames.

        The ``_VIDEO_SUFFIX`` is appended to *prompt* before calling the model.
        Up to ``_MAX_OVERALL_FRAMES`` frames are selected by evenly-spaced
        indices; all frames are used when the list is shorter.

        Args:
            sample_frames_b64: Base-64-encoded frames representing the video.
            prompt: Caller-supplied prompt prefix.

        Returns:
            The model's text response, or ``"[analysis unavailable]"`` on error.
        """
        full_prompt = f"{prompt} {_VIDEO_SUFFIX}"
        frames = _subsample(sample_frames_b64, _MAX_OVERALL_FRAMES)
        try:
            response = self._client.generate(
                model=self._model,
                prompt=full_prompt,
                images=frames,
            )
            return response["response"]
        except Exception:
            logger.warning("Overall video description failed", exc_info=True)
            return _ANALYSIS_UNAVAILABLE


def _subsample(frames: list[str], max_count: int) -> list[str]:
    """Return up to *max_count* evenly-spaced elements from *frames*.

    When ``len(frames) <= max_count`` the original list is returned unchanged.
    Indices are computed with ``round()`` so that the selection is spread as
    evenly as possible across the full range.
    """
    n = len(frames)
    if n <= max_count:
        return frames

    indices = [round(i * (n - 1) / (max_count - 1)) for i in range(max_count)]
    return [frames[i] for i in indices]
