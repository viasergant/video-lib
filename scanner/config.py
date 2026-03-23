from __future__ import annotations

SUPPORTED_EXTENSIONS: tuple[str, ...] = (
    ".mp4",
    ".avi",
    ".mkv",
    ".mov",
    ".wmv",
    ".flv",
    ".webm",
    ".m4v",
    ".mpg",
    ".mpeg",
)

DEFAULT_OLLAMA_MODEL: str = "llava:latest"
DEFAULT_OLLAMA_HOST: str = "http://localhost:11434"

SCENE_THRESHOLD: float = 0.35
MAX_FRAMES_PER_SCENE: int = 1
FRAME_SAMPLE_INTERVAL: float = 2.0

WHISPER_MODEL: str = "base"
WHISPER_DEVICE: str = "cpu"
WHISPER_COMPUTE_TYPE: str = "int8"
