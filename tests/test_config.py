from scanner.config import (
    DEFAULT_OLLAMA_HOST,
    DEFAULT_OLLAMA_MODEL,
    FRAME_SAMPLE_INTERVAL,
    MAX_FRAMES_PER_SCENE,
    SCENE_THRESHOLD,
    SUPPORTED_EXTENSIONS,
    WHISPER_COMPUTE_TYPE,
    WHISPER_DEVICE,
    WHISPER_MODEL,
)


def test_supported_extensions_are_lowercase_dotted():
    for ext in SUPPORTED_EXTENSIONS:
        assert ext.startswith("."), f"{ext!r} must start with '.'"
        assert ext == ext.lower(), f"{ext!r} must be lowercase"


def test_supported_extensions_includes_common_formats():
    for fmt in (".mp4", ".avi", ".mkv", ".mov"):
        assert fmt in SUPPORTED_EXTENSIONS, f"{fmt} should be in SUPPORTED_EXTENSIONS"


def test_supported_extensions_no_duplicates():
    assert len(SUPPORTED_EXTENSIONS) == len(set(SUPPORTED_EXTENSIONS))


def test_default_ollama_model():
    assert DEFAULT_OLLAMA_MODEL == "llava:latest"


def test_default_ollama_host():
    assert DEFAULT_OLLAMA_HOST.startswith("http")


def test_scene_threshold():
    assert SCENE_THRESHOLD == 0.35
    assert 0.0 < SCENE_THRESHOLD < 1.0


def test_max_frames_per_scene():
    assert MAX_FRAMES_PER_SCENE == 1
    assert isinstance(MAX_FRAMES_PER_SCENE, int)


def test_frame_sample_interval():
    assert FRAME_SAMPLE_INTERVAL == 2.0
    assert isinstance(FRAME_SAMPLE_INTERVAL, float)


def test_whisper_model():
    assert WHISPER_MODEL == "base"


def test_whisper_device():
    assert WHISPER_DEVICE == "cpu"


def test_whisper_compute_type():
    assert WHISPER_COMPUTE_TYPE == "int8"
