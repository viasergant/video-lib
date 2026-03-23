# Video Library Scanner — Implementation Plan

## Overview

A Python CLI tool that scans a folder of video files, analyzes each video using local AI models (LLaVA via Ollama + faster-whisper), detects timestamped events (scene changes, speech, objects/people), and outputs one JSON sidecar file per video. Videos with an existing sidecar are skipped on re-scan.

---

## CLI Usage

```bash
python scan.py --folder /path/to/videos [--prompt "custom prompt"] [--recursive] [--model llava:latest] [--whisper-model base] [--workers 1] [--verbose]
```

---

## Project Structure

```
video-lib/
├── pyproject.toml                  # Dependencies, entry point
├── requirements.txt                # Pinned deps
├── initial-plan.md                 # This file
├── scan.py                         # Thin entry point shim
│
└── scanner/
    ├── __init__.py
    ├── cli.py                      # click CLI + orchestration
    ├── config.py                   # Constants and defaults
    ├── models.py                   # Dataclasses (VideoAnalysis, Event, TranscriptSegment)
    ├── pipeline.py                 # VideoProcessor — wires all stages
    │
    ├── video/
    │   ├── __init__.py
    │   ├── probe.py                # ffprobe wrapper (duration, fps, codec)
    │   └── frames.py               # Frame extraction + base64 encoding (OpenCV)
    │
    ├── analysis/
    │   ├── __init__.py
    │   ├── scene.py                # Scene change detection (histogram correlation)
    │   ├── vision.py               # LLaVA/Ollama integration
    │   └── speech.py               # faster-whisper transcription + audio extraction
    │
    ├── output/
    │   ├── __init__.py
    │   └── sidecar.py              # JSON sidecar read/write/skip logic
    │
    └── utils/
        ├── __init__.py
        ├── logging.py              # Logging setup
        └── time_format.py          # seconds → "HH:MM:SS"
```

---

## Output JSON Schema

Each video produces a sidecar file named `<video_filename>.json` stored next to the video:

```json
{
  "file": "video.mp4",
  "duration_seconds": 120.5,
  "analyzed_at": "2026-03-23T10:00:00Z",
  "model": "llava:latest",
  "description": "Overall description of the video",
  "events": [
    {
      "timestamp_seconds": 5.2,
      "timestamp_formatted": "00:00:05",
      "type": "scene_change | speech | object | custom",
      "description": "What happens at this moment"
    }
  ],
  "transcript": [
    { "start": 1.2, "end": 3.5, "text": "spoken words" }
  ]
}
```

---

## Processing Pipeline (per video)

```
discover_video_files(folder, recursive)
  → skip if sidecar exists
  → probe_video()                        [ffprobe: duration, fps, codec]
  → detect_scene_changes()               [OpenCV histogram correlation]
  → extract_audio() + transcribe()       [ffmpeg-python + faster-whisper]
  → build_analysis_timestamps()          [scene timestamps + fallback grid]
  → extract_frames() + describe_frame()  [OpenCV + LLaVA via Ollama]
  → describe_video_overall()             [up to 5 representative frames]
  → assemble VideoAnalysis               [events sorted by timestamp]
  → write_sidecar() atomically
```

---

## Module Details

### `scanner/models.py` — Data Contract

```python
@dataclass
class TranscriptSegment:
    start: float
    end: float
    text: str

@dataclass
class Event:
    timestamp_seconds: float
    timestamp_formatted: str   # "HH:MM:SS"
    type: Literal["scene_change", "speech", "object", "custom"]
    description: str

@dataclass
class VideoAnalysis:
    file: str
    duration_seconds: float
    analyzed_at: str           # ISO 8601 UTC
    model: str
    description: str
    events: List[Event]
    transcript: List[TranscriptSegment]

    def to_dict(self) -> dict: ...
```

### `scanner/config.py` — Constants

```python
SUPPORTED_EXTENSIONS = frozenset({".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".flv", ".wmv"})
DEFAULT_OLLAMA_MODEL = "llava:latest"
DEFAULT_OLLAMA_HOST = "http://localhost:11434"
SCENE_THRESHOLD = 0.35          # histogram correlation drop threshold
MAX_FRAMES_PER_SCENE = 1
FRAME_SAMPLE_INTERVAL = 2.0     # fallback: one frame every N seconds
WHISPER_MODEL = "base"
WHISPER_DEVICE = "cpu"
WHISPER_COMPUTE_TYPE = "int8"
```

### `scanner/video/probe.py` — Video Metadata

Uses `ffprobe -v quiet -print_format json -show_streams -show_format` via `subprocess`.

Returns `VideoMeta(path, duration, fps, width, height, codec)`.

### `scanner/video/frames.py` — Frame Extraction

- `extract_frames(video_path, timestamps, max_dimension=640)` — uses `cv2.VideoCapture`, seeks to each timestamp, resizes preserving aspect ratio, yields `(timestamp, frame_ndarray)`
- `frames_to_base64_jpeg(frame)` — encodes to JPEG bytes, returns base64 string for Ollama payload

### `scanner/analysis/scene.py` — Scene Detection

**Algorithm (OpenCV histogram correlation):**

1. Sample video at up to 5 fps (`sample_every_n_frames = max(1, int(fps / 5))`)
2. Convert each frame to HSV colorspace
3. Compute 2D histogram over H+S channels (50×60 bins) — illumination-invariant
4. Compare consecutive histograms with `cv2.HISTCMP_CORREL` (1.0 = identical)
5. Scene change if `correlation < 1 - SCENE_THRESHOLD` (i.e., below ~0.65)
6. Minimum 0.5s gap between detections to suppress flash-frame noise
7. Return sorted list of `timestamp_seconds`

### `scanner/analysis/vision.py` — LLaVA/Ollama

```python
class LLaVAAnalyzer:
    def describe_frame(self, frame_b64: str, prompt: str) -> str:
        # Single-frame analysis via ollama.Client.generate(images=[frame_b64])

    def describe_video_overall(self, sample_frames_b64: list[str], prompt: str) -> str:
        # Multi-image prompt: up to 5 evenly-spaced frames → overall description
```

Per-scene prompt: `f"{user_prompt}\nFocus on: objects visible, people and actions, text on screen, scene setting."`

Overall prompt: `f"{user_prompt}\nYou are seeing multiple frames from a video in sequence. Provide a single coherent overall description."`

### `scanner/analysis/speech.py` — Whisper Transcription

**Audio extraction:**
```python
ffmpeg.input(video_path).output(wav_path, ac=1, ar=16000, acodec="pcm_s16le", vn=None).run()
```

**Transcription:**
```python
model.transcribe(audio_path, beam_size=5, word_timestamps=True,
                 vad_filter=True,  # silero-VAD suppresses hallucinations
                 vad_parameters={"min_silence_duration_ms": 500})
```

- `WhisperModel` loaded **once** at CLI startup, shared across all videos
- Each segment also emits an `Event(type="speech")`

### `scanner/output/sidecar.py` — JSON Sidecar

- `sidecar_path(video_path)` → `Path(str(video_path) + ".json")`
- `sidecar_exists(video_path)` → bool (skip check)
- `write_sidecar()` — atomic write: write to `.json.tmp` then `os.replace()` to final path
- `read_sidecar()` → `VideoAnalysis | None`

### `scanner/pipeline.py` — Orchestrator

`VideoProcessor.process(video_path)` runs the full pipeline, catching exceptions per-stage so one bad stage (e.g. scene detection failure) produces a partial but still-written result.

### `scanner/cli.py` — CLI Entry Point

```python
@click.command()
@click.option("--folder", required=True, type=click.Path(exists=True, file_okay=False))
@click.option("--prompt", default="Describe what is happening in this video.")
@click.option("--recursive", is_flag=True)
@click.option("--model", default=DEFAULT_OLLAMA_MODEL)
@click.option("--whisper-model", default=WHISPER_MODEL)
@click.option("--workers", default=1, type=int)
@click.option("--verbose", is_flag=True)
def main(...):
    # 1. Check Ollama connectivity via client.list()
    # 2. Load WhisperModel once
    # 3. Discover video files
    # 4. Process each (skip if sidecar exists)
```

---

## Error Handling Strategy

| Stage | Failure behavior |
|-------|-----------------|
| `probe_video` | Skip video entirely, log ERROR, no sidecar written (retried next run) |
| `detect_scene_changes` | Log WARNING, continue with empty scene list |
| `transcribe` | Log WARNING, continue with empty transcript |
| `describe_frame` (LLaVA) | Log WARNING, event gets `description="[analysis unavailable]"` |
| `write_sidecar` | Log ERROR, no sidecar written (retried next run) |
| Ollama unreachable at startup | Exit immediately with clear error message |

Temp WAV files always cleaned up in `finally` via `shutil.rmtree`.

---

## Dependencies

### `pyproject.toml`

```toml
[project]
name = "video-lib-scanner"
version = "0.1.0"
requires-python = ">=3.10"

dependencies = [
    "click>=8.1",
    "opencv-python-headless>=4.8",
    "ffmpeg-python>=0.2.0",
    "faster-whisper>=1.0.0",
    "ollama>=0.3.0",
    "numpy>=1.24",
]

[project.scripts]
scan = "scanner.cli:main"

[project.optional-dependencies]
dev = ["pytest>=7", "pytest-mock", "black", "ruff", "mypy"]
```

### System Requirements (pre-installed)

- `ffmpeg` binary (with AAC/PCM support)
- Ollama daemon running with `llava:latest` pulled (`ollama pull llava:latest`)
- Python 3.10+

---

## Implementation Order

1. `pyproject.toml`
2. `scanner/models.py` — data contract (all others depend on this)
3. `scanner/config.py`
4. `scanner/utils/time_format.py`, `scanner/utils/logging.py`
5. `scanner/video/probe.py`, `scanner/video/frames.py`
6. `scanner/analysis/scene.py`, `scanner/analysis/speech.py`, `scanner/analysis/vision.py`
7. `scanner/output/sidecar.py`
8. `scanner/pipeline.py`
9. `scanner/cli.py`
10. `scan.py` (entry point shim)
11. `requirements.txt`

---

## Verification

```bash
# Install
pip install -e ".[dev]"

# Ensure Ollama is running with LLaVA
ollama pull llava:latest

# Run on a test folder
python scan.py --folder ./test_videos --recursive --verbose

# Inspect sidecar output
cat ./test_videos/sample.mp4.json
```
