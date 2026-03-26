# video-lib

A command-line tool that scans a folder of video files and generates a JSON sidecar for each one. Each sidecar contains an overall description, a per-event timeline, and a speech transcript produced by combining a LLaVA vision model (via Ollama) with faster-whisper.

---

## Prerequisites

| Requirement | Version | Notes |
|-------------|---------|-------|
| Python | >= 3.10 | |
| ffmpeg | any recent | Must be on `PATH`; used for audio extraction and frame probing |
| Ollama | any recent | Must be running at `http://localhost:11434` before invoking the scanner |
| llava model | any tag | Pull with `ollama pull llava:latest` |

Verify your setup:

```bash
ffmpeg -version
ollama list        # should show llava:latest (or your chosen model)
```

---

## Installation

### Linux (Ubuntu 22.04 LTS)

Run the automated installer (requires `sudo`):

```bash
sudo bash install.sh
```

This installs Python 3.10+, ffmpeg, Ollama, pulls the `llava:latest` model, creates a `.venv` virtualenv, and installs all Python dependencies. Safe to run multiple times.

After installation:

```bash
source .venv/bin/activate
scan --folder /path/to/videos
```

### Basic (run-only)

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install .
```

### Development (includes linters, type-checker, and test runner)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

The `scan` console script is registered in `pyproject.toml` and is available after either install. Alternatively, run `python scan.py` directly from the repo root without installing.

---

## CLI Usage

```
scan --folder PATH [OPTIONS]
```

All flags:

| Flag | Default | Description |
|------|---------|-------------|
| `--folder PATH` | (required) | Directory to scan for video files |
| `--prompt TEXT` | `"Describe what is happening in this video."` | Prompt passed to the vision model for each frame and the overall description |
| `--recursive` | off | Recurse into sub-directories when discovering video files |
| `--model TEXT` | `llava:latest` | Ollama model name to use for vision inference |
| `--whisper-model TEXT` | `base` | faster-whisper model size (`tiny`, `base`, `small`, `medium`, `large`) |
| `--workers INT` | `1` | Number of parallel video-processing workers |
| `--verbose` | off | Enable DEBUG-level logging |

### Examples

Scan a single directory:

```bash
scan --folder /mnt/videos
```

Scan recursively with a custom prompt and a larger Whisper model:

```bash
scan --folder /mnt/videos --recursive \
     --prompt "List every object you see." \
     --whisper-model small
```

Use a different Ollama model and two parallel workers:

```bash
scan --folder /mnt/videos --model llava:13b --workers 2
```

---

## Output Format

For each video file `<name>.mp4` the scanner writes `<name>.mp4.json` in the same directory.

### JSON schema

```json
{
  "file": "/absolute/path/to/video.mp4",
  "duration_seconds": 142.5,
  "analyzed_at": "2024-06-01T12:00:00Z",
  "model": "llava:latest",
  "description": "A person walks through a park and sits on a bench.",
  "events": [
    {
      "timestamp_seconds": 0.0,
      "timestamp_formatted": "00:00:00",
      "type": "scene_change",
      "description": "Wide shot of an empty park path."
    },
    {
      "timestamp_seconds": 18.4,
      "timestamp_formatted": "00:00:18",
      "type": "speech",
      "description": "Hello, is anyone here?"
    }
  ],
  "transcript": [
    {
      "start": 18.4,
      "end": 20.1,
      "text": "Hello, is anyone here?"
    }
  ]
}
```

`type` in an event is one of: `scene_change`, `speech`, `object`, `custom`.

### Idempotency

The scanner checks for an existing sidecar before processing each video. If `<name>.mp4.json` already exists the file is skipped and counted as `skipped` in the final summary line. Re-running the scanner against the same folder is therefore safe and cheap.

---

## Project Structure

```
video-lib/
├── scan.py                        # Entry-point shim (python scan.py)
├── pyproject.toml                 # Build config, dependencies, scripts
├── requirements.txt               # Pinned full dependency lockfile
├── scanner/
│   ├── cli.py                     # Click CLI — argument parsing, orchestration
│   ├── config.py                  # Constants: extensions, model names, thresholds
│   ├── models.py                  # Dataclasses: VideoAnalysis, Event, TranscriptSegment
│   ├── pipeline.py                # VideoProcessor — ties all stages together
│   ├── analysis/
│   │   ├── scene.py               # Scene-change detection via OpenCV
│   │   ├── speech.py              # Speech transcription via faster-whisper + ffmpeg
│   │   └── vision.py              # Frame/video description via LLaVA (Ollama)
│   ├── output/
│   │   └── sidecar.py             # JSON sidecar read/write (atomic writes)
│   ├── utils/
│   │   ├── logging.py             # Logging setup helper
│   │   └── time_format.py         # Seconds -> HH:MM:SS formatter
│   └── video/
│       ├── frames.py              # Frame extraction from video
│       └── probe.py               # Video metadata probing via ffmpeg
└── tests/                         # pytest test suite (mirrors scanner/ layout)
```

---

## Troubleshooting

**Ollama is not reachable**

```
ERROR: Cannot connect to Ollama at http://localhost:11434: ...
```

Start Ollama with `ollama serve` (or `ollama run llava:latest`) and verify it is listening on port 11434 before running the scanner.

**No video files found**

```
No video files found in the specified folder.
```

Check the folder path and that your files have a supported extension: `.mp4`, `.avi`, `.mkv`, `.mov`, `.wmv`, `.flv`, `.webm`, `.m4v`, `.mpg`, `.mpeg`. If files are in sub-directories, pass `--recursive`.

**Whisper model downloads on first run**

faster-whisper downloads model weights from Hugging Face on first use. Subsequent runs use the cached model. Pass `--whisper-model tiny` for a much faster download if you just want to test.

**Summary shows `failed=N`**

Run with `--verbose` to see the full traceback for each failure. Common causes are corrupt video files or codec issues with ffmpeg.

**ffmpeg not found**

```
FileNotFoundError: [Errno 2] No such file or directory: 'ffmpeg'
```

Install ffmpeg and ensure it is on your `PATH`. On macOS: `brew install ffmpeg`. On Debian/Ubuntu: `sudo apt install ffmpeg`.
