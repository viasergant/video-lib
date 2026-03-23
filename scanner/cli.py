from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import click

from scanner.config import (
    DEFAULT_OLLAMA_HOST,
    DEFAULT_OLLAMA_MODEL,
    SUPPORTED_EXTENSIONS,
    WHISPER_COMPUTE_TYPE,
    WHISPER_DEVICE,
    WHISPER_MODEL,
)
from scanner.output.sidecar import sidecar_exists
from scanner.pipeline import VideoProcessor
from scanner.utils.logging import setup_logging

if TYPE_CHECKING:
    import ollama
    from faster_whisper import WhisperModel

logger = logging.getLogger(__name__)


@click.command()
@click.option(
    "--folder",
    required=True,
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
    help="Path to the folder to scan for video files.",
)
@click.option(
    "--prompt",
    default="Describe what is happening in this video.",
    show_default=True,
    help="Prompt passed to the vision model for each frame and overall description.",
)
@click.option(
    "--recursive",
    is_flag=True,
    default=False,
    help="Recurse into sub-folders when discovering video files.",
)
@click.option(
    "--model",
    default=DEFAULT_OLLAMA_MODEL,
    show_default=True,
    help="Ollama model name to use for vision inference.",
)
@click.option(
    "--whisper-model",
    "whisper_model_name",
    default=WHISPER_MODEL,
    show_default=True,
    help="faster-whisper model size (e.g. tiny, base, small, medium, large).",
)
@click.option(
    "--workers",
    default=1,
    show_default=True,
    type=int,
    help="Number of parallel video-processing workers.",
)
@click.option(
    "--verbose",
    is_flag=True,
    default=False,
    help="Enable DEBUG-level logging.",
)
def main(
    folder: Path,
    prompt: str,
    recursive: bool,
    model: str,
    whisper_model_name: str,
    workers: int,
    verbose: bool,
) -> None:
    """Scan a folder for video files and generate JSON sidecar analysis files."""
    setup_logging(verbose=verbose)

    # lazy imports so the module stays importable even when these heavy
    # optional packages are absent (e.g. during unit-test collection)
    import ollama as _ollama
    from faster_whisper import WhisperModel as _WhisperModel

    # Step 1: check Ollama connectivity
    ollama_client = _ollama.Client(host=DEFAULT_OLLAMA_HOST)
    try:
        ollama_client.list()
    except Exception as exc:
        click.echo(
            f"ERROR: Cannot connect to Ollama at {DEFAULT_OLLAMA_HOST}: {exc}",
            err=True,
        )
        sys.exit(1)

    logger.debug("Ollama connectivity confirmed at %s", DEFAULT_OLLAMA_HOST)

    # Step 2: load WhisperModel once, shared across all videos
    logger.info("Loading Whisper model '%s' ...", whisper_model_name)
    whisper = _WhisperModel(
        whisper_model_name,
        device=WHISPER_DEVICE,
        compute_type=WHISPER_COMPUTE_TYPE,
    )
    logger.debug("Whisper model loaded")

    # Step 3: discover video files
    glob_fn = folder.rglob if recursive else folder.glob
    video_files: list[Path] = sorted(
        p
        for ext in SUPPORTED_EXTENSIONS
        for p in glob_fn(f"*{ext}")
        if p.is_file()
    )

    if not video_files:
        click.echo("No video files found in the specified folder.", err=True)
        sys.exit(0)

    logger.info("Found %d video file(s) in %s", len(video_files), folder)

    # Step 4: process each video; skip if sidecar already exists
    processor = VideoProcessor(
        ollama_client=ollama_client,
        whisper_model=whisper,
        model=model,
        prompt=prompt,
    )

    processed = 0
    skipped = 0
    failed = 0

    for video_path in video_files:
        if sidecar_exists(video_path):
            logger.info("Skipping %s (sidecar already exists)", video_path)
            skipped += 1
            continue

        logger.info("Processing %s ...", video_path)
        try:
            processor.process(video_path)
            processed += 1
        except Exception:
            logger.error("Failed to process %s", video_path, exc_info=True)
            failed += 1

    click.echo(
        f"Done. processed={processed}, skipped={skipped}, failed={failed}"
    )
