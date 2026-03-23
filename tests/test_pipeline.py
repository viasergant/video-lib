from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import numpy as np
import pytest

from scanner.models import Event, TranscriptSegment, VideoAnalysis, VideoMeta
from scanner.pipeline import VideoProcessor, _build_frame_timestamps

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

_VIDEO_PATH = Path("/fake/video.mp4")
_MODEL = "llava:latest"
_PROMPT = "Describe what is happening."

_META = VideoMeta(
    path=str(_VIDEO_PATH),
    duration=10.0,
    fps=30.0,
    width=1920,
    height=1080,
    codec="h264",
)

_SPEECH_SEGMENT = TranscriptSegment(start=1.0, end=3.0, text="Hello world")
_SPEECH_EVENT = Event(
    timestamp_seconds=1.0,
    timestamp_formatted="00:00:01",
    type="speech",
    description="Hello world",
)


def _make_processor(
    ollama_client: MagicMock | None = None,
    whisper_model: MagicMock | None = None,
    model: str = _MODEL,
    prompt: str = _PROMPT,
) -> VideoProcessor:
    return VideoProcessor(
        ollama_client=ollama_client or MagicMock(),
        whisper_model=whisper_model or MagicMock(),
        model=model,
        prompt=prompt,
    )


def _patched_process(monkeypatch, *, scene_timestamps=None, transcript=None,
                     speech_events=None, frames=None, frame_b64="b64frame",
                     describe_frame_return="frame desc",
                     describe_overall_return="overall desc",
                     write_sidecar_fn=None):
    """Return (processor, mocks-dict) with all external calls patched."""
    if scene_timestamps is None:
        scene_timestamps = [1.0, 3.0, 6.0]
    if transcript is None:
        transcript = []
    if speech_events is None:
        speech_events = []
    if frames is None:
        frames = [(ts, np.zeros((10, 10, 3), dtype=np.uint8)) for ts in scene_timestamps]

    mock_probe = MagicMock(return_value=_META)
    mock_detect = MagicMock(return_value=scene_timestamps)
    mock_transcribe = MagicMock(return_value=(transcript, speech_events))
    mock_extract = MagicMock(return_value=iter(frames))
    mock_b64 = MagicMock(return_value=frame_b64)
    mock_write = write_sidecar_fn or MagicMock()

    ollama_client = MagicMock()
    ollama_client.generate.return_value = {"response": describe_frame_return}

    monkeypatch.setattr("scanner.pipeline.probe_video", mock_probe)
    monkeypatch.setattr("scanner.pipeline.detect_scene_changes", mock_detect)
    monkeypatch.setattr("scanner.pipeline.transcribe_speech", mock_transcribe)
    monkeypatch.setattr("scanner.pipeline.extract_frames", mock_extract)
    monkeypatch.setattr("scanner.pipeline.frames_to_base64_jpeg", mock_b64)
    monkeypatch.setattr("scanner.pipeline.write_sidecar", mock_write)

    # overall description needs a separate return value
    mock_overall = MagicMock(return_value=describe_overall_return)

    processor = _make_processor(ollama_client=ollama_client)
    processor._analyzer.describe_video_overall = mock_overall

    mocks = dict(
        probe=mock_probe,
        detect=mock_detect,
        transcribe=mock_transcribe,
        extract=mock_extract,
        b64=mock_b64,
        write=mock_write,
        overall=mock_overall,
        client=ollama_client,
    )
    return processor, mocks


# ---------------------------------------------------------------------------
# _build_frame_timestamps
# ---------------------------------------------------------------------------

class TestBuildFrameTimestamps:
    def test_uses_scene_timestamps_when_three_or_more(self):
        scenes = [1.0, 5.0, 9.0]
        result = _build_frame_timestamps(scenes, duration=20.0)
        assert result == sorted(scenes)

    def test_uses_scene_timestamps_when_more_than_three(self):
        scenes = [0.5, 2.0, 5.0, 8.0, 11.0]
        result = _build_frame_timestamps(scenes, duration=15.0)
        assert result == sorted(scenes)

    def test_falls_back_to_grid_when_fewer_than_three_scenes(self):
        result = _build_frame_timestamps([1.0, 2.0], duration=10.0)
        # 10s / 2s interval = 5 frames on grid
        assert len(result) == 5
        assert all(0.0 <= t < 10.0 for t in result)

    def test_falls_back_to_grid_when_no_scenes(self):
        result = _build_frame_timestamps([], duration=6.0)
        assert len(result) == 3
        assert result[0] == pytest.approx(0.0)

    def test_grid_capped_at_30_frames(self):
        # duration=200s → 100 intervals, capped at 30
        result = _build_frame_timestamps([], duration=200.0)
        assert len(result) == 30

    def test_grid_returns_empty_for_zero_duration(self):
        assert _build_frame_timestamps([], duration=0.0) == []

    def test_grid_returns_empty_for_negative_duration(self):
        assert _build_frame_timestamps([], duration=-5.0) == []

    def test_grid_returns_sorted_timestamps(self):
        result = _build_frame_timestamps([], duration=10.0)
        assert result == sorted(result)

    def test_scene_timestamps_returned_sorted(self):
        scenes = [9.0, 1.0, 5.0]
        result = _build_frame_timestamps(scenes, duration=20.0)
        assert result == [1.0, 5.0, 9.0]

    def test_exactly_three_scenes_uses_scene_path(self):
        scenes = [1.0, 2.0, 3.0]
        result = _build_frame_timestamps(scenes, duration=10.0)
        assert result == scenes


# ---------------------------------------------------------------------------
# VideoProcessor.process — happy path
# ---------------------------------------------------------------------------

class TestVideoProcessorHappyPath:
    def test_returns_video_analysis(self, monkeypatch):
        processor, _ = _patched_process(monkeypatch)
        result = processor.process(_VIDEO_PATH)
        assert isinstance(result, VideoAnalysis)

    def test_analysis_file_is_video_path(self, monkeypatch):
        processor, _ = _patched_process(monkeypatch)
        result = processor.process(_VIDEO_PATH)
        assert result.file == str(_VIDEO_PATH)

    def test_analysis_duration_from_meta(self, monkeypatch):
        processor, _ = _patched_process(monkeypatch)
        result = processor.process(_VIDEO_PATH)
        assert result.duration_seconds == _META.duration

    def test_analysis_model_matches(self, monkeypatch):
        processor, _ = _patched_process(monkeypatch)
        result = processor.process(_VIDEO_PATH)
        assert result.model == _MODEL

    def test_analysis_description_is_overall(self, monkeypatch):
        processor, _ = _patched_process(
            monkeypatch, describe_overall_return="The video shows a cat."
        )
        result = processor.process(_VIDEO_PATH)
        assert result.description == "The video shows a cat."

    def test_analyzed_at_is_iso_utc(self, monkeypatch):
        processor, _ = _patched_process(monkeypatch)
        result = processor.process(_VIDEO_PATH)
        assert result.analyzed_at.endswith("Z")
        assert "T" in result.analyzed_at

    def test_sidecar_written(self, monkeypatch):
        processor, mocks = _patched_process(monkeypatch)
        processor.process(_VIDEO_PATH)
        mocks["write"].assert_called_once()

    def test_probe_called_with_video_path(self, monkeypatch):
        processor, mocks = _patched_process(monkeypatch)
        processor.process(_VIDEO_PATH)
        mocks["probe"].assert_called_once_with(_VIDEO_PATH)

    def test_scene_detection_called_with_meta_fps(self, monkeypatch):
        processor, mocks = _patched_process(monkeypatch)
        processor.process(_VIDEO_PATH)
        mocks["detect"].assert_called_once_with(_VIDEO_PATH, _META.fps)

    def test_transcription_called_with_whisper_model(self, monkeypatch):
        processor, mocks = _patched_process(monkeypatch)
        processor.process(_VIDEO_PATH)
        mocks["transcribe"].assert_called_once_with(_VIDEO_PATH, processor._whisper_model)


# ---------------------------------------------------------------------------
# Events assembly
# ---------------------------------------------------------------------------

class TestEventsAssembly:
    def test_scene_change_events_in_result(self, monkeypatch):
        processor, _ = _patched_process(monkeypatch, scene_timestamps=[2.0, 5.0, 8.0])
        result = processor.process(_VIDEO_PATH)
        scene_evts = [e for e in result.events if e.type == "scene_change"]
        assert len(scene_evts) == 3

    def test_speech_events_in_result(self, monkeypatch):
        processor, _ = _patched_process(
            monkeypatch,
            transcript=[_SPEECH_SEGMENT],
            speech_events=[_SPEECH_EVENT],
        )
        result = processor.process(_VIDEO_PATH)
        speech_evts = [e for e in result.events if e.type == "speech"]
        assert len(speech_evts) == 1
        assert speech_evts[0].description == "Hello world"

    def test_frame_description_events_in_result(self, monkeypatch):
        scenes = [1.0, 4.0, 7.0]
        processor, _ = _patched_process(
            monkeypatch,
            scene_timestamps=scenes,
            describe_frame_return="a person",
        )
        result = processor.process(_VIDEO_PATH)
        frame_evts = [e for e in result.events if e.type == "custom"]
        assert len(frame_evts) == 3
        assert all(e.description == "a person" for e in frame_evts)

    def test_events_sorted_by_timestamp(self, monkeypatch):
        speech_evt = Event(
            timestamp_seconds=3.5,
            timestamp_formatted="00:00:03",
            type="speech",
            description="hi",
        )
        processor, _ = _patched_process(
            monkeypatch,
            scene_timestamps=[1.0, 5.0, 9.0],
            speech_events=[speech_evt],
        )
        result = processor.process(_VIDEO_PATH)
        timestamps = [e.timestamp_seconds for e in result.events]
        assert timestamps == sorted(timestamps)

    def test_transcript_included(self, monkeypatch):
        processor, _ = _patched_process(
            monkeypatch,
            transcript=[_SPEECH_SEGMENT],
            speech_events=[_SPEECH_EVENT],
        )
        result = processor.process(_VIDEO_PATH)
        assert len(result.transcript) == 1
        assert result.transcript[0].text == "Hello world"


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

class TestErrorHandling:
    def test_probe_failure_raises(self, monkeypatch):
        monkeypatch.setattr(
            "scanner.pipeline.probe_video",
            MagicMock(side_effect=RuntimeError("ffprobe failed")),
        )
        processor = _make_processor()
        with pytest.raises(RuntimeError):
            processor.process(_VIDEO_PATH)

    def test_probe_failure_does_not_write_sidecar(self, monkeypatch):
        mock_write = MagicMock()
        monkeypatch.setattr(
            "scanner.pipeline.probe_video",
            MagicMock(side_effect=RuntimeError("ffprobe failed")),
        )
        monkeypatch.setattr("scanner.pipeline.write_sidecar", mock_write)
        processor = _make_processor()
        with pytest.raises(RuntimeError):
            processor.process(_VIDEO_PATH)
        mock_write.assert_not_called()

    def test_frame_extraction_failure_still_writes_sidecar(self, monkeypatch):
        mock_probe = MagicMock(return_value=_META)
        mock_detect = MagicMock(return_value=[1.0, 3.0, 6.0])
        mock_transcribe = MagicMock(return_value=([], []))
        mock_extract = MagicMock(side_effect=RuntimeError("cv2 error"))
        mock_write = MagicMock()

        ollama_client = MagicMock()
        ollama_client.generate.return_value = {"response": "desc"}

        monkeypatch.setattr("scanner.pipeline.probe_video", mock_probe)
        monkeypatch.setattr("scanner.pipeline.detect_scene_changes", mock_detect)
        monkeypatch.setattr("scanner.pipeline.transcribe_speech", mock_transcribe)
        monkeypatch.setattr("scanner.pipeline.extract_frames", mock_extract)
        monkeypatch.setattr("scanner.pipeline.write_sidecar", mock_write)

        processor = _make_processor(ollama_client=ollama_client)
        processor._analyzer.describe_video_overall = MagicMock(return_value="overall")
        result = processor.process(_VIDEO_PATH)

        mock_write.assert_called_once()
        assert isinstance(result, VideoAnalysis)

    def test_jpeg_encoding_failure_skips_frame(self, monkeypatch):
        """A frame that fails JPEG encoding is skipped but others still processed."""
        scenes = [1.0, 3.0, 6.0]
        frames = [(ts, np.zeros((10, 10, 3), dtype=np.uint8)) for ts in scenes]

        call_count = 0

        def flaky_b64(frame):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise RuntimeError("encode failed")
            return "b64data"

        monkeypatch.setattr("scanner.pipeline.probe_video", MagicMock(return_value=_META))
        monkeypatch.setattr("scanner.pipeline.detect_scene_changes", MagicMock(return_value=scenes))
        monkeypatch.setattr("scanner.pipeline.transcribe_speech", MagicMock(return_value=([], [])))
        monkeypatch.setattr("scanner.pipeline.extract_frames", MagicMock(return_value=iter(frames)))
        monkeypatch.setattr("scanner.pipeline.frames_to_base64_jpeg", flaky_b64)
        monkeypatch.setattr("scanner.pipeline.write_sidecar", MagicMock())

        ollama_client = MagicMock()
        ollama_client.generate.return_value = {"response": "frame desc"}
        processor = _make_processor(ollama_client=ollama_client)
        processor._analyzer.describe_video_overall = MagicMock(return_value="overall")

        result = processor.process(_VIDEO_PATH)
        frame_evts = [e for e in result.events if e.type == "custom"]
        # 2 of 3 frames succeed
        assert len(frame_evts) == 2

    def test_overall_description_unavailable_when_no_frames(self, monkeypatch):
        processor, _ = _patched_process(
            monkeypatch,
            scene_timestamps=[1.0, 3.0, 6.0],
            describe_overall_return="[analysis unavailable]",
        )
        result = processor.process(_VIDEO_PATH)
        assert result.description == "[analysis unavailable]"

    def test_logs_warning_when_frame_extraction_fails(self, monkeypatch, caplog):
        monkeypatch.setattr("scanner.pipeline.probe_video", MagicMock(return_value=_META))
        monkeypatch.setattr(
            "scanner.pipeline.detect_scene_changes", MagicMock(return_value=[1.0, 3.0, 6.0])
        )
        monkeypatch.setattr(
            "scanner.pipeline.transcribe_speech", MagicMock(return_value=([], []))
        )
        monkeypatch.setattr(
            "scanner.pipeline.extract_frames",
            MagicMock(side_effect=RuntimeError("boom")),
        )
        monkeypatch.setattr("scanner.pipeline.write_sidecar", MagicMock())

        ollama_client = MagicMock()
        ollama_client.generate.return_value = {"response": "desc"}
        processor = _make_processor(ollama_client=ollama_client)
        processor._analyzer.describe_video_overall = MagicMock(return_value="overall")

        with caplog.at_level(logging.WARNING, logger="scanner.pipeline"):
            processor.process(_VIDEO_PATH)

        assert any("Frame extraction failed" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# Grid fallback
# ---------------------------------------------------------------------------

class TestGridFallback:
    def test_grid_used_when_fewer_than_3_scene_changes(self, monkeypatch):
        processor, mocks = _patched_process(
            monkeypatch,
            scene_timestamps=[2.0],  # only 1 → fallback
            frames=[],  # no frames to iterate for simplicity
        )
        # patch extract_frames to capture the timestamps it is called with
        captured_ts = []

        def fake_extract(path, timestamps, **kwargs):
            captured_ts.extend(timestamps)
            return iter([])

        monkeypatch.setattr("scanner.pipeline.extract_frames", fake_extract)
        processor.process(_VIDEO_PATH)

        # 10s duration / 2s = 5 grid timestamps
        assert len(captured_ts) == 5

    def test_scene_timestamps_used_when_3_or_more(self, monkeypatch):
        scenes = [1.0, 4.0, 7.0]
        captured_ts = []

        def fake_extract(path, timestamps, **kwargs):
            captured_ts.extend(timestamps)
            return iter([])

        processor, mocks = _patched_process(
            monkeypatch,
            scene_timestamps=scenes,
            frames=[],
        )
        monkeypatch.setattr("scanner.pipeline.extract_frames", fake_extract)
        processor.process(_VIDEO_PATH)

        assert captured_ts == sorted(scenes)
