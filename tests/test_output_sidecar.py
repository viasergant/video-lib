from __future__ import annotations

import json
from pathlib import Path

import pytest

from scanner.models import Event, TranscriptSegment, VideoAnalysis
from scanner.output.sidecar import (
    read_sidecar,
    sidecar_exists,
    sidecar_path,
    write_sidecar,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_analysis(**kwargs) -> VideoAnalysis:
    defaults = {
        "file": "video.mp4",
        "duration_seconds": 120.5,
        "analyzed_at": "2026-03-23T10:00:00Z",
        "model": "llava:latest",
        "description": "Test description",
    }
    return VideoAnalysis(**{**defaults, **kwargs})


def _make_event(**kwargs) -> Event:
    defaults = {
        "timestamp_seconds": 5.0,
        "timestamp_formatted": "00:00:05",
        "type": "scene_change",
        "description": "A cut",
    }
    return Event(**{**defaults, **kwargs})


def _make_segment(**kwargs) -> TranscriptSegment:
    defaults = {"start": 1.0, "end": 3.0, "text": "hello world"}
    return TranscriptSegment(**{**defaults, **kwargs})


# ---------------------------------------------------------------------------
# sidecar_path
# ---------------------------------------------------------------------------

class TestSidecarPath:
    def test_appends_json_to_str_path(self):
        result = sidecar_path("video.mp4")
        assert result == Path("video.mp4.json")

    def test_appends_json_to_pathlib_path(self):
        result = sidecar_path(Path("/data/movie.mkv"))
        assert result == Path("/data/movie.mkv.json")

    def test_returns_path_object(self):
        assert isinstance(sidecar_path("video.mp4"), Path)

    def test_no_extension_video(self):
        result = sidecar_path("video")
        assert result == Path("video.json")

    def test_nested_path(self):
        result = sidecar_path("/some/dir/clip.avi")
        assert result == Path("/some/dir/clip.avi.json")


# ---------------------------------------------------------------------------
# sidecar_exists
# ---------------------------------------------------------------------------

class TestSidecarExists:
    def test_returns_false_when_no_sidecar(self, tmp_path):
        video = tmp_path / "video.mp4"
        assert sidecar_exists(video) is False

    def test_returns_true_when_sidecar_present(self, tmp_path):
        video = tmp_path / "video.mp4"
        (tmp_path / "video.mp4.json").write_text("{}", encoding="utf-8")
        assert sidecar_exists(video) is True

    def test_video_file_existence_is_irrelevant(self, tmp_path):
        # The video itself does not need to exist; only the sidecar matters.
        video = tmp_path / "ghost.mp4"
        (tmp_path / "ghost.mp4.json").write_text("{}", encoding="utf-8")
        assert sidecar_exists(video) is True


# ---------------------------------------------------------------------------
# write_sidecar
# ---------------------------------------------------------------------------

class TestWriteSidecar:
    def test_creates_json_file_next_to_video(self, tmp_path):
        video = tmp_path / "clip.mp4"
        analysis = _make_analysis(file=str(video))
        write_sidecar(video, analysis)
        assert (tmp_path / "clip.mp4.json").exists()

    def test_no_tmp_file_left_after_write(self, tmp_path):
        video = tmp_path / "clip.mp4"
        write_sidecar(video, _make_analysis())
        assert not (tmp_path / "clip.mp4.json.tmp").exists()

    def test_written_content_is_valid_json(self, tmp_path):
        video = tmp_path / "clip.mp4"
        write_sidecar(video, _make_analysis())
        raw = (tmp_path / "clip.mp4.json").read_text(encoding="utf-8")
        data = json.loads(raw)
        assert data["file"] == "video.mp4"

    def test_overwrites_existing_sidecar(self, tmp_path):
        video = tmp_path / "clip.mp4"
        write_sidecar(video, _make_analysis(description="first"))
        write_sidecar(video, _make_analysis(description="second"))
        raw = (tmp_path / "clip.mp4.json").read_text(encoding="utf-8")
        assert json.loads(raw)["description"] == "second"

    def test_atomic_write_uses_tmp_then_replaces(self, tmp_path, monkeypatch):
        """Verify the tmp file is written before being replaced."""
        tmp_seen: list[str] = []
        original_replace = __import__("os").replace

        def _tracking_replace(src: str, dst: str) -> None:
            tmp_seen.append(src)
            original_replace(src, dst)

        monkeypatch.setattr("scanner.output.sidecar.os.replace", _tracking_replace)

        video = tmp_path / "clip.mp4"
        write_sidecar(video, _make_analysis())

        assert len(tmp_seen) == 1
        assert str(tmp_seen[0]).endswith(".tmp")

    def test_serialises_events_and_transcript(self, tmp_path):
        video = tmp_path / "clip.mp4"
        analysis = _make_analysis(
            events=[_make_event()],
            transcript=[_make_segment()],
        )
        write_sidecar(video, analysis)
        data = json.loads((tmp_path / "clip.mp4.json").read_text(encoding="utf-8"))
        assert len(data["events"]) == 1
        assert data["events"][0]["type"] == "scene_change"
        assert len(data["transcript"]) == 1
        assert data["transcript"][0]["text"] == "hello world"


# ---------------------------------------------------------------------------
# read_sidecar
# ---------------------------------------------------------------------------

class TestReadSidecar:
    def test_returns_none_when_file_missing(self, tmp_path):
        video = tmp_path / "ghost.mp4"
        assert read_sidecar(video) is None

    def test_returns_none_for_malformed_json(self, tmp_path):
        video = tmp_path / "bad.mp4"
        (tmp_path / "bad.mp4.json").write_text("not json", encoding="utf-8")
        assert read_sidecar(video) is None

    def test_returns_none_for_missing_required_keys(self, tmp_path):
        video = tmp_path / "incomplete.mp4"
        (tmp_path / "incomplete.mp4.json").write_text(
            json.dumps({"file": "incomplete.mp4"}), encoding="utf-8"
        )
        assert read_sidecar(video) is None

    def test_roundtrip_minimal_analysis(self, tmp_path):
        video = tmp_path / "video.mp4"
        original = _make_analysis()
        write_sidecar(video, original)
        recovered = read_sidecar(video)

        assert recovered is not None
        assert recovered.file == original.file
        assert recovered.duration_seconds == original.duration_seconds
        assert recovered.analyzed_at == original.analyzed_at
        assert recovered.model == original.model
        assert recovered.description == original.description
        assert recovered.events == []
        assert recovered.transcript == []

    def test_roundtrip_with_events_and_transcript(self, tmp_path):
        video = tmp_path / "video.mp4"
        original = _make_analysis(
            events=[_make_event(timestamp_seconds=10.0, type="speech")],
            transcript=[_make_segment(start=2.0, end=5.0, text="spoken")],
        )
        write_sidecar(video, original)
        recovered = read_sidecar(video)

        assert recovered is not None
        assert len(recovered.events) == 1
        assert recovered.events[0].timestamp_seconds == 10.0
        assert recovered.events[0].type == "speech"

        assert len(recovered.transcript) == 1
        assert recovered.transcript[0].start == 2.0
        assert recovered.transcript[0].text == "spoken"

    def test_returns_video_analysis_instance(self, tmp_path):
        video = tmp_path / "video.mp4"
        write_sidecar(video, _make_analysis())
        result = read_sidecar(video)
        assert isinstance(result, VideoAnalysis)

    def test_returns_none_for_empty_json_object(self, tmp_path):
        video = tmp_path / "empty.mp4"
        (tmp_path / "empty.mp4.json").write_text("{}", encoding="utf-8")
        assert read_sidecar(video) is None

    def test_warns_on_malformed_file(self, tmp_path, caplog):
        import logging

        video = tmp_path / "bad.mp4"
        (tmp_path / "bad.mp4.json").write_text("not json", encoding="utf-8")

        with caplog.at_level(logging.WARNING, logger="scanner.output.sidecar"):
            read_sidecar(video)

        assert any("Failed to read sidecar" in r.message for r in caplog.records)
