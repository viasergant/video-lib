import pytest
from scanner.models import Event, TranscriptSegment, VideoAnalysis


def make_event(**kwargs):
    defaults = {
        "timestamp_seconds": 5.2,
        "timestamp_formatted": "00:00:05",
        "type": "scene_change",
        "description": "A scene transition",
    }
    return Event(**{**defaults, **kwargs})


def make_segment(**kwargs):
    defaults = {"start": 1.2, "end": 3.5, "text": "spoken words"}
    return TranscriptSegment(**{**defaults, **kwargs})


def make_analysis(**kwargs):
    defaults = {
        "file": "video.mp4",
        "duration_seconds": 120.5,
        "analyzed_at": "2026-03-23T10:00:00Z",
        "model": "llava:latest",
        "description": "Overall description",
    }
    return VideoAnalysis(**{**defaults, **kwargs})


class TestTranscriptSegment:
    def test_fields(self):
        seg = make_segment()
        assert seg.start == 1.2
        assert seg.end == 3.5
        assert seg.text == "spoken words"


class TestEvent:
    def test_fields(self):
        evt = make_event()
        assert evt.timestamp_seconds == 5.2
        assert evt.timestamp_formatted == "00:00:05"
        assert evt.type == "scene_change"
        assert evt.description == "A scene transition"

    @pytest.mark.parametrize("event_type", ["scene_change", "speech", "object", "custom"])
    def test_all_valid_types(self, event_type):
        evt = make_event(type=event_type)
        assert evt.type == event_type


class TestVideoAnalysis:
    def test_defaults_empty_lists(self):
        va = make_analysis()
        assert va.events == []
        assert va.transcript == []

    def test_fields(self):
        va = make_analysis(duration_seconds=60.0)
        assert va.file == "video.mp4"
        assert va.duration_seconds == 60.0
        assert va.analyzed_at == "2026-03-23T10:00:00Z"
        assert va.model == "llava:latest"
        assert va.description == "Overall description"

    def test_to_dict_structure(self):
        evt = make_event()
        seg = make_segment()
        va = make_analysis(events=[evt], transcript=[seg])
        d = va.to_dict()

        assert d["file"] == "video.mp4"
        assert d["duration_seconds"] == 120.5
        assert d["analyzed_at"] == "2026-03-23T10:00:00Z"
        assert d["model"] == "llava:latest"
        assert d["description"] == "Overall description"

        assert len(d["events"]) == 1
        e = d["events"][0]
        assert e["timestamp_seconds"] == 5.2
        assert e["timestamp_formatted"] == "00:00:05"
        assert e["type"] == "scene_change"
        assert e["description"] == "A scene transition"

        assert len(d["transcript"]) == 1
        s = d["transcript"][0]
        assert s["start"] == 1.2
        assert s["end"] == 3.5
        assert s["text"] == "spoken words"

    def test_to_dict_empty_events_and_transcript(self):
        va = make_analysis()
        d = va.to_dict()
        assert d["events"] == []
        assert d["transcript"] == []

    def test_to_dict_multiple_events(self):
        events = [
            make_event(timestamp_seconds=1.0, type="speech"),
            make_event(timestamp_seconds=10.0, type="object"),
        ]
        va = make_analysis(events=events)
        d = va.to_dict()
        assert len(d["events"]) == 2
        assert d["events"][0]["type"] == "speech"
        assert d["events"][1]["type"] == "object"

    def test_mutable_defaults_are_independent(self):
        va1 = make_analysis()
        va2 = make_analysis()
        va1.events.append(make_event())
        assert va2.events == []
