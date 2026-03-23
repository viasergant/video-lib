from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class TranscriptSegment:
    start: float
    end: float
    text: str


@dataclass
class Event:
    timestamp_seconds: float
    timestamp_formatted: str  # "HH:MM:SS"
    type: Literal["scene_change", "speech", "object", "custom"]
    description: str


@dataclass
class VideoAnalysis:
    file: str
    duration_seconds: float
    analyzed_at: str  # ISO 8601 UTC
    model: str
    description: str
    events: list[Event] = field(default_factory=list)
    transcript: list[TranscriptSegment] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "file": self.file,
            "duration_seconds": self.duration_seconds,
            "analyzed_at": self.analyzed_at,
            "model": self.model,
            "description": self.description,
            "events": [
                {
                    "timestamp_seconds": e.timestamp_seconds,
                    "timestamp_formatted": e.timestamp_formatted,
                    "type": e.type,
                    "description": e.description,
                }
                for e in self.events
            ],
            "transcript": [
                {"start": s.start, "end": s.end, "text": s.text}
                for s in self.transcript
            ],
        }
