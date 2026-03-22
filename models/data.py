"""Data models used across all pipeline stages."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Detection:
    frame_num: int
    track_id: int
    bbox: tuple  # (x1, y1, x2, y2)
    confidence: float
    class_id: int  # 0=person, 32=ball

    @property
    def center(self) -> tuple:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    @property
    def width(self) -> float:
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> float:
        return self.bbox[3] - self.bbox[1]

    def to_dict(self) -> dict:
        return {
            "frame_num": self.frame_num,
            "track_id": self.track_id,
            "bbox": list(self.bbox),
            "confidence": self.confidence,
            "class_id": self.class_id,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Detection":
        return cls(
            frame_num=d["frame_num"],
            track_id=d["track_id"],
            bbox=tuple(d["bbox"]),
            confidence=d["confidence"],
            class_id=d["class_id"],
        )


@dataclass
class PlayerIdentity:
    track_ids: list  # May have multiple track IDs (merged)
    jersey_number: Optional[int] = None
    jersey_color: str = ""
    team: str = ""  # "team_a" or "team_b"
    display_name: str = ""
    is_target: bool = False

    def to_dict(self) -> dict:
        return {
            "track_ids": self.track_ids,
            "jersey_number": self.jersey_number,
            "jersey_color": self.jersey_color,
            "team": self.team,
            "display_name": self.display_name,
            "is_target": self.is_target,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "PlayerIdentity":
        return cls(**d)


@dataclass
class TimeRange:
    start_frame: int
    end_frame: int
    start_time: str = ""  # "MM:SS"
    end_time: str = ""

    def to_dict(self) -> dict:
        return {
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "start_time": self.start_time,
            "end_time": self.end_time,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "TimeRange":
        return cls(**d)


@dataclass
class KeyMoment:
    moment_type: str  # "sprint", "direction_change", "ball_involvement", "positional_shift"
    time_range: TimeRange
    description: str = ""

    def to_dict(self) -> dict:
        return {
            "moment_type": self.moment_type,
            "time_range": self.time_range.to_dict(),
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "KeyMoment":
        return cls(
            moment_type=d["moment_type"],
            time_range=TimeRange.from_dict(d["time_range"]),
            description=d["description"],
        )


@dataclass
class PlayerStats:
    player_id: str
    team: str
    jersey_number: Optional[int]
    inferred_position: str
    total_distance_m: float
    avg_speed_mps: float
    max_speed_mps: float
    sprint_count: int
    time_in_thirds: dict  # {"defensive": %, "middle": %, "attacking": %}
    sprint_episodes: list  # list of TimeRange dicts
    ball_proximity_episodes: list  # list of TimeRange dicts
    key_moments: list  # list of KeyMoment dicts
    heatmap: list  # 2D grid
    frames_visible: int
    total_time_visible_s: float
    is_target: bool = False

    def to_dict(self) -> dict:
        return {
            "player_id": self.player_id,
            "team": self.team,
            "jersey_number": self.jersey_number,
            "inferred_position": self.inferred_position,
            "total_distance_m": round(self.total_distance_m, 1),
            "avg_speed_mps": round(self.avg_speed_mps, 2),
            "max_speed_mps": round(self.max_speed_mps, 2),
            "sprint_count": self.sprint_count,
            "time_in_thirds": {k: round(v, 1) for k, v in self.time_in_thirds.items()},
            "sprint_episodes": [e.to_dict() if hasattr(e, "to_dict") else e for e in self.sprint_episodes],
            "ball_proximity_episodes": [e.to_dict() if hasattr(e, "to_dict") else e for e in self.ball_proximity_episodes],
            "key_moments": [m.to_dict() if hasattr(m, "to_dict") else m for m in self.key_moments],
            "heatmap": self.heatmap,
            "frames_visible": self.frames_visible,
            "total_time_visible_s": round(self.total_time_visible_s, 1),
            "is_target": self.is_target,
        }


@dataclass
class Suggestion:
    timestamp_start: str
    timestamp_end: str
    description: str
    recommendation: str
    reasoning: str

    def to_dict(self) -> dict:
        return {
            "timestamp_start": self.timestamp_start,
            "timestamp_end": self.timestamp_end,
            "description": self.description,
            "recommendation": self.recommendation,
            "reasoning": self.reasoning,
        }


@dataclass
class PlayerAnalysis:
    player_id: str
    summary: str
    strengths: list
    areas_to_improve: list
    suggestions: list  # list of Suggestion dicts

    def to_dict(self) -> dict:
        return {
            "player_id": self.player_id,
            "summary": self.summary,
            "strengths": self.strengths,
            "areas_to_improve": self.areas_to_improve,
            "suggestions": [s.to_dict() if hasattr(s, "to_dict") else s for s in self.suggestions],
        }


@dataclass
class TeamStrategy:
    team: str
    observations: list  # list of dicts with title, description, timestamp_refs, recommendation

    def to_dict(self) -> dict:
        return {
            "team": self.team,
            "observations": self.observations,
        }
