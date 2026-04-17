from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class ClipInfo:
    start: float
    end: float
    video: str
    clip_id: int
    caption: str = ""
    keyframe: str = ""
    novelty_score: float = 0.0


@dataclass
class UtilityInfo:
    utility_score: float
    evidence_type: str
    supporting_queries: list[str] = field(default_factory=list)


@dataclass
class MemoryState:
    summary_history: list[str] = field(default_factory=list)
    structured_history: list[str] = field(default_factory=list)
    visual_history: list[str] = field(default_factory=list)


@dataclass
class BudgetState:
    storage_left: int
    retrieval_left: int = 0


@dataclass
class TeacherDecision:
    teacher_effort: str
    teacher_action: str
    write_content: str
    memory_call: dict[str, Any] = field(default_factory=dict)


@dataclass
class WriteTrajectoryStep:
    video_id: str
    step: int
    clip: ClipInfo
    memory_state: MemoryState
    budget_state: BudgetState
    teacher_effort: str
    teacher_action: str
    write_content: str
    memory_call: dict[str, Any] = field(default_factory=dict)
    post_memory_state: MemoryState | None = None
    post_budget_state: BudgetState | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
