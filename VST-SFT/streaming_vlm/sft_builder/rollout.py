from __future__ import annotations

from copy import deepcopy
from typing import Any

from .schema import BudgetState, MemoryState, WriteTrajectoryStep
from .segment import segment_sample
from .teacher_policy import ACTION_COST, apply_teacher_policy
from .utility_estimator import aggregate_clip_utility, normalize_utility


def init_memory_state() -> MemoryState:
    return MemoryState()


def init_budget_state(storage_budget: int, retrieval_budget: int = 0) -> BudgetState:
    return BudgetState(storage_left=storage_budget, retrieval_left=retrieval_budget)


def apply_action(memory_state: MemoryState, budget_state: BudgetState, action: str, write_content: str) -> tuple[MemoryState, BudgetState]:
    new_memory = deepcopy(memory_state)
    new_budget = deepcopy(budget_state)

    if action == "write_summary" and write_content:
        new_memory.summary_history.append(write_content)
    elif action == "write_visual" and write_content:
        new_memory.visual_history.append(write_content)
    elif action == "write_structured" and write_content:
        new_memory.structured_history.append(write_content)

    new_budget.storage_left = max(0, new_budget.storage_left - ACTION_COST[action])
    return new_memory, new_budget


def rollout_video(
    sample: dict[str, Any],
    clip_seconds: float = 4.0,
    storage_budget: int = 100,
    retrieval_budget: int = 0,
) -> list[dict[str, Any]]:
    """
    Expected raw sample format:
    {
      "video_id": str,
      "video_path": str,
      "duration": float,
      "queries": [
        {
          "question": str,
          "weight": optional float,
          "evidence_spans": [{"start": float, "end": float}, ...],
          "type": optional str
        }, ...
      ],
      "clip_captions": optional list[str],
      "keyframes": optional list[str],
      "novelty_scores": optional list[float]
    }
    """
    clips = segment_sample(sample, clip_seconds=clip_seconds)
    utility_by_clip = normalize_utility(aggregate_clip_utility(clips, sample.get("queries", [])))

    memory_state = init_memory_state()
    budget_state = init_budget_state(storage_budget, retrieval_budget)
    records: list[dict[str, Any]] = []

    for step_idx, clip in enumerate(clips):
        utility_info = utility_by_clip.get(clip.clip_id)
        if utility_info is None:
            continue

        pre_memory = deepcopy(memory_state)
        pre_budget = deepcopy(budget_state)
        teacher = apply_teacher_policy(
            clip=clip,
            utility_info=utility_info,
            memory_state=pre_memory,
            budget_state=pre_budget,
        )
        post_memory, post_budget = apply_action(
            memory_state=pre_memory,
            budget_state=pre_budget,
            action=teacher.teacher_action,
            write_content=teacher.write_content,
        )
        record = WriteTrajectoryStep(
            video_id=str(sample["video_id"]),
            step=step_idx,
            clip=clip,
            memory_state=pre_memory,
            budget_state=pre_budget,
            teacher_effort=teacher.teacher_effort,
            teacher_action=teacher.teacher_action,
            write_content=teacher.write_content,
            memory_call=teacher.memory_call,
            post_memory_state=post_memory,
            post_budget_state=post_budget,
            meta={
                "utility_score": utility_info.utility_score,
                "evidence_type": utility_info.evidence_type,
                "supporting_queries": utility_info.supporting_queries,
            },
        )
        records.append(record.to_dict())
        memory_state, budget_state = post_memory, post_budget

    return records
