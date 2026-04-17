from __future__ import annotations

from typing import Any

from .schema import BudgetState, ClipInfo, MemoryState, TeacherDecision, UtilityInfo


ACTION_COST = {
    "skip": 0,
    "write_summary": 1,
    "write_structured": 2,
    "write_visual": 3,
}


ACTION_MEMORY_LEVEL = {
    "skip": "none",
    "write_summary": "summary",
    "write_structured": "structured",
    "write_visual": "visual",
}


def assign_teacher_effort(utility_score: float, novelty_score: float, tau_low: float = 0.2, tau_high: float = 0.6) -> str:
    if utility_score < tau_low:
        return "cheap"
    if utility_score > tau_high:
        return "expensive"
    return "expensive" if novelty_score >= 0.5 else "cheap"


def assign_teacher_action(
    utility_info: UtilityInfo,
    novelty_score: float,
    effort: str,
) -> str:
    if utility_info.utility_score < 0.1 and novelty_score < 0.2:
        return "skip"
    if effort == "cheap":
        return "write_summary"
    if utility_info.evidence_type == "detail":
        return "write_visual"
    if utility_info.evidence_type == "event_fact":
        return "write_structured"
    return "write_summary"


def synthesize_write_content(
    action: str,
    clip: ClipInfo,
    utility_info: UtilityInfo,
    memory_state: MemoryState,
) -> str:
    caption = clip.caption.strip() or f"clip_{clip.clip_id} from {clip.start:.1f}s to {clip.end:.1f}s"
    if action == "skip":
        return ""
    if action == "write_summary":
        return f"[summary] {caption}"
    if action == "write_visual":
        visual_ref = clip.keyframe or f"{clip.start:.1f}-{clip.end:.1f}s"
        return f"[visual] keyframe={visual_ref}; evidence={caption}"
    if action == "write_structured":
        return f"[structured] event={caption}; support_queries={len(utility_info.supporting_queries)}"
    raise ValueError(f"Unknown action: {action}")


def synthesize_memory_call(
    effort: str,
    action: str,
    write_content: str,
    clip: ClipInfo,
    utility_info: UtilityInfo,
) -> dict[str, Any]:
    memory_level = ACTION_MEMORY_LEVEL[action]
    cost = ACTION_COST[action]

    if action == "skip":
        content: dict[str, Any] | str = ""
    elif action == "write_summary":
        content = {
            "time_span": [clip.start, clip.end],
            "text": write_content,
        }
    elif action == "write_visual":
        # v0 follows a DeepEyes-style text-ref visual observation: store a keyframe
        # reference plus a short anchor. The updater can later materialize image_ref.
        content = {
            "time_span": [clip.start, clip.end],
            "image_ref": clip.keyframe or "",
            "text_anchor": write_content,
            "visual_target": "keyframe",
        }
    elif action == "write_structured":
        content = {
            "time_span": [clip.start, clip.end],
            "event": write_content,
            "supporting_queries": utility_info.supporting_queries,
        }
    else:
        raise ValueError(f"Unknown action: {action}")

    return {
        "effort": effort,
        "action": action,
        "memory_level": memory_level,
        "content": content,
        "cost": cost,
    }


def apply_teacher_policy(
    clip: ClipInfo,
    utility_info: UtilityInfo,
    memory_state: MemoryState,
    budget_state: BudgetState,
) -> TeacherDecision:
    effort = assign_teacher_effort(utility_info.utility_score, clip.novelty_score)
    action = assign_teacher_action(utility_info, clip.novelty_score, effort)
    content = synthesize_write_content(action, clip, utility_info, memory_state)
    memory_call = synthesize_memory_call(effort, action, content, clip, utility_info)
    return TeacherDecision(
        teacher_effort=effort,
        teacher_action=action,
        write_content=content,
        memory_call=memory_call,
    )
