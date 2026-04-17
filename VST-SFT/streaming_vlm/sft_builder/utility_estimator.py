from __future__ import annotations

from typing import Any

from .schema import ClipInfo, UtilityInfo


def overlap_ratio(clip_start: float, clip_end: float, span_start: float, span_end: float) -> float:
    inter = max(0.0, min(clip_end, span_end) - max(clip_start, span_start))
    union = max(1e-6, max(clip_end, span_end) - min(clip_start, span_start))
    return inter / union


def infer_evidence_type(query_record: dict[str, Any]) -> str:
    """Very small v0 heuristic. Replace later with stronger teacher or judge."""
    q_type = str(query_record.get("type", "")).lower()
    question = str(query_record.get("question", "")).lower()
    if any(word in q_type or word in question for word in ["color", "text", "ocr", "appearance", "object", "detail"]):
        return "detail"
    if any(word in q_type or word in question for word in ["event", "state", "relation", "fact", "count", "order", "why", "cause"]):
        return "event_fact"
    return "context"


def aggregate_clip_utility(clips: list[ClipInfo], queries: list[dict[str, Any]]) -> dict[int, UtilityInfo]:
    """
    Expected query format:
    {
      "question": str,
      "weight": optional float,
      "evidence_spans": [{"start": float, "end": float}, ...],
      "type": optional str
    }
    """
    out: dict[int, UtilityInfo] = {
        clip.clip_id: UtilityInfo(utility_score=0.0, evidence_type="context", supporting_queries=[])
        for clip in clips
    }
    evidence_type_priority = {"context": 0, "detail": 1, "event_fact": 2}

    for q in queries:
        weight = float(q.get("weight", 1.0))
        evidence_type = infer_evidence_type(q)
        spans = q.get("evidence_spans", [])
        if not spans:
            continue
        for clip in clips:
            support = 0.0
            for span in spans:
                support = max(
                    support,
                    overlap_ratio(
                        clip.start,
                        clip.end,
                        float(span["start"]),
                        float(span["end"]),
                    ),
                )
            if support <= 0:
                continue
            item = out[clip.clip_id]
            item.utility_score += weight * support
            item.supporting_queries.append(str(q.get("question", "")))
            if evidence_type_priority[evidence_type] > evidence_type_priority[item.evidence_type]:
                item.evidence_type = evidence_type

    return out


def normalize_utility(utility_by_clip: dict[int, UtilityInfo]) -> dict[int, UtilityInfo]:
    max_score = max((item.utility_score for item in utility_by_clip.values()), default=0.0)
    if max_score <= 0:
        return utility_by_clip
    for item in utility_by_clip.values():
        item.utility_score = item.utility_score / max_score
    return utility_by_clip

