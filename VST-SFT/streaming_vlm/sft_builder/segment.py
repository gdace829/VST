from __future__ import annotations

import math
from typing import Any

from .schema import ClipInfo


def fixed_segment_video(video_id: str, video_path: str, duration: float, clip_seconds: float) -> list[ClipInfo]:
    """Split a video into fixed-length causal clips."""
    if clip_seconds <= 0:
        raise ValueError(f"clip_seconds must be positive, got {clip_seconds}")
    if duration <= 0:
        raise ValueError(f"duration must be positive, got {duration}")

    clips: list[ClipInfo] = []
    num_clips = int(math.ceil(duration / clip_seconds))
    for clip_id in range(num_clips):
        start = clip_id * clip_seconds
        end = min(duration, start + clip_seconds)
        clips.append(
            ClipInfo(
                start=float(start),
                end=float(end),
                video=video_path,
                clip_id=clip_id,
            )
        )
    return clips


def attach_lightweight_features(
    clips: list[ClipInfo],
    clip_captions: list[str] | None = None,
    keyframes: list[str] | None = None,
    novelty_scores: list[float] | None = None,
) -> list[ClipInfo]:
    """Attach optional lightweight cues to segmented clips."""
    for idx, clip in enumerate(clips):
        if clip_captions is not None and idx < len(clip_captions):
            clip.caption = clip_captions[idx]
        if keyframes is not None and idx < len(keyframes):
            clip.keyframe = keyframes[idx]
        if novelty_scores is not None and idx < len(novelty_scores):
            clip.novelty_score = float(novelty_scores[idx])
    return clips


def segment_sample(sample: dict[str, Any], clip_seconds: float) -> list[ClipInfo]:
    """
    Expected raw sample fields:
    {
      "video_id": str,
      "video_path": str,
      "duration": float,
      "clip_captions": [optional],
      "keyframes": [optional],
      "novelty_scores": [optional]
    }
    """
    clips = fixed_segment_video(
        video_id=sample["video_id"],
        video_path=sample["video_path"],
        duration=float(sample["duration"]),
        clip_seconds=clip_seconds,
    )
    return attach_lightweight_features(
        clips,
        clip_captions=sample.get("clip_captions"),
        keyframes=sample.get("keyframes"),
        novelty_scores=sample.get("novelty_scores"),
    )

