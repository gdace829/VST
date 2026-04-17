#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from streaming_vlm.sft_builder.export import export_step_jsonl
from streaming_vlm.sft_builder.rollout import rollout_video


def load_raw_samples(path: str) -> list[dict]:
    path_obj = Path(path)
    if path_obj.suffix == ".json":
        return json.loads(path_obj.read_text(encoding="utf-8"))
    if path_obj.suffix == ".jsonl":
        with path_obj.open("r", encoding="utf-8") as f:
            return [json.loads(line) for line in f]
    raise ValueError(f"Unsupported input file: {path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build write-trajectory SFT JSONL from raw video+QA samples.")
    parser.add_argument("--input", type=str, required=True, help="Raw sample json/jsonl path.")
    parser.add_argument("--output", type=str, required=True, help="Output step-level write trajectory jsonl.")
    parser.add_argument("--clip_seconds", type=float, default=4.0)
    parser.add_argument("--storage_budget", type=int, default=100)
    args = parser.parse_args()

    raw_samples = load_raw_samples(args.input)
    all_records: list[dict] = []
    for sample in raw_samples:
        all_records.extend(
            rollout_video(
                sample,
                clip_seconds=args.clip_seconds,
                storage_budget=args.storage_budget,
            )
        )

    export_step_jsonl(all_records, args.output)
    print(f"Exported {len(all_records)} steps to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
