from __future__ import annotations

import json
from pathlib import Path


def export_step_jsonl(records: list[dict], output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    seek_offsets: list[int] = []
    with output_path.open("w", encoding="utf-8") as f:
        for record in records:
            seek_offsets.append(f.tell())
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    seek_path = output_path.with_name(output_path.stem + "_seeks.jsonl")
    with seek_path.open("w", encoding="utf-8") as f:
        json.dump(seek_offsets, f)

