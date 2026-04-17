from __future__ import annotations

from dataclasses import dataclass
import json
import os

import torch
from torch.utils.data import Dataset
from transformers import AutoProcessor
from qwen_vl_utils.vision_process import process_vision_info

from streaming_vlm.utils.get_qwen_range import get_qwen_range


@dataclass
class WriteTrajDataArguments:
    train_write_traj_paths: list[str] | None = None
    eval_write_traj_paths: list[str] | None = None
    text_sink: int = 0
    text_sliding_window: int = 0


class WriteTrajectoryDataset(Dataset):
    """
    Minimal SFT dataset for:
    (state -> teacher_action + write_content)

    This mimics the existing lmm_dataset.py style:
    - read one JSONL record by seek offset
    - convert one record into a chat-style prompt
    - build NTP labels only on assistant spans
    """

    def __init__(
        self,
        *,
        train_write_traj_paths: list[str] | None = None,
        eval_write_traj_paths: list[str] | None = None,
        processor: AutoProcessor,
        return_conversation: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.processor = processor
        self.return_conversation = return_conversation
        self.handles: list[tuple[str, int]] = []
        self.text_sink = kwargs.get("text_sink", 0)
        self.text_sliding_window = kwargs.get("text_sliding_window", 0)

        annotation_paths = eval_write_traj_paths if eval_write_traj_paths is not None else train_write_traj_paths
        if not annotation_paths:
            raise ValueError("No write trajectory paths provided.")

        for annotation_path in annotation_paths:
            annotation_path = str(annotation_path)
            if not annotation_path.endswith(".jsonl"):
                raise ValueError(f"Expected .jsonl file, got: {annotation_path}")
            stem = annotation_path.rsplit(".jsonl", 1)[0]
            seek_path = stem + "_seeks.jsonl"
            with open(seek_path, "r", encoding="utf-8") as f:
                seeks = json.load(f)
            self.handles.extend((annotation_path, seek) for seek in seeks)

        self.im_start_id, self.assistant_id, self.newline_id, self.im_end_id = (
            processor.tokenizer("<|im_start|>assistant\n<|im_end|>").input_ids
        )
        self.get_range = get_qwen_range

    def __len__(self) -> int:
        return len(self.handles)

    def load_record(self, index: int) -> dict:
        annotation_path, seek = self.handles[index]
        with open(annotation_path, "r", encoding="utf-8") as f:
            f.seek(seek)
            return json.loads(f.readline())

    def build_state_text(self, record: dict) -> str:
        clip = record["clip"]
        budget = record["budget_state"]
        meta = record.get("meta", {})
        return (
            "Current observation and decision context:\n"
            f"Current clip: {clip['start']:.1f}-{clip['end']:.1f}s\n"
            f"Clip caption: {clip.get('caption', '')}\n"
            "\nBudget state:\n"
            f"- storage_left: {budget.get('storage_left', 0)}\n"
            f"- retrieval_left: {budget.get('retrieval_left', 0)}\n"
            "\nTeacher-side diagnostic signals:\n"
            f"Teacher utility score: {meta.get('utility_score', 0.0)}\n"
            f"Teacher evidence type: {meta.get('evidence_type', 'context')}\n"
            "\nTask: decide the write-time memory action and generate the memory update."
        )

    def build_previous_text(self, record: dict) -> str:
        mem = record["memory_state"]
        previous_lines: list[str] = [
            "Serialized hierarchical memory state.",
            "This is an implementation container for memory slots, not a VST-style textual thought chain.",
            "",
            "[summary_memory]",
        ]
        if mem.get("summary_history"):
            previous_lines.extend(mem["summary_history"])
        else:
            previous_lines.append("<empty>")
        previous_lines.append("")
        previous_lines.append("[structured_memory]")
        if mem.get("structured_history"):
            previous_lines.extend(mem["structured_history"])
        else:
            previous_lines.append("<empty>")
        previous_lines.append("")
        previous_lines.append("[visual_memory]")
        if mem.get("visual_history"):
            previous_lines.extend(mem["visual_history"])
        else:
            previous_lines.append("<empty>")
        return "\n".join(previous_lines).strip()

    def build_target_text(self, record: dict) -> str:
        memory_call = record.get("memory_call")
        if memory_call:
            return (
                "<memory_call>\n"
                + json.dumps(memory_call, ensure_ascii=False, indent=2)
                + "\n</memory_call>"
            )

        # Backward-compatible fallback for early debug JSONL files.
        effort = record.get("teacher_effort", "")
        action = record["teacher_action"]
        write_content = record.get("write_content", "")
        fallback_call = {
            "effort": effort,
            "action": action,
            "memory_level": "unknown",
            "content": write_content,
            "cost": None,
        }
        return (
            "<memory_call>\n"
            + json.dumps(fallback_call, ensure_ascii=False, indent=2)
            + "\n</memory_call>"
        )

    def build_conversation(self, record: dict) -> list[dict]:
        clip = record["clip"]
        video_path = clip["video"]
        if not os.path.exists(video_path):
            dataset_root = os.environ.get("DATASET_PATH", "")
            candidate = os.path.join(dataset_root, video_path)
            if os.path.exists(candidate):
                video_path = candidate

        content = [
            {
                "type": "video",
                "video": video_path,
                "video_start": clip["start"],
                "video_end": clip["end"],
            },
            {
                "type": "text",
                "text": self.build_state_text(record),
            },
        ]
        return [{"role": "previous text", "content": self.build_previous_text(record)}] + [
            {"role": "user", "content": content},
            {"role": "assistant", "content": [{"type": "text", "text": self.build_target_text(record)}]},
        ]

    def __getitem__(self, index: int):
        record = self.load_record(index)
        conversation = self.build_conversation(record)
        text = self.processor.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=False,
            return_tensors="pt",
        )
        image_inputs, video_inputs = process_vision_info(conversation[1:])

        inputs = self.processor(
            text=text,
            images=image_inputs,
            videos=video_inputs,
            return_tensors="pt",
        )

        if self.text_sink != 0 or self.text_sliding_window != 0:
            previous_text_start_idx, previous_text_end_idx = self.get_range(
                inputs.input_ids,
                "previous text",
                0,
                contain_lf=True,
            )
            need_truncate = (
                previous_text_start_idx + self.text_sink + self.text_sliding_window
                <= previous_text_end_idx + 1
            )
            if need_truncate:
                truncate_start = previous_text_start_idx + self.text_sink
                truncate_end = previous_text_end_idx - self.text_sliding_window
                inputs["input_ids"] = torch.cat(
                    [inputs.input_ids[:, :truncate_start], inputs.input_ids[:, truncate_end + 1 :]],
                    dim=1,
                ).contiguous()
                if "attention_mask" in inputs:
                    inputs["attention_mask"] = torch.cat(
                        [
                            inputs.attention_mask[:, :truncate_start],
                            inputs.attention_mask[:, truncate_end + 1 :],
                        ],
                        dim=1,
                    ).contiguous()

        input_ids = inputs.input_ids
        attention_mask = inputs.attention_mask

        labels = torch.full_like(input_ids, fill_value=-100, dtype=input_ids.dtype)
        im_start_idxs = (input_ids == self.im_start_id).nonzero()
        im_end_idxs = (input_ids == self.im_end_id).nonzero()
        for (sample_idx, im_start_idx), (_, im_end_idx) in zip(im_start_idxs, im_end_idxs):
            if input_ids[sample_idx, im_start_idx + 1] == self.assistant_id:
                labels[sample_idx, im_start_idx + 3 : im_end_idx + 1] = input_ids[sample_idx, im_start_idx + 3 : im_end_idx + 1]

        out = dict(inputs)
        out["labels"] = labels
        if self.return_conversation:
            out["conversation"] = conversation
            out["record"] = record
        return out

    def data_collator(self, batched_inputs, **kwargs):
        assert len(batched_inputs) == 1, "batch size must be 1"
        return batched_inputs[0]
