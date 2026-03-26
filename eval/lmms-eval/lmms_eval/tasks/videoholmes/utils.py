import os
import re
from typing import Dict
from loguru import logger as eval_logger


DEFAULT_VIDEO_PATH = "/path/to/video_holmes/videos"  # modify this path !!!


def video_holmes_doc_to_visual(doc, lmms_eval_specific_kwargs=None):
    """Build visual inputs from one sample."""
    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {}

    video_path = lmms_eval_specific_kwargs.get("video_path", DEFAULT_VIDEO_PATH)
    video_id = doc.get("video")
    full_video_path = os.path.join(video_path, video_id)

    if not os.path.exists(full_video_path):
        eval_logger.warning(f"Video path: {full_video_path} does not exist.")

    media_dict = {
        "video_read_type": "decord",
    }

    return [full_video_path, media_dict]


def video_holmes_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    """Build the task prompt from question and candidate options."""
    question = doc.get("question")
    candidates = doc.get("candidates", [])

    options_list = []
    for idx, cand in enumerate(candidates):
        letter = chr(65 + idx)
        options_list.append(f"{letter}. {cand}")

    options_str = "\n".join(options_list)

    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {}

    cot_prompt = lmms_eval_specific_kwargs.get(
        "cot_prompt",
        "Based on the given video, reason and answer the single-choice question. Provide your reasoning between the <think> and </think> tags, and then give your final answer between the <answer> and </answer> tags.",
    )
    format_template = lmms_eval_specific_kwargs.get(
        "format_template",
        "The question is: {question}\nThe options are:\n{options}\nYour answer:",
    )

    formatted_question = format_template.format(question=question, options=options_str)
    full_prompt = f"{cot_prompt} {formatted_question}"
    return full_prompt


def video_holmes_process_results(doc, results):
    """Process one prediction and return a normalized scoring record."""
    pred = results[0]

    gt_text = doc.get("answer")
    candidates = doc.get("candidates", [])
    subtask = doc.get("subtask")

    correct_answer_letter = None
    try:
        gt_index = candidates.index(gt_text)
        correct_answer_letter = chr(65 + gt_index)
    except ValueError:
        print(f"Warning: GT text '{gt_text}' not found in candidates: {candidates}")
        correct_answer_letter = "UNKNOWN"

    pattern = r"<answer>\s*(.*?)\s*</answer>"
    try:
        matches = re.findall(pattern, pred, re.DOTALL)
    except Exception:
        matches = []

    if matches:
        choice = matches[-1].strip()
    else:
        choice = pred.strip()

    predicted_answer = "WRONG"

    found_pred = False
    for i in range(len(candidates)):
        letter = chr(65 + i)
        if f"{letter} " in choice or f"{letter}:" in choice or f"[{letter}" in choice:
            predicted_answer = letter
            found_pred = True
            break

    if not found_pred:
        for i in range(len(candidates)):
            letter = chr(65 + i)
            if letter in choice:
                predicted_answer = letter
                break

    score = 1 if predicted_answer == correct_answer_letter else 0

    data_dict = {
        "pred_answer": predicted_answer,
        "gt_answer": correct_answer_letter,
        "score": score,
        "question_type": subtask,
        "raw_pred": pred,
    }

    return {"video_holmes_accuracy": data_dict}


def video_holmes_aggregate_results(results):
    """Aggregate per-sample results and print per-subtask metrics."""
    total_answered = 0
    total_correct = 0

    type_stats: Dict[str, Dict[str, int]] = {}

    for result in results:
        total_answered += 1
        score = result["score"]
        total_correct += score
        q_type = result["question_type"]

        if q_type not in type_stats:
            type_stats[q_type] = {"count": 0, "correct": 0}

        type_stats[q_type]["count"] += 1
        type_stats[q_type]["correct"] += score

    overall_acc = 100 * total_correct / total_answered if total_answered > 0 else 0
    print(f"\n{'='*20} Video Holmes Results {'='*20}")
    print(f"Total Instances: {total_answered}")
    print(f"Total Correct:   {total_correct}")
    print(f"Overall Accuracy: {overall_acc:.2f}%")
    print(f"{'-'*60}")

    sorted_types = sorted(type_stats.keys())

    for q_type in sorted_types:
        stats = type_stats[q_type]
        count = stats["count"]
        correct = stats["correct"]
        acc = (correct / count) * 100 if count > 0 else 0
        print(f"Subtask [{q_type}]: Instances={count}, Correct={correct}, Accuracy={acc:.2f}%")

    print(f"{'='*60}\n")

    return overall_acc