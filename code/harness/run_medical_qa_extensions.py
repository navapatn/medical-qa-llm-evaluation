#!/usr/bin/env python3
"""Run a frozen, resumable medical-QA extension study.

The runner preserves the paper-faithful dataset universe, can reuse matching
baseline predictions, and supports information-ablation, abstention, position,
and candidate-verification tests. Runs are deterministic to plan, resumable,
auditable, and guarded by both a request cap and an observed API-cost cap.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from llm_api import build_provider_clients, stable_hash
from paper_run_tracking import freeze_run_inputs, record_cost_checkpoint
from replicate_medical_reasoning import (
    DEFAULT_SYSTEM_PROMPT,
    build_messages,
    load_dataset_bundle,
    parse_answer,
    row_cost,
    select_shots,
)
from robustness.medical_qa import CHOICE_LABELS, parse_boolean


CODEX_ROOT = Path(__file__).resolve().parent
PAPERS_ROOT = CODEX_ROOT / "papers"
WORD_RE = re.compile(r"[a-z0-9]+")
UNUSABLE_FINISH_REASONS = {"error"}

# The reconstruction condition intentionally supplies only item-level genre, not
# subject/topic metadata or any answer-bearing information. PubMedQA is omitted
# from this condition in the production protocol: its fixed yes/no/maybe options
# do not identify a particular study question.
RECONSTRUCTION_GENRE_PROMPTS = {
    "MedQA-USMLE": (
        "This was a USMLE-style medical item. It may be a clinical-management "
        "or professional-responsibility question. Reconstruct the minimal stem "
        "supported by the choices; do not invent unsupported patient details."
    ),
    "MedMCQA": (
        "This was a medical entrance-examination item. It may test clinical "
        "medicine or basic medical science. Do not assume a patient vignette or "
        "invent case details that are not supported by the choices."
    ),
}


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(errors="replace") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    )


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def tokenize(text: str) -> list[str]:
    return WORD_RE.findall(text.lower())


def token_f1(prediction: str, reference: str) -> float:
    predicted = Counter(tokenize(prediction))
    expected = Counter(tokenize(reference))
    if not predicted or not expected:
        return 0.0
    overlap = sum((predicted & expected).values())
    precision = overlap / sum(predicted.values())
    recall = overlap / sum(expected.values())
    return 2 * precision * recall / (precision + recall) if overlap else 0.0


def normalized_contains(prediction: str, reference: str) -> bool:
    pred = " ".join(tokenize(prediction))
    ref = " ".join(tokenize(reference))
    return bool(ref and (ref in pred or pred in ref))


def stratified_sample(
    examples: list[dict[str, Any]], *, count: int, seed: int
) -> list[dict[str, Any]]:
    """Sample near-equally by correct position while preserving source order."""
    if count > len(examples):
        raise ValueError(f"Requested {count} examples from a pool of {len(examples)}")
    if count == len(examples):
        # A full-split study is a census, not a sample. Returning every item in
        # source order avoids imposing an impossible balanced-label quota on an
        # intentionally imbalanced benchmark.
        return [
            {**example, "source_position": position}
            for position, example in enumerate(examples)
        ]
    buckets: dict[int, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    for position, example in enumerate(examples):
        buckets[int(example["answer_index"])].append((position, example))
    labels = sorted(buckets)
    base, remainder = divmod(count, len(labels))
    selected: list[tuple[int, dict[str, Any]]] = []
    for label_position, label in enumerate(labels):
        bucket = buckets[label][:]
        random.Random(seed + label * 1009).shuffle(bucket)
        quota = base + (1 if label_position < remainder else 0)
        if len(bucket) < quota:
            raise ValueError(f"Not enough answer-position {label} examples")
        selected.extend(bucket[:quota])
    return [
        {**example, "source_position": position}
        for position, example in sorted(selected)
    ]


def insert_idk(choices: list[str], position: int) -> tuple[list[str], list[int | None]]:
    output = list(choices)
    output.insert(position, "I don't know")
    mapping: list[int | None] = []
    original_index = 0
    for idx in range(len(output)):
        if idx == position:
            mapping.append(None)
        else:
            mapping.append(original_index)
            original_index += 1
    return output, mapping


def extension_messages(
    item: dict[str, Any], shots: list[dict[str, Any]], system_prompt: str
) -> list[dict[str, str]]:
    variant = item["variant"]
    if variant.startswith("choice_rotation_"):
        modified = {
            **item,
            "choices": item["choices"],
        }
        return build_messages(modified, shots, system_prompt=system_prompt)

    if variant in {"idk_answerable", "idk_unanswerable"}:
        idk_system = (
            "As a healthcare professional, provide an expert response. Choose "
            "the single best option when the information supports one; otherwise "
            "choose the I don't know option."
        )
        modified = {**item, "choices": item["choices"]}
        return build_messages(modified, shots, system_prompt=idk_system)

    if variant == "choices_to_question":
        choices = "\n".join(
            f"{CHOICE_LABELS[index]}. {choice}"
            for index, choice in enumerate(item["choices"])
        )
        return [{
            "role": "user",
            "content": (
                "Below are answer choices from a medical multiple-choice item, "
                "but the question stem is hidden. Reconstruct one most likely "
                "medical question for which these choices are plausible.\n\n"
                f"Item genre: {item.get('reconstruction_genre_prompt', 'This is a medical multiple-choice item.')}\n\n"
                "Output only the reconstructed question, ending with a question "
                "mark.\n\n"
                f"Choices:\n{choices}"
            ),
        }]

    if variant == "generative_no_choices":
        return [
            {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    "Answer the following medical question without seeing answer "
                    "choices. Give a concise answer and end with exactly "
                    "`Answer: <answer phrase>`.\n\n"
                    f"Question: {item['question']}"
                ),
            },
        ]

    if variant == "candidate_verification":
        return [{
            "role": "user",
            "content": (
                "Decide whether the candidate is the single best answer to the "
                "medical question. A candidate can be medically true without "
                "being the best answer. Respond with exactly True or False.\n\n"
                f"Question: {item['question']}\n\n"
                f"Candidate answer: {item['candidate']}"
            ),
        }]
    raise ValueError(f"Unknown extension variant: {variant}")


def build_extension_items(
    selected: list[dict[str, Any]],
    *,
    shots: list[dict[str, Any]],
    dataset_cfg: dict[str, Any],
    variants: set[str],
    system_prompt: str,
    variant_extra_body: dict[str, dict[str, Any]] | None = None,
    variant_max_tokens: dict[str, int] | None = None,
    choices_to_question_datasets: set[str] | None = None,
) -> list[dict[str, Any]]:
    variant_extra_body = variant_extra_body or {}
    variant_max_tokens = variant_max_tokens or {}
    items: list[dict[str, Any]] = []
    for sample_position, example in enumerate(selected):
        chosen_shots = select_shots(
            shots,
            n_shots=len(example.get("selected_shot_ids", [])) or 5,
            example_position=int(example["source_position"]),
            dataset_cfg=dataset_cfg,
        )
        common = {
            "dataset": example["dataset"],
            "example_id": example["id"],
            "question": example["question"],
            "original_choices": list(example["choices"]),
            "answer_index": int(example["answer_index"]),
            "answer_text": example["answer_text"],
            "source_position": int(example["source_position"]),
            "sample_position": sample_position,
            "shot_ids": [shot["id"] for shot in chosen_shots],
        }

        eligible_for_reconstruction = (
            choices_to_question_datasets is None
            or str(example["dataset"]) in choices_to_question_datasets
        )
        if "choices_to_question" in variants and eligible_for_reconstruction:
            choices_extra_body = variant_extra_body.get("choices_to_question")
            choices_item = {
                **common,
                "variant": "choices_to_question",
                "choices": list(example["choices"]),
                "reconstruction_genre_prompt": RECONSTRUCTION_GENRE_PROMPTS.get(
                    str(example["dataset"]),
                    "This is a medical multiple-choice item. Reconstruct only what the choices support.",
                ),
                "max_tokens": int(variant_max_tokens.get("choices_to_question", 768)),
            }
            # Only providers with an explicit compatible override receive one.
            # This keeps the reconstruction condition portable to non-reasoning
            # models while allowing model-specific frozen configurations to
            # disable native thinking for the concise question-generation task.
            if choices_extra_body is not None:
                choices_item["extra_body"] = choices_extra_body
            items.append(choices_item)

        if "generative_no_choices" in variants:
            generative_item = {
                **common,
                "variant": "generative_no_choices",
                "max_tokens": int(variant_max_tokens.get("generative_no_choices", 768)),
            }
            if "generative_no_choices" in variant_extra_body:
                generative_item["extra_body"] = variant_extra_body[
                    "generative_no_choices"
                ]
            items.append(generative_item)

        if "idk_answerable" in variants:
            position = sample_position % (len(example["choices"]) + 1)
            choices, mapping = insert_idk(example["choices"], position)
            items.append({
                **common,
                "variant": "idk_answerable",
                "choices": choices,
                "choice_mapping": mapping,
                "idk_position": position,
                "expected_letter": CHOICE_LABELS[mapping.index(example["answer_index"])],
                "max_tokens": 2048,
            })

        if "idk_unanswerable" in variants:
            position = (sample_position + 2) % (len(example["choices"]) + 1)
            choices, mapping = insert_idk(example["choices"], position)
            items.append({
                **common,
                "variant": "idk_unanswerable",
                "question": (
                    "The original clinical or research question stem has been "
                    "withheld, so there is not enough information to determine "
                    "which substantive option is best."
                ),
                "choices": choices,
                "choice_mapping": mapping,
                "idk_position": position,
                "expected_letter": CHOICE_LABELS[position],
                "max_tokens": 2048,
            })

        if "choice_rotation" in variants:
            n_choices = len(example["choices"])
            for shift in range(1, n_choices):
                permutation = list(range(shift, n_choices)) + list(range(shift))
                items.append({
                    **common,
                    "variant": f"choice_rotation_{shift}",
                    "choices": [example["choices"][index] for index in permutation],
                    "choice_mapping": permutation,
                    "expected_letter": CHOICE_LABELS[
                        permutation.index(example["answer_index"])
                    ],
                    "max_tokens": 2048,
                })

        if "candidate_verification" in variants:
            for choice_index, candidate in enumerate(example["choices"]):
                items.append({
                    **common,
                    "variant": "candidate_verification",
                    "candidate": candidate,
                    "choice_index": choice_index,
                    "expected_boolean": choice_index == example["answer_index"],
                    "max_tokens": 768,
                })

        for item in items:
            if item.get("item_id"):
                continue
            item["messages"] = extension_messages(item, chosen_shots, system_prompt)
            item["item_id"] = stable_hash({
                "dataset": item["dataset"],
                "example_id": item["example_id"],
                "variant": item["variant"],
                "choices": item.get("choices"),
                "candidate": item.get("candidate"),
                "choice_index": item.get("choice_index"),
                "choice_mapping": item.get("choice_mapping"),
                "idk_position": item.get("idk_position"),
                "reconstruction_genre_prompt": item.get("reconstruction_genre_prompt"),
            })[:20]
    return items


def shard_extension_plan(
    items: list[dict[str, Any]],
    selections: list[dict[str, Any]],
    *,
    shard_count: int,
    shard_index: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Partition a frozen full plan without changing any item payload."""

    if shard_count < 1:
        raise ValueError("item_shard_count must be at least 1")
    if not 0 <= shard_index < shard_count:
        raise ValueError(
            "item_shard_index must be between 0 and item_shard_count - 1"
        )
    if shard_count == 1:
        return items, selections

    sharded_items = [
        item for position, item in enumerate(items)
        if position % shard_count == shard_index
    ]
    selected_keys = {
        (str(item["dataset"]), str(item["example_id"]))
        for item in sharded_items
    }
    sharded_selections = [
        row for row in selections
        if (str(row["dataset"]), str(row["id"])) in selected_keys
    ]
    return sharded_items, sharded_selections


def extract_answer_phrase(output: str) -> str:
    matches = re.findall(r"(?im)^\s*answer\s*:\s*(.+?)\s*$", output)
    return matches[-1].strip() if matches else output.strip()


def score_output(item: dict[str, Any], output: str) -> dict[str, Any]:
    variant = item["variant"]
    result: dict[str, Any] = {
        "correct": None,
        "abstained": False,
        "parsed_answer": None,
        "predicted_original_index": None,
    }
    if variant.startswith("choice_rotation_") or variant.startswith("idk_"):
        letter = parse_answer(output, item["choices"])
        result["parsed_answer"] = letter
        if letter:
            displayed_index = CHOICE_LABELS.index(letter)
            mapping = item["choice_mapping"]
            result["predicted_original_index"] = mapping[displayed_index]
            result["abstained"] = mapping[displayed_index] is None
        result["correct"] = letter == item["expected_letter"] if letter else False
        return result

    if variant == "candidate_verification":
        parsed = parse_boolean(output)
        result["parsed_answer"] = parsed
        result["correct"] = (
            parsed == item["expected_boolean"] if parsed is not None else False
        )
        return result

    if variant == "generative_no_choices":
        phrase = extract_answer_phrase(output)
        scores = [token_f1(phrase, choice) for choice in item["original_choices"]]
        best = max(scores) if scores else 0.0
        winners = [index for index, score in enumerate(scores) if score == best and score > 0]
        option_index = winners[0] if len(winners) == 1 else None
        strict = normalized_contains(phrase, item["answer_text"])
        result.update({
            "parsed_answer": phrase,
            "strict_correct": strict,
            "option_recovery_correct": option_index == item["answer_index"],
            "correct": strict or option_index == item["answer_index"],
            "answer_token_f1": token_f1(phrase, item["answer_text"]),
            "predicted_original_index": option_index,
        })
        return result

    if variant == "choices_to_question":
        result.update({
            "parsed_answer": output.strip(),
            "question_token_f1": token_f1(output, item["question"]),
            "question_term_recall": token_f1(item["question"], output),
        })
        return result
    raise ValueError(f"Unknown extension variant: {variant}")


def latest_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return the first successful observation for each planned request.

    Successful rows are final under the runner's resume contract. If an
    interrupted concurrent worker later appends a duplicate success, that late
    row remains in the raw audit log (and its charge remains reportable) but it
    must not replace the experimental observation. A success does supersede an
    earlier error; repeated errors retain their latest diagnostic.
    """
    latest: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        key = (str(row.get("model")), str(row.get("item_id")))
        previous = latest.get(key)
        if previous is None:
            latest[key] = row
            continue
        # A transport-level ``stop`` with no visible answer is not a usable
        # generation. Treat it as recoverable so a later bounded recovery can
        # supersede it rather than being hidden behind an apparently successful
        # earlier API response.
        previous_success = (
            not previous.get("api_error")
            and not previous.get("truncated", False)
            and previous.get("finish_reason") not in UNUSABLE_FINISH_REASONS
            and bool(str(previous.get("output") or "").strip())
        )
        row_success = (
            not row.get("api_error")
            and not row.get("truncated", False)
            and row.get("finish_reason") not in UNUSABLE_FINISH_REASONS
            and bool(str(row.get("output") or "").strip())
        )
        if row_success and not previous_success:
            latest[key] = row
        elif not row_success and not previous_success and str(
            row.get("created_at", "")
        ) >= str(previous.get("created_at", "")):
            latest[key] = row
    return list(latest.values())


def load_baseline_rows(
    baseline_dir: Path, selected_keys: set[tuple[str, str]], model_id: str
) -> list[dict[str, Any]]:
    latest: dict[tuple[str, str], dict[str, Any]] = {}
    for path in sorted(baseline_dir.rglob("predictions.jsonl")):
        for row in read_jsonl(path):
            key = (str(row.get("dataset")), str(row.get("example_id")))
            if key not in selected_keys or row.get("model") != model_id:
                continue
            if row.get("api_error") or row.get("truncated"):
                continue
            previous = latest.get(key)
            if previous is None or str(row.get("created_at", "")) >= str(previous.get("created_at", "")):
                latest[key] = row
    missing = sorted(selected_keys - set(latest))
    if missing:
        raise RuntimeError(
            f"Missing {len(missing)} reusable baseline rows for {model_id}; "
            f"first={missing[:3]}"
        )
    return [latest[key] for key in sorted(latest)]


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def summarize(
    rows: list[dict[str, Any]], baseline_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    successful = [
        row
        for row in latest_rows(rows)
        if not row.get("api_error")
        and not row.get("truncated", False)
        and row.get("finish_reason") not in UNUSABLE_FINISH_REASONS
    ]
    by_variant: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in successful:
        by_variant[row["variant"]].append(row)

    variants = []
    for variant, group in sorted(by_variant.items()):
        scored = [row for row in group if row.get("correct") is not None]
        variants.append({
            "variant": variant,
            "n": len(group),
            "accuracy": mean([float(row["correct"]) for row in scored]),
            "abstention_rate": mean([float(row.get("abstained", False)) for row in group]),
            "strict_accuracy": mean([
                float(row["strict_correct"]) for row in group if row.get("strict_correct") is not None
            ]),
            "option_recovery_accuracy": mean([
                float(row["option_recovery_correct"])
                for row in group if row.get("option_recovery_correct") is not None
            ]),
            "mean_question_token_f1": mean([
                float(row["question_token_f1"])
                for row in group if row.get("question_token_f1") is not None
            ]),
        })

    baseline_map = {
        (row["dataset"], row["example_id"]): row for row in baseline_rows
    }
    rotations: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in successful:
        if row["variant"].startswith("choice_rotation_"):
            rotations[(row["dataset"], row["example_id"])].append(row)
    invariant = []
    all_rotation_correct = []
    for key, group in rotations.items():
        baseline = baseline_map[key]
        baseline_index = (
            CHOICE_LABELS.index(baseline["parsed_answer"])
            if baseline.get("parsed_answer") in CHOICE_LABELS
            else None
        )
        predictions = [baseline_index] + [row.get("predicted_original_index") for row in group]
        invariant.append(None not in predictions and len(set(predictions)) == 1)
        all_rotation_correct.append(all(row.get("correct") is True for row in group))

    tf_by_question: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in by_variant.get("candidate_verification", []):
        tf_by_question[(row["dataset"], row["example_id"])].append(row)
    tf_exact = []
    tf_correct_true = []
    tf_distractor_false = []
    for group in tf_by_question.values():
        tf_exact.append(all(row.get("correct") is True for row in group))
        tf_correct_true.extend(
            row.get("parsed_answer") is True
            for row in group if row.get("expected_boolean") is True
        )
        tf_distractor_false.extend(
            row.get("parsed_answer") is False
            for row in group if row.get("expected_boolean") is False
        )

    return {
        "baseline": {
            "n": len(baseline_rows),
            "accuracy": mean([float(row.get("correct", False)) for row in baseline_rows]),
        },
        "variants": variants,
        "position_robustness": {
            "questions": len(rotations),
            "fully_invariant_rate_including_baseline": mean([float(value) for value in invariant]),
            "all_new_rotations_correct_rate": mean([float(value) for value in all_rotation_correct]),
        },
        "candidate_verification": {
            "questions": len(tf_by_question),
            "exact_question_accuracy": mean([float(value) for value in tf_exact]),
            "correct_candidate_sensitivity": mean([float(value) for value in tf_correct_true]),
            "distractor_specificity": mean([float(value) for value in tf_distractor_false]),
        },
    }


def write_summary(path: Path, manifest: dict[str, Any], metrics: dict[str, Any]) -> None:
    def pct(value: float | None) -> str:
        return "NA" if value is None else f"{100 * value:.1f}%"

    lines = [
        "# Medical-QA Extension Study",
        "",
        f"Generated: `{manifest['generated_at']}`",
        f"Sample: {manifest['selected_questions']} questions",
        f"Completed extension calls: {manifest['completed_requests']}/{manifest['planned_requests']}",
        f"Observed extension cost: `${manifest['observed_cost_usd']:.4f}`",
        f"Total API charges including duplicate audit rows: `${manifest.get('raw_api_charges_usd', manifest['observed_cost_usd']):.4f}`",
        "",
        "## Main results",
        "",
        f"- Reused paper-faithful Qwen baseline accuracy: {pct(metrics['baseline']['accuracy'])}.",
        f"- Fully position-invariant across baseline and every rotation: {pct(metrics['position_robustness']['fully_invariant_rate_including_baseline'])}.",
        f"- Every new rotation answered correctly: {pct(metrics['position_robustness']['all_new_rotations_correct_rate'])}.",
        f"- Candidate-verification exact question accuracy: {pct(metrics['candidate_verification']['exact_question_accuracy'])}.",
        f"- Correct-candidate sensitivity: {pct(metrics['candidate_verification']['correct_candidate_sensitivity'])}.",
        f"- Distractor specificity: {pct(metrics['candidate_verification']['distractor_specificity'])}.",
        "",
        "## Variant-level metrics",
        "",
        "| Variant | N | Accuracy | Abstention | Strict generative | Option recovery | Question token F1 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in metrics["variants"]:
        f1 = "NA" if row["mean_question_token_f1"] is None else f"{row['mean_question_token_f1']:.3f}"
        lines.append(
            f"| {row['variant']} | {row['n']} | {pct(row['accuracy'])} | "
            f"{pct(row['abstention_rate'])} | {pct(row['strict_accuracy'])} | "
            f"{pct(row['option_recovery_accuracy'])} | {f1} |"
        )
    path.write_text("\n".join(lines).rstrip() + "\n")


def checkpoint(
    *,
    config: dict[str, Any],
    items: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    baseline_rows: list[dict[str, Any]],
    out_dir: Path,
    model: dict[str, Any],
    stopped_for_cost: bool = False,
) -> dict[str, Any]:
    latest = latest_rows(rows)
    cost = sum(row_cost(row, model) for row in latest if not row.get("api_error"))
    raw_api_charges = sum(
        row_cost(row, model) for row in rows if not row.get("api_error")
    )
    metrics = summarize(latest, baseline_rows)
    write_json(out_dir / "metrics.json", metrics)
    write_jsonl(out_dir / "baseline_predictions.jsonl", baseline_rows)
    manifest = {
        "generated_at": now_iso(),
        "condition": config.get("condition"),
        "config_sha256": stable_hash(config),
        "model": {key: model.get(key) for key in ["label", "provider", "model", "temperature"]},
        "selected_questions": len({
            (str(item["dataset"]), str(item["example_id"])) for item in items
        }),
        "baseline_available": bool(baseline_rows),
        "planned_requests": len(items),
        "completed_requests": sum(
            1
            for row in latest
            if not row.get("api_error")
            and not row.get("truncated", False)
            and row.get("finish_reason") not in UNUSABLE_FINISH_REASONS
        ),
        "observed_cost_usd": cost,
        "raw_api_charges_usd": raw_api_charges,
        "duplicate_audit_rows": len(rows) - len(latest),
        "max_cost_usd": float(config.get("max_cost_usd", 5)),
        "stopped_for_cost": stopped_for_cost,
        "variant_counts": dict(sorted(Counter(item["variant"] for item in items).items())),
        "dataset_counts": dict(sorted(Counter(row["dataset"] for row in baseline_rows).items())),
        "outputs": {
            "planned_items": str(out_dir / "planned_items.jsonl"),
            "predictions": str(out_dir / "predictions.jsonl"),
            "baseline_predictions": str(out_dir / "baseline_predictions.jsonl"),
            "metrics": str(out_dir / "metrics.json"),
            "summary": str(out_dir / "summary.md"),
            "config_snapshot": str(out_dir / "config.snapshot.json"),
            "frozen_run_inputs": str(out_dir / "frozen_run_inputs.json"),
            "cost_checkpoints_jsonl": str(out_dir / "cost_checkpoints.jsonl"),
            "cost_checkpoints_csv": str(out_dir / "cost_checkpoints.csv"),
        },
    }
    write_json(out_dir / "run_manifest.json", manifest)
    write_summary(out_dir / "summary.md", manifest, metrics)
    return manifest


def run(
    config: dict[str, Any],
    *,
    paper_root: Path,
    out_dir: Path,
    execute_api: bool,
    max_new_requests: int | None = None,
) -> dict[str, Any]:
    models = [model for model in config.get("models", []) if model.get("enabled", True)]
    if len(models) != 1:
        raise RuntimeError("Each extension run requires exactly one enabled model.")
    model = models[0]
    system_prompt = str(config.get("system_prompt", DEFAULT_SYSTEM_PROMPT))
    variants = set(config["variants"])
    reconstruction_datasets = config.get("choices_to_question_datasets")
    if reconstruction_datasets is not None:
        reconstruction_datasets = {str(name) for name in reconstruction_datasets}
    all_items: list[dict[str, Any]] = []
    selections: list[dict[str, Any]] = []

    for dataset_index, dataset_cfg in enumerate(config["datasets"]):
        source_cfg = {key: value for key, value in dataset_cfg.items() if key not in {"extension_sample_size"}}
        examples, shots, _ = load_dataset_bundle(
            paper_root, source_cfg, int(config.get("n_shots", 5))
        )
        selected = stratified_sample(
            examples,
            count=int(dataset_cfg["extension_sample_size"]),
            seed=int(config.get("selection_seed", 0)) + dataset_index * 100_003,
        )
        selections.extend(selected)
        all_items.extend(build_extension_items(
            selected,
            shots=shots,
            dataset_cfg=source_cfg,
            variants=variants,
            system_prompt=system_prompt,
            variant_extra_body=config.get("variant_extra_body"),
            variant_max_tokens=config.get("variant_max_tokens"),
            choices_to_question_datasets=reconstruction_datasets,
        ))

    all_items, selections = shard_extension_plan(
        all_items,
        selections,
        shard_count=int(config.get("item_shard_count", 1)),
        shard_index=int(config.get("item_shard_index", 0)),
    )
    selected_keys = {
        (str(item["dataset"]), str(item["example_id"]))
        for item in all_items
    }

    max_requests = int(config.get("max_requests", 2660))
    if len(all_items) > max_requests:
        raise RuntimeError(f"Plan has {len(all_items)} calls, exceeding max_requests={max_requests}")
    if len({item["item_id"] for item in all_items}) != len(all_items):
        raise RuntimeError("Extension item IDs are not unique")

    baseline_dir = paper_root / config["baseline_run_dir"]
    try:
        baseline_rows = load_baseline_rows(
            baseline_dir, selected_keys, model["model"]
        )
    except RuntimeError:
        # The two information-ablation variants can be generated and judged
        # without a reusable multiple-choice baseline. This opt-in is for a
        # legacy source run with incomplete main predictions; its extension
        # analysis records the absent baseline rather than silently mixing in
        # a recovery under a different decoding condition.
        if not config.get("allow_incomplete_baseline", False):
            raise
        baseline_rows = []
    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / "planned_items.jsonl", all_items)
    write_jsonl(out_dir / "selected_questions.jsonl", selections)
    freeze_run_inputs(
        out_dir=out_dir,
        config=config,
        planned_items_path=out_dir / "planned_items.jsonl",
        source_paths=[
            Path(__file__),
            CODEX_ROOT / "llm_api.py",
            CODEX_ROOT / "paper_run_tracking.py",
        ],
        study_stage="extension_generation",
    )

    providers = build_provider_clients(config["providers"], out_dir / "api_cache")
    client = providers[model["provider"]]
    predictions_path = out_dir / "predictions.jsonl"
    rows = read_jsonl(predictions_path)
    completed = {
        (row.get("model"), row.get("item_id"))
        for row in latest_rows(rows)
        if not row.get("api_error")
        and not row.get("truncated", False)
        and row.get("finish_reason") not in UNUSABLE_FINISH_REASONS
    }
    pending = [item for item in all_items if (model["model"], item["item_id"]) not in completed]
    if max_new_requests is not None:
        if max_new_requests < 1:
            raise ValueError("max_new_requests must be positive")
        pending = pending[:max_new_requests]
    if not execute_api:
        return checkpoint(
            config=config, items=all_items, rows=rows, baseline_rows=baseline_rows,
            out_dir=out_dir, model=model,
        )

    max_cost = float(config.get("max_cost_usd", 5))
    workers = int(config.get("max_workers", 8))
    progress_every = int(config.get("progress_every", 100))
    stopped_for_cost = False

    def call(item: dict[str, Any]) -> dict[str, Any]:
        try:
            output, meta = client.complete(
                model=model["model"],
                messages=item["messages"],
                temperature=float(model.get("temperature", 0.5)),
                max_tokens=int(item["max_tokens"]),
                max_tokens_field=str(model.get("max_tokens_field", "max_tokens")),
                extra_body=item.get("extra_body", model.get("extra_body")),
            )
            finish_reason = meta.get("finish_reason")
            truncated = finish_reason in {"length", "max_tokens"}
            score = (
                {"correct": None, "abstained": False, "parsed_answer": None}
                if truncated
                else score_output(item, output)
            )
            error = None
        except Exception as exc:  # noqa: BLE001
            output = ""
            meta = {"usage": {}, "cache_hit": False, "finish_reason": None}
            score = {"correct": None, "abstained": False, "parsed_answer": None}
            truncated = False
            error = str(exc)
        return {
            "created_at": now_iso(),
            "provider": model["provider"],
            "model_label": model.get("label") or model["model"],
            "model": model["model"],
            **{key: item.get(key) for key in [
                "dataset", "example_id", "item_id", "variant", "answer_index",
                "answer_text", "expected_letter", "expected_boolean", "choice_index",
                "idk_position", "choice_mapping",
            ]},
            "output": output,
            "reasoning": meta.get("reasoning", ""),
            "reasoning_details": meta.get("reasoning_details"),
            "assistant_message": meta.get("assistant_message"),
            **score,
            "usage": meta.get("usage", {}),
            "cache_hit": meta.get("cache_hit", False),
            "finish_reason": meta.get("finish_reason"),
            "response_id": meta.get("response_id"),
            "response_model": meta.get("response_model"),
            "truncated": truncated,
            "api_error": error,
        }

    index = 0
    while index < len(pending):
        current_cost = sum(
            row_cost(row, model) for row in latest_rows(rows) if not row.get("api_error")
        )
        if current_cost >= max_cost:
            stopped_for_cost = True
            break
        batch = pending[index:index + workers]
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(call, item) for item in batch]
            for future in as_completed(futures):
                row = future.result()
                rows.append(row)
                append_jsonl(predictions_path, row)
                if (
                    not row.get("api_error")
                    and not row.get("truncated", False)
                    and row.get("finish_reason") not in UNUSABLE_FINISH_REASONS
                ):
                    completed.add((row.get("model"), row.get("item_id")))
                if (
                    not row.get("api_error")
                    and not row.get("truncated", False)
                    and row.get("finish_reason") not in UNUSABLE_FINISH_REASONS
                    and len(completed) > 0
                    and len(completed) % progress_every == 0
                ):
                    manifest = checkpoint(
                        config=config,
                        items=all_items,
                        rows=rows,
                        baseline_rows=baseline_rows,
                        out_dir=out_dir,
                        model=model,
                    )
                    record_cost_checkpoint(
                        out_dir=out_dir,
                        study_stage="extension_generation",
                        planned_rows=len(all_items),
                        raw_rows=rows,
                        canonical_rows=latest_rows(rows),
                        cost_fn=lambda prediction: row_cost(prediction, model),
                        interval=progress_every,
                    )
                    print(
                        f"[medical-extensions] progress="
                        f"{manifest['completed_requests']}/{manifest['planned_requests']} "
                        f"cost=${manifest['observed_cost_usd']:.4f}",
                        flush=True,
                    )
        index += len(batch)

    manifest = checkpoint(
        config=config, items=all_items, rows=rows, baseline_rows=baseline_rows,
        out_dir=out_dir, model=model, stopped_for_cost=stopped_for_cost,
    )
    record_cost_checkpoint(
        out_dir=out_dir,
        study_stage="extension_generation",
        planned_rows=len(all_items),
        raw_rows=rows,
        canonical_rows=latest_rows(rows),
        cost_fn=lambda prediction: row_cost(prediction, model),
        interval=progress_every,
        force_final=True,
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--paper-id", default="cellpress-medical-reasoning")
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--execute-api", action="store_true")
    parser.add_argument("--max-cost-usd", type=float)
    parser.add_argument(
        "--choices-to-question-datasets",
        nargs="+",
        help="Dataset names eligible for the choices-to-question condition.",
    )
    parser.add_argument(
        "--max-new-requests",
        type=int,
        help="Execute at most this many currently pending calls, then checkpoint.",
    )
    parser.add_argument(
        "--item-shard-count",
        type=int,
        help="Partition the fully constructed extension plan into modulo shards.",
    )
    parser.add_argument(
        "--item-shard-index",
        type=int,
        help="Zero-based modulo shard index selected by --item-shard-count.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = read_json(args.config)
    if args.max_cost_usd is not None:
        config["max_cost_usd"] = args.max_cost_usd
    if args.choices_to_question_datasets is not None:
        config["choices_to_question_datasets"] = args.choices_to_question_datasets
    if args.item_shard_count is not None:
        config["item_shard_count"] = args.item_shard_count
    if args.item_shard_index is not None:
        config["item_shard_index"] = args.item_shard_index
    paper_root = PAPERS_ROOT / args.paper_id
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_dir or paper_root / "runs" / f"medical_extensions_{stamp}"
    manifest = run(
        config,
        paper_root=paper_root,
        out_dir=out_dir,
        execute_api=args.execute_api,
        max_new_requests=args.max_new_requests,
    )
    print(
        f"[medical-extensions] mode={'api' if args.execute_api else 'plan'} "
        f"questions={manifest['selected_questions']} calls={manifest['planned_requests']} "
        f"cost=${manifest['observed_cost_usd']:.4f}"
    )
    print(f"[medical-extensions] summary={manifest['outputs']['summary']}")


if __name__ == "__main__":
    main()
