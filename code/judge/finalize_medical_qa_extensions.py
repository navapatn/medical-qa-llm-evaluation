#!/usr/bin/env python3
"""Validate paper extensions and create analysis- and judge-ready artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from llm_api import stable_hash
from paper_run_tracking import file_sha256, usage_counts
from replicate_medical_reasoning import row_cost
from run_medical_qa_extensions import latest_rows


UNUSABLE_FINISH_REASONS = {"error"}


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
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    )


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def is_usable(row: dict[str, Any]) -> bool:
    return (
        not row.get("api_error")
        and not row.get("truncated", False)
        and row.get("finish_reason") not in UNUSABLE_FINISH_REASONS
        and bool(str(row.get("output") or "").strip())
    )


def reconstruction_judge_messages(item: dict[str, Any], row: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": "Return only valid JSON.",
        },
        {
            "role": "user",
            "content": (
                "Do the reference and generated questions test the same specific "
                "medical knowledge? Allow paraphrases. Same broad topic is not enough.\n"
                f"Reference: {item['question']}\n"
                f"Generated: {row['output'].strip()}\n"
                '{"same": true}'
            ),
        },
    ]


def answer_semantic_judge_messages(item: dict[str, Any], row: dict[str, Any]) -> list[dict[str, str]]:
    answer_phrase = str(row.get("parsed_answer") or row.get("output") or "").strip()
    return [
        {
            "role": "system",
            "content": "Return only valid JSON.",
        },
        {
            "role": "user",
            "content": (
                "Does the response give the same substantive answer as the reference? "
                "Allow equivalent wording. Mark false if incomplete or materially different.\n"
                f"Question: {item['question']}\n"
                f"Reference answer: {item['answer_text']}\n"
                f"Response: {answer_phrase}\n"
                '{"correct": true}'
            ),
        },
    ]


def build_judge_items(
    rows: list[dict[str, Any]], planned: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    output = []
    for row in rows:
        item = planned[row["item_id"]]
        if row["variant"] == "choices_to_question":
            task = "question_reconstruction_fidelity"
            messages = reconstruction_judge_messages(item, row)
        elif row["variant"] == "generative_no_choices":
            task = "free_answer_semantic_correctness"
            messages = answer_semantic_judge_messages(item, row)
        else:
            continue
        judge_item_id = stable_hash({
            "source_item_id": row["item_id"],
            "source_model": row["model"],
            "task": task,
            "messages": messages,
        })[:24]
        output.append({
            "judge_item_id": judge_item_id,
            "task": task,
            "dataset": row["dataset"],
            "example_id": row["example_id"],
            "source_generation_item_id": row["item_id"],
            "source_model": row["model"],
            "source_response_id": row.get("response_id"),
            "messages": messages,
            "messages_sha256": stable_hash(messages),
            # Gold is retained outside the prompt for leakage-free local scoring.
            "gold_option": chr(65 + int(item["answer_index"])),
        })
    return output


def write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "dataset",
        "variant",
        "n",
        "cost_usd",
        "input_tokens",
        "output_tokens",
        "reasoning_tokens",
        "mean_question_token_f1",
        "question_mark_compliance",
        "local_strict_accuracy",
        "local_option_recovery_accuracy",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in fields} for row in rows)


def cumulative_cost_checkpoints(
    rows: list[dict[str, Any]], *, interval: int, model: dict[str, Any]
) -> list[dict[str, Any]]:
    checkpoints = list(range(interval, len(rows) + 1, interval))
    if not checkpoints or checkpoints[-1] != len(rows):
        checkpoints.append(len(rows))
    output = []
    for completed in checkpoints:
        prefix = rows[:completed]
        totals: dict[str, int] = defaultdict(int)
        for row in prefix:
            for key, value in usage_counts(row.get("usage", {}) or {}).items():
                totals[key] += value
        output.append({
            "completed_rows": completed,
            "checkpoint_kind": (
                "interval" if completed % interval == 0 else "final"
            ),
            "cost_usd": sum(row_cost(row, model) for row in prefix),
            **totals,
        })
    return output


def write_cost_checkpoint_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "completed_rows",
        "checkpoint_kind",
        "cost_usd",
        "input_tokens",
        "output_tokens",
        "reasoning_tokens",
        "cached_input_tokens",
        "cache_write_tokens",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in fields} for row in rows)


def finalize(run_dir: Path) -> dict[str, Any]:
    config = read_json(run_dir / "config.snapshot.json")
    model = [model for model in config["models"] if model.get("enabled", True)][0]
    planned_rows = read_jsonl(run_dir / "planned_items.jsonl")
    planned = {row["item_id"]: row for row in planned_rows}
    prediction_paths = sorted(run_dir.rglob("predictions.jsonl"))
    raw_rows = [row for path in prediction_paths for row in read_jsonl(path)]
    canonical = latest_rows(raw_rows)
    usable = [row for row in canonical if is_usable(row)]
    expected = {(model["model"], item_id) for item_id in planned}
    observed = {(row["model"], row["item_id"]) for row in usable}
    missing = sorted(expected - observed)
    unexpected = sorted(observed - expected)
    if missing or unexpected:
        raise RuntimeError(
            f"Extension run is incomplete or contaminated: missing={len(missing)}, "
            f"unexpected={len(unexpected)}, first_missing={missing[:3]}"
        )

    plan_order = {item["item_id"]: index for index, item in enumerate(planned_rows)}
    usable.sort(key=lambda row: plan_order[row["item_id"]])
    write_jsonl(run_dir / "canonical_predictions.jsonl", usable)
    cost_checkpoints = cumulative_cost_checkpoints(
        usable, interval=int(config.get("progress_every", 200)), model=model
    )
    write_jsonl(run_dir / "combined_cost_checkpoints.jsonl", cost_checkpoints)
    write_cost_checkpoint_csv(
        run_dir / "combined_cost_checkpoints.csv", cost_checkpoints
    )

    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in usable:
        groups[(row["dataset"], row["variant"])].append(row)
    summary_rows = []
    for (dataset, variant), rows in sorted(groups.items()):
        totals: dict[str, int] = defaultdict(int)
        for row in rows:
            for key, value in usage_counts(row.get("usage", {}) or {}).items():
                totals[key] += value
        summary = {
            "dataset": dataset,
            "variant": variant,
            "n": len(rows),
            "cost_usd": sum(row_cost(row, model) for row in rows),
            **totals,
            "mean_question_token_f1": None,
            "question_mark_compliance": None,
            "local_strict_accuracy": None,
            "local_option_recovery_accuracy": None,
        }
        if variant == "choices_to_question":
            summary["mean_question_token_f1"] = mean([
                float(row["question_token_f1"]) for row in rows
            ])
            summary["question_mark_compliance"] = mean([
                float(str(row.get("output", "")).strip().endswith("?")) for row in rows
            ])
        else:
            summary["local_strict_accuracy"] = mean([
                float(bool(row.get("strict_correct"))) for row in rows
            ])
            summary["local_option_recovery_accuracy"] = mean([
                float(bool(row.get("option_recovery_correct"))) for row in rows
            ])
        summary_rows.append(summary)
    write_summary_csv(run_dir / "paper_analysis_summary.csv", summary_rows)

    judge_items = build_judge_items(usable, planned)
    write_jsonl(run_dir / "judge_items.jsonl", judge_items)
    judge_counts: dict[str, int] = defaultdict(int)
    for item in judge_items:
        judge_counts[item["task"]] += 1
    judge_manifest = {
        "schema_version": "cellpress.judge_plan.v1",
        "generated_at": now_iso(),
        "source_model": model["model"],
        "planned_judge_calls": len(judge_items),
        "task_counts": dict(sorted(judge_counts.items())),
        "reference_choices_excluded_from_judge_prompts": True,
        "reference_correct_answer_included_only_for_free_answer_judgment": True,
        "judge_items_sha256": file_sha256(run_dir / "judge_items.jsonl"),
        "judge_items": str(run_dir / "judge_items.jsonl"),
    }
    write_json(run_dir / "judge_plan_manifest.json", judge_manifest)

    baseline = read_jsonl(run_dir / "baseline_predictions.jsonl")
    baseline_by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in baseline:
        baseline_by_dataset[row["dataset"]].append(row)
    analysis = {
        "schema_version": "cellpress.extension_analysis.v1",
        "generated_at": now_iso(),
        "model": model,
        "planned_calls": len(planned),
        "completed_calls": len(usable),
        "raw_rows": len(raw_rows),
        "duplicate_or_retry_rows": len(raw_rows) - len(canonical),
        "api_error_rows": sum(bool(row.get("api_error")) for row in raw_rows),
        "truncated_rows": sum(bool(row.get("truncated")) for row in raw_rows),
        "finish_reasons": dict(sorted(
            (reason, sum(row.get("finish_reason") == reason for row in usable))
            for reason in {row.get("finish_reason") for row in usable}
        )),
        "total_cost_usd": sum(
            row_cost(row, model) for row in raw_rows if not row.get("api_error")
        ),
        "source_prediction_files": [
            {"path": str(path), "sha256": file_sha256(path)}
            for path in prediction_paths
        ],
        "baseline_accuracy": {
            dataset: sum(bool(row.get("correct")) for row in rows) / len(rows)
            for dataset, rows in sorted(baseline_by_dataset.items())
        },
        "condition_results": summary_rows,
        "judge_plan": judge_manifest,
        "artifacts": {
            "canonical_predictions": str(run_dir / "canonical_predictions.jsonl"),
            "paper_analysis_summary_csv": str(run_dir / "paper_analysis_summary.csv"),
            "combined_cost_checkpoints_jsonl": str(
                run_dir / "combined_cost_checkpoints.jsonl"
            ),
            "combined_cost_checkpoints_csv": str(
                run_dir / "combined_cost_checkpoints.csv"
            ),
            "judge_items": str(run_dir / "judge_items.jsonl"),
            "judge_plan_manifest": str(run_dir / "judge_plan_manifest.json"),
        },
    }
    write_json(run_dir / "paper_analysis.json", analysis)
    return analysis


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    print(json.dumps(finalize(args.run_dir), indent=2))


if __name__ == "__main__":
    main()
