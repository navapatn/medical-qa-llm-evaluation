#!/usr/bin/env python3
"""Validate, consolidate, and analyze sharded medical-QA judge outputs."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from paper_run_tracking import file_sha256, usage_counts
from replicate_medical_reasoning import row_cost
from run_medical_qa_judge import is_transport_usable, latest_rows


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


def wilson(successes: int, n: int, z: float = 1.959963984540054) -> list[float] | None:
    if n == 0:
        return None
    p = successes / n
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    margin = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / denominator
    return [center - margin, center + margin]


def group_metric(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    parsed = [row for row in rows if not row.get("parse_error")]
    task = str(rows[0]["task"]) if rows else ""
    result: dict[str, Any] = {
        "task": task,
        "n": n,
        "parsed_n": len(parsed),
        "parse_coverage": len(parsed) / n if n else None,
    }
    if task == "question_reconstruction_fidelity":
        matches = sum(bool(row.get("target_match")) for row in parsed)
        # Boolean-only judge prompts (the datasetaware plan) return no numeric
        # fidelity score, so guard against None instead of crashing on int(None).
        fidelity = [
            int(row["fidelity"]) for row in parsed if row.get("fidelity") is not None
        ]
        result.update({
            "target_match_count": matches,
            "target_match_rate": matches / n if n else None,
            "target_match_ci95": wilson(matches, n),
            "mean_fidelity": sum(fidelity) / len(fidelity) if fidelity else None,
            "fidelity_distribution": dict(sorted(Counter(fidelity).items())),
        })
    elif task == "free_answer_option_mapping":
        correct = sum(bool(row.get("judge_correct")) for row in parsed)
        mapped = Counter(str(row.get("mapped_option")) for row in parsed)
        result.update({
            "correct_count": correct,
            "mapping_accuracy": correct / n if n else None,
            "mapping_accuracy_ci95": wilson(correct, n),
            "substantive_option_rate": sum(mapped[label] for label in "ABCD") / n if n else None,
            "none_rate": mapped["NONE"] / n if n else None,
            "ambiguous_rate": mapped["AMBIGUOUS"] / n if n else None,
            "mapped_option_distribution": dict(sorted(mapped.items())),
        })
    elif task == "free_answer_semantic_correctness":
        correct = sum(bool(row.get("judge_correct")) for row in parsed)
        result.update({
            "correct_count": correct,
            "semantic_accuracy": correct / n if n else None,
            "semantic_accuracy_ci95": wilson(correct, n),
        })
    return result


def cumulative_checkpoints(
    rows: list[dict[str, Any]], *, interval: int, model: dict[str, Any]
) -> list[dict[str, Any]]:
    targets = list(range(interval, len(rows) + 1, interval))
    if not targets or targets[-1] != len(rows):
        targets.append(len(rows))
    output = []
    for completed in targets:
        prefix = rows[:completed]
        totals: Counter[str] = Counter()
        for row in prefix:
            totals.update(usage_counts(row.get("usage", {}) or {}))
        output.append({
            "completed_rows": completed,
            "checkpoint_kind": "interval" if completed % interval == 0 else "final",
            "cost_usd": sum(row_cost(row, model) for row in prefix),
            **dict(totals),
            "parse_coverage": sum(not row.get("parse_error") for row in prefix) / completed,
        })
    return output


def write_checkpoint_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "completed_rows", "checkpoint_kind", "cost_usd", "input_tokens",
        "output_tokens", "reasoning_tokens", "cached_input_tokens",
        "cache_write_tokens", "parse_coverage",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in fields} for row in rows)


def write_metric_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "dataset", "task", "n", "parsed_n", "parse_coverage",
        "target_match_rate", "mean_fidelity", "mapping_accuracy", "semantic_accuracy",
        "substantive_option_rate", "none_rate", "ambiguous_rate", "cost_usd",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in fields} for row in rows)


def finalize(run_dir: Path) -> dict[str, Any]:
    config = read_json(run_dir / "config.snapshot.json")
    model = [model for model in config["models"] if model.get("enabled", True)][0]
    planned_rows = read_jsonl(run_dir / "planned_items.jsonl")
    plan_order = {
        row["judge_item_id"]: index for index, row in enumerate(planned_rows)
    }
    prediction_paths = sorted(run_dir.rglob("predictions.jsonl"))
    raw_rows = [row for path in prediction_paths for row in read_jsonl(path)]
    canonical = latest_rows(raw_rows)
    usable = [row for row in canonical if is_transport_usable(row)]
    expected = {(model["model"], item_id) for item_id in plan_order}
    observed = {(row["model"], row["judge_item_id"]) for row in usable}
    missing = sorted(expected - observed)
    unexpected = sorted(observed - expected)
    if missing or unexpected:
        raise RuntimeError(
            f"Judge run is incomplete or contaminated: missing={len(missing)}, "
            f"unexpected={len(unexpected)}, first_missing={missing[:3]}"
        )
    usable.sort(key=lambda row: plan_order[row["judge_item_id"]])
    write_jsonl(run_dir / "canonical_judgments.jsonl", usable)

    task_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    dataset_task_groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in usable:
        task_groups[row["task"]].append(row)
        dataset_task_groups[(row["dataset"], row["task"])].append(row)

    task_metrics = {
        task: group_metric(rows) for task, rows in sorted(task_groups.items())
    }
    dataset_metrics = []
    for (dataset, task), rows in sorted(dataset_task_groups.items()):
        metric = {"dataset": dataset, **group_metric(rows)}
        metric["cost_usd"] = sum(row_cost(row, model) for row in rows)
        dataset_metrics.append(metric)
    write_metric_csv(run_dir / "judge_metrics.csv", dataset_metrics)

    checkpoints = cumulative_checkpoints(
        usable, interval=int(config.get("progress_every", 200)), model=model
    )
    write_jsonl(run_dir / "combined_cost_checkpoints.jsonl", checkpoints)
    write_checkpoint_csv(run_dir / "combined_cost_checkpoints.csv", checkpoints)

    analysis = {
        "schema_version": "cellpress.judge_analysis.v1",
        "generated_at": now_iso(),
        "source_model": config["source_model"],
        "judge_model": model,
        "planned_calls": len(planned_rows),
        "completed_calls": len(usable),
        "parsed_calls": sum(not row.get("parse_error") for row in usable),
        "parse_error_calls": sum(bool(row.get("parse_error")) for row in usable),
        "raw_rows": len(raw_rows),
        "duplicate_or_retry_rows": len(raw_rows) - len(canonical),
        "api_error_rows": sum(bool(row.get("api_error")) for row in raw_rows),
        "truncated_rows": sum(bool(row.get("truncated")) for row in raw_rows),
        "response_models": sorted({str(row.get("response_model")) for row in usable}),
        "total_cost_usd": sum(
            row_cost(row, model) for row in raw_rows if not row.get("api_error")
        ),
        "canonical_cost_usd": sum(row_cost(row, model) for row in usable),
        "task_metrics": task_metrics,
        "dataset_metrics": dataset_metrics,
        "source_prediction_files": [
            {"path": str(path), "sha256": file_sha256(path)}
            for path in prediction_paths
        ],
        "artifacts": {
            "canonical_judgments": str(run_dir / "canonical_judgments.jsonl"),
            "judge_metrics_csv": str(run_dir / "judge_metrics.csv"),
            "combined_cost_checkpoints_csv": str(
                run_dir / "combined_cost_checkpoints.csv"
            ),
        },
    }
    write_json(run_dir / "judge_analysis.json", analysis)
    return analysis


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    print(json.dumps(finalize(args.run_dir), indent=2))


if __name__ == "__main__":
    main()
