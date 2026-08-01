#!/usr/bin/env python3
"""Validate and consolidate sharded medical-QA paper-production outputs."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from paper_run_tracking import file_sha256, usage_counts
from replicate_medical_reasoning import (
    UNUSABLE_FINISH_REASONS,
    latest_prediction_rows,
    row_cost,
    summarize,
    write_metrics_csv,
)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    # Iterate by physical newline. ``str.splitlines`` also splits Unicode line
    # separators that can legitimately occur inside JSON string values.
    with path.open(errors="replace") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    )


def is_usable(row: dict[str, Any]) -> bool:
    return (
        not row.get("api_error")
        and not row.get("truncated", False)
        and row.get("finish_reason") not in UNUSABLE_FINISH_REASONS
    )


def cumulative_checkpoints(
    rows: list[dict[str, Any]], *, interval: int, cost_for
) -> list[dict[str, Any]]:
    chronological = sorted(rows, key=lambda row: str(row.get("created_at", "")))
    checkpoints = list(range(interval, len(chronological) + 1, interval))
    if not checkpoints or checkpoints[-1] != len(chronological):
        checkpoints.append(len(chronological))
    output = []
    for completed in checkpoints:
        prefix = chronological[:completed]
        token_totals: Counter[str] = Counter()
        for row in prefix:
            token_totals.update(usage_counts(row.get("usage", {}) or {}))
        output.append({
            "completed_rows": completed,
            "checkpoint_kind": (
                "interval" if completed % interval == 0 else "final"
            ),
            "cost_usd": sum(cost_for(row) for row in prefix),
            **dict(token_totals),
            "accuracy": sum(bool(row.get("correct")) for row in prefix) / completed,
        })
    return output


def write_checkpoint_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "completed_rows",
        "checkpoint_kind",
        "cost_usd",
        "input_tokens",
        "output_tokens",
        "reasoning_tokens",
        "cached_input_tokens",
        "cache_write_tokens",
        "accuracy",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in fields} for row in rows)


def finalize(run_dir: Path, interval: int) -> dict[str, Any]:
    config = read_json(run_dir / "config.snapshot.json")
    planned_items = read_jsonl(run_dir / "planned_items.jsonl")
    models = [model for model in config["models"] if model.get("enabled", True)]
    if len(models) != 1:
        raise RuntimeError("Paper-run finalization currently expects one model")
    model = models[0]
    prediction_paths = sorted(run_dir.rglob("predictions.jsonl"))
    raw_rows = [row for path in prediction_paths for row in read_jsonl(path)]
    canonical = latest_prediction_rows(raw_rows)
    usable = [row for row in canonical if is_usable(row)]

    expected = {
        (model["provider"], model["model"], item["item_id"], 0)
        for item in planned_items
    }
    observed = {
        (row["provider"], row["model"], row["item_id"], int(row["sample_index"]))
        for row in usable
    }
    missing = sorted(expected - observed)
    unexpected = sorted(observed - expected)
    if missing or unexpected:
        raise RuntimeError(
            f"Run is incomplete or contaminated: missing={len(missing)}, "
            f"unexpected={len(unexpected)}, first_missing={missing[:3]}, "
            f"first_unexpected={unexpected[:3]}"
        )

    # Recovery-only attempts may use a different prompt policy while retaining
    # the same provider/model logical prediction.  Preserve that provenance on
    # every row, but normalize the analysis label to the registered primary
    # model so the combined k=1 metrics include all canonical predictions.
    for row in usable:
        actual_label = row.get("model_label")
        if actual_label != model["label"]:
            row["generation_model_label"] = actual_label
            row["model_label"] = model["label"]

    plan_order = {item["item_id"]: index for index, item in enumerate(planned_items)}
    usable.sort(key=lambda row: (plan_order[row["item_id"]], row["sample_index"]))
    write_jsonl(run_dir / "combined_predictions.jsonl", usable)
    metrics = summarize(usable, planned_items, models, [1])
    write_json(run_dir / "combined_metrics.json", metrics)
    write_metrics_csv(run_dir / "combined_metrics.csv", metrics)

    model_lookup = {(model["provider"], model["model"]): model}
    cost_for = lambda row: row_cost(
        row, model_lookup[(row["provider"], row["model"])]
    )
    checkpoints = cumulative_checkpoints(
        usable, interval=interval, cost_for=cost_for
    )
    write_jsonl(run_dir / "combined_cost_checkpoints.jsonl", checkpoints)
    write_checkpoint_csv(run_dir / "combined_cost_checkpoints.csv", checkpoints)

    by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in usable:
        by_dataset[row["dataset"]].append(row)
    dataset_summary = []
    for dataset, rows in sorted(by_dataset.items()):
        dataset_summary.append({
            "dataset": dataset,
            "n": len(rows),
            "accuracy": sum(bool(row.get("correct")) for row in rows) / len(rows),
            "parse_coverage": sum(row.get("parsed_answer") is not None for row in rows)
            / len(rows),
            "cost_usd": sum(cost_for(row) for row in rows),
        })

    recovery_exceptions_path = run_dir / "recovery_exceptions.json"
    recovery_exceptions = (
        read_json(recovery_exceptions_path)
        if recovery_exceptions_path.exists()
        else []
    )

    manifest = {
        "schema_version": "cellpress.combined_paper_run.v1",
        "generated_at": now_iso(),
        "run_dir": str(run_dir),
        "model": model,
        "planned_rows": len(expected),
        "completed_rows": len(usable),
        "raw_prediction_rows": len(raw_rows),
        "duplicate_or_retry_rows": len(raw_rows) - len(canonical),
        "failed_canonical_rows": sum(not is_usable(row) for row in canonical),
        "truncated_raw_rows": sum(bool(row.get("truncated")) for row in raw_rows),
        "api_error_raw_rows": sum(bool(row.get("api_error")) for row in raw_rows),
        "response_models": sorted({str(row.get("response_model")) for row in usable}),
        "total_cost_usd": sum(cost_for(row) for row in raw_rows if not row.get("api_error")),
        "canonical_cost_usd": sum(cost_for(row) for row in usable),
        "dataset_results": dataset_summary,
        "recovery_exceptions": recovery_exceptions,
        "source_prediction_files": [
            {"path": str(path), "sha256": file_sha256(path)}
            for path in prediction_paths
        ],
        "artifacts": {
            "combined_predictions": str(run_dir / "combined_predictions.jsonl"),
            "combined_metrics_json": str(run_dir / "combined_metrics.json"),
            "combined_metrics_csv": str(run_dir / "combined_metrics.csv"),
            "combined_cost_checkpoints_jsonl": str(
                run_dir / "combined_cost_checkpoints.jsonl"
            ),
            "combined_cost_checkpoints_csv": str(
                run_dir / "combined_cost_checkpoints.csv"
            ),
            "recovery_exceptions": str(recovery_exceptions_path),
        },
    }
    write_json(run_dir / "combined_run_manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--interval", type=int, default=200)
    args = parser.parse_args()
    manifest = finalize(args.run_dir, args.interval)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
