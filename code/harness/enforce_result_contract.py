#!/usr/bin/env python3
"""Enforce the final result contract for a paper run.

The contract is intentionally deterministic:
- non-null metric cells are complete;
- null metric cells are acceptable only when classified as terminal blockers;
- retryable null cells must be repaired by the orchestrator before finalization.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from audit_contracts import classify_error_text


CODEX_ROOT = Path(__file__).resolve().parent
TERMINAL_NULL_CLASSES = {
    "dataset_blocked",
    "api_model_unavailable",
    "api_budget_exceeded",
}
RETRYABLE_NULL_CLASSES = {
    "missing_result_file",
    "invalid_json",
    "null_metric",
    "all_null_metrics",
    "implementation_error",
    "smoke_only",
    "timeout",
    "setup_failure",
}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")


def inspect_condition(paper_root: Path, condition: str, result_name: str) -> dict[str, Any]:
    path = paper_root / "conditions" / f"condition_{condition}" / result_name
    if not path.exists():
        return {
            "condition": condition,
            "result_path": str(path),
            "status": "missing_result",
            "metrics": 0,
            "non_null": 0,
            "terminal_nulls": [],
            "retryable_nulls": [
                {
                    "row_id": None,
                    "col_id": None,
                    "failure_class": "missing_result_file",
                    "reason": f"{path} is missing",
                }
            ],
        }
    result = read_json(path)
    terminal_nulls: list[dict[str, Any]] = []
    retryable_nulls: list[dict[str, Any]] = []
    metrics = [m for m in result.get("metrics", []) if isinstance(m, dict)]
    for metric in metrics:
        if metric.get("value") is not None:
            continue
        reason = metric.get("notes") or result.get("failure_diagnosis") or "metric value is null"
        failure_class = classify_error_text(reason, default="null_metric")
        cell = {
            "row_id": metric.get("row_id"),
            "col_id": metric.get("col_id"),
            "failure_class": failure_class,
            "reason": reason,
        }
        if failure_class in TERMINAL_NULL_CLASSES:
            terminal_nulls.append(cell)
        else:
            retryable_nulls.append(cell)
    return {
        "condition": condition,
        "result_path": str(path),
        "status": result.get("status"),
        "metrics": len(metrics),
        "non_null": sum(1 for m in metrics if m.get("value") is not None),
        "terminal_nulls": terminal_nulls,
        "retryable_nulls": retryable_nulls,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper-id", required=True)
    parser.add_argument("--conditions", default="A,B,C")
    parser.add_argument("--result-name", default="result_merged.json")
    parser.add_argument("--out", default=None)
    parser.add_argument(
        "--fail-on-retryable",
        action="store_true",
        help="Exit non-zero when retryable null cells remain.",
    )
    args = parser.parse_args()

    conditions = [c.strip().upper() for c in args.conditions.split(",") if c.strip()]
    paper_root = CODEX_ROOT / "papers" / args.paper_id
    condition_reports = [inspect_condition(paper_root, c, args.result_name) for c in conditions]
    retryable = [cell for report in condition_reports for cell in report["retryable_nulls"]]
    terminal = [cell for report in condition_reports for cell in report["terminal_nulls"]]
    non_null = sum(report["non_null"] for report in condition_reports)
    metrics = sum(report["metrics"] for report in condition_reports)
    failure_class_counts = Counter(cell["failure_class"] for cell in retryable + terminal)
    status = "complete" if not retryable and not terminal else ("terminal_partial" if not retryable else "repair_required")
    payload = {
        "paper_id": args.paper_id,
        "result_name": args.result_name,
        "status": status,
        "metrics": metrics,
        "non_null": non_null,
        "null": len(retryable) + len(terminal),
        "retryable_null_count": len(retryable),
        "terminal_null_count": len(terminal),
        "failure_class_counts": dict(failure_class_counts),
        "conditions": condition_reports,
    }
    out = Path(args.out) if args.out else paper_root / f"result_contract_{args.result_name.replace('/', '_')}.json"
    write_json(out, payload)
    print(
        f"[result-contract] paper={args.paper_id} status={status} "
        f"non_null={non_null}/{metrics} retryable_nulls={len(retryable)} terminal_nulls={len(terminal)}"
    )
    print(f"[result-contract] report={out}")
    if args.fail_on_retryable and retryable:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
