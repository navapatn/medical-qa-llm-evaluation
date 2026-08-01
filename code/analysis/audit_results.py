#!/usr/bin/env python3
"""Summarize reproduction result quality for one paper."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from memory_client import record_audit_summary
from run_codex_condition import validate_result


CODEX_ROOT = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper", required=True)
    parser.add_argument("--result-name", default="result.json")
    parser.add_argument("--ground-truth-name", default="ground_truth.json")
    args = parser.parse_args()

    paper_root = CODEX_ROOT / "papers" / args.paper
    rows = []
    for label in ["A", "B", "C"]:
        result_path = paper_root / "conditions" / f"condition_{label}" / args.result_name
        ground_truth_path = paper_root / "conditions" / f"condition_{label}" / args.ground_truth_name
        summary = validate_result(result_path, ground_truth_path)
        rows.append((label, result_path, summary))

    print(f"paper={args.paper} result_name={args.result_name}")
    for label, result_path, summary in rows:
        status = "usable" if summary["usable"] else "not_usable"
        print(
            f"condition_{label}: {status} "
            f"failure_class={summary['failure_class']} "
            f"result_status={summary['status']} "
            f"metrics={summary['n_metrics']} "
            f"non_null={summary['n_non_null_metrics']} "
            f"null={summary['n_null_metrics']} "
            f"retry={summary['retry_scope']} "
            f"path={result_path}"
        )
        if summary["errors"]:
            print(f"  errors: {'; '.join(summary['errors'])}")
        if summary["failed_metric_cells"]:
            print(f"  failed_cells: {len(summary['failed_metric_cells'])}")

    failure_counts = {}
    retry_items = []
    for label, result_path, summary in rows:
        failure_counts[summary["failure_class"]] = failure_counts.get(summary["failure_class"], 0) + 1
        if summary["retry_recommended"]:
            retry_items.append({
                "condition": label,
                "result_path": str(result_path),
                "failure_class": summary["failure_class"],
                "failure_classes": summary["failure_classes"],
                "retry_scope": summary["retry_scope"],
                "retry_hint": summary["retry_hint"],
                "failed_metric_cells": summary["failed_metric_cells"],
            })
    audit = {
        "paper_id": args.paper,
        "result_name": args.result_name,
        "failure_class_counts": failure_counts,
        "retry_plan_inputs": retry_items,
        "conditions": [
            {
                "condition": label,
                "result_path": str(result_path),
                **summary,
            }
            for label, result_path, summary in rows
        ],
    }
    audit_path = paper_root / f"result_audit_{args.result_name.replace('/', '_')}.json"
    audit_path.write_text(json.dumps(audit, indent=2))
    print(f"audit={audit_path}")
    record_audit_summary(paper_id=args.paper, result_name=args.result_name, audit=audit)


if __name__ == "__main__":
    main()
