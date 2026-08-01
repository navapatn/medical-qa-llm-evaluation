#!/usr/bin/env python3
"""Build final paper-style reproduction summaries from condition results."""

from __future__ import annotations

import argparse
import csv
import json
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from audit_contracts import classify_error_text, sanitize_error_text


CODEX_ROOT = Path(__file__).resolve().parent
DEFAULT_RESULT_NAMES = [
    "result_secondpass_merged.json",
    "result_repair_merged.json",
    "result_merged.json",
    "result_retry_merged.json",
    "result.json",
]


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def normalize_text(value: Any) -> str:
    return "" if value is None else str(value)


def metric_key(metric: dict[str, Any]) -> tuple[str, str, str]:
    return (
        normalize_text(metric.get("table_id")),
        normalize_text(metric.get("row_id")),
        normalize_text(metric.get("col_id")),
    )


def cell_key(row_id: Any, col_id: Any) -> str:
    return f"{normalize_text(row_id)} | {normalize_text(col_id)}"


def fallback_metric_key(key: tuple[str, str, str]) -> tuple[str, str, str]:
    return "", key[1], key[2]


def format_value(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, (int, float)):
        return f"{float(value):.4g}"
    return str(value)


def load_ground_truth(paper_root: Path, conditions: list[str]) -> dict[str, Any]:
    benchmark_path = paper_root / "ground_truth.benchmark.json"
    if benchmark_path.exists():
        try:
            return read_json(benchmark_path)
        except Exception:
            pass

    candidates = [paper_root / "ground_truth.unfiltered.json", paper_root / "ground_truth.full_columns.json", paper_root / "ground_truth.json"]
    for condition in conditions:
        cond = paper_root / "conditions" / f"condition_{condition}"
        candidates.extend([cond / "ground_truth_merged.json", cond / "ground_truth.json"])

    first: dict[str, Any] | None = None
    tables: OrderedDict[str, dict[str, Any]] = OrderedDict()
    metrics: OrderedDict[tuple[str, str, str], dict[str, Any]] = OrderedDict()
    for path in candidates:
        if not path.exists():
            continue
        try:
            gt = read_json(path)
        except Exception:
            continue
        if first is None:
            first = dict(gt)
        for table in gt.get("tables", []):
            table_id = normalize_text(table.get("table_id") or f"table_{len(tables) + 1}")
            tables.setdefault(table_id, table)
        for metric in gt.get("expected_metrics", []):
            if not isinstance(metric, dict):
                continue
            metrics.setdefault(metric_key(metric), metric)

    base = first or {"paper_id": paper_root.name, "tables": [], "expected_metrics": []}
    base["tables"] = list(tables.values())
    base["expected_metrics"] = list(metrics.values())
    return base


def load_blocked_cells(paper_root: Path) -> dict[tuple[str, str, str], dict[str, Any]]:
    out: dict[tuple[str, str, str], dict[str, Any]] = {}
    for path in [paper_root / "blocked_cells.json", paper_root / "data_feasibility_plan.json"]:
        if not path.exists():
            continue
        try:
            payload = read_json(path)
        except Exception:
            continue
        cells = payload.get("cells") if isinstance(payload.get("cells"), list) else payload.get("blocked_cells", [])
        for cell in cells or []:
            if not isinstance(cell, dict):
                continue
            key = (
                normalize_text(cell.get("table_id")),
                normalize_text(cell.get("row_id")),
                normalize_text(cell.get("col_id")),
            )
            out[key] = cell
            if key[0]:
                out.setdefault(fallback_metric_key(key), cell)
    return out


def source_rank(path: Path, preferred_names: list[str]) -> int:
    try:
        name_rank = preferred_names.index(path.name)
    except ValueError:
        name_rank = len(preferred_names)
    secondpass_bonus = -2 if "secondpass" in str(path) else 0
    repair_bonus = -1 if "repair" in str(path) or "retry" in str(path) else 0
    return name_rank + secondpass_bonus + repair_bonus


def iter_condition_result_paths(paper_root: Path, condition: str, preferred_names: list[str]) -> list[Path]:
    paths: list[Path] = []
    cond_root = paper_root / "conditions" / f"condition_{condition}"
    for name in preferred_names:
        path = cond_root / name
        if path.exists():
            paths.append(path)
    for path in sorted(cond_root.glob("result*.json")) if cond_root.exists() else []:
        if path.name.startswith("result_audit_") or path in paths:
            continue
        paths.append(path)
    shard_root = paper_root / "sharded_runs"
    if shard_root.exists():
        paths.extend(sorted(shard_root.glob(f"*/**/condition_{condition}/result.json")))
    return sorted(set(paths), key=lambda p: (source_rank(p, preferred_names), -p.stat().st_mtime))


def result_status_score(status: str | None) -> int:
    if status == "success":
        return 3
    if status == "partial":
        return 2
    if status in {"execution_failed", "could_not_implement"}:
        return 1
    return 0


def collect_candidates(paper_root: Path, condition: str, preferred_names: list[str]) -> dict[tuple[str, str, str], list[dict[str, Any]]]:
    candidates: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for path in iter_condition_result_paths(paper_root, condition, preferred_names):
        try:
            result = read_json(path)
        except Exception as exc:
            candidates.setdefault(("", ""), []).append({
                "value": None,
                "source_path": str(path),
                "status": "invalid_json",
                "notes": f"Invalid JSON: {exc}",
                "failure_class": "invalid_json",
            })
            continue
        status = result.get("status")
        diagnosis = sanitize_error_text(normalize_text(result.get("failure_diagnosis")))
        for metric in result.get("metrics", []):
            if not isinstance(metric, dict):
                continue
            key = metric_key(metric)
            notes = sanitize_error_text(normalize_text(metric.get("notes")))
            failure_class = metric.get("failure_class")
            if metric.get("value") is None:
                failure_class = failure_class or classify_error_text(notes or diagnosis, default="null_metric")
            candidates.setdefault(key, []).append({
                "table_id": key[0],
                "row_id": key[1],
                "col_id": key[2],
                "value": metric.get("value"),
                "notes": notes,
                "status": status,
                "source_path": str(path),
                "source_name": path.name,
                "source_rank": source_rank(path, preferred_names),
                "source_mtime": path.stat().st_mtime,
                "failure_class": failure_class,
                "failure_diagnosis": diagnosis,
            })
    return candidates


def choose_candidate(items: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not items:
        return None
    return sorted(
        items,
        key=lambda item: (
            0 if item.get("value") is not None else 1,
            -result_status_score(item.get("status")),
            item.get("source_rank", 999),
            -float(item.get("source_mtime", 0)),
        ),
    )[0]


def choose_candidate_for_key(
    candidates: dict[tuple[str, str, str], list[dict[str, Any]]],
    key: tuple[str, str, str],
) -> tuple[dict[str, Any] | None, str]:
    exact = choose_candidate(candidates.get(key, []))
    if exact is not None:
        return exact, "exact"
    fallback = choose_candidate(candidates.get(fallback_metric_key(key), []))
    if fallback is not None:
        return fallback, "row_col_fallback"
    return None, "missing"


def table_dimensions(metrics: list[dict[str, Any]]) -> tuple[list[str], list[str]]:
    rows = list(OrderedDict((normalize_text(m.get("row_id")), None) for m in metrics).keys())
    cols = list(OrderedDict((normalize_text(m.get("col_id")), None) for m in metrics).keys())
    return rows, cols


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    widths = [len(h) for h in headers]
    for row in rows:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], len(cell))
    def line(parts: list[str]) -> str:
        return "| " + " | ".join(part.ljust(widths[idx]) for idx, part in enumerate(parts)) + " |"
    sep = "| " + " | ".join("-" * widths[idx] for idx in range(len(headers))) + " |"
    return "\n".join([line(headers), sep, *[line(row) for row in rows]])


def build_summary(paper_id: str, conditions: list[str], preferred_names: list[str]) -> dict[str, Any]:
    paper_root = CODEX_ROOT / "papers" / paper_id
    gt = load_ground_truth(paper_root, conditions)
    blocked = load_blocked_cells(paper_root)
    expected = [m for m in gt.get("expected_metrics", []) if isinstance(m, dict)]
    by_condition = {condition: collect_candidates(paper_root, condition, preferred_names) for condition in conditions}

    final_cells = []
    counts = {condition: {"expected": 0, "non_null": 0, "blocked": 0, "missing": 0} for condition in conditions}
    for metric in expected:
        key = metric_key(metric)
        blocked_info = blocked.get(key) or blocked.get(fallback_metric_key(key))
        for condition in conditions:
            counts[condition]["expected"] += 1
            selected, match_scope = choose_candidate_for_key(by_condition[condition], key)
            value = selected.get("value") if selected else None
            status = "success" if value is not None else "missing"
            reason = None
            if blocked_info and value is None:
                status = "blocked"
                reason = (
                    blocked_info.get("blocked_reason")
                    or blocked_info.get("block_reason")
                    or blocked_info.get("suggested_next_action")
                    or blocked_info.get("availability_status")
                    or "dataset is not runnable"
                )
                counts[condition]["blocked"] += 1
            elif value is None:
                reason = selected.get("notes") or selected.get("failure_diagnosis") if selected else "no usable result was produced"
                counts[condition]["missing"] += 1
            else:
                counts[condition]["non_null"] += 1
            final_cells.append({
                "table_id": key[0],
                "row_id": key[1],
                "col_id": key[2],
                "condition": condition,
                "value": value,
                "status": status,
                "reason": reason,
                "source_path": selected.get("source_path") if selected else None,
                "source_status": selected.get("status") if selected else None,
                "source_notes": selected.get("notes") if selected else None,
                "candidate_match_scope": match_scope,
                "paper_reported_value": metric.get("value"),
                "dataset_availability": blocked_info.get("availability_status") if blocked_info else "runnable_or_unknown",
            })

    overall_status = "success"
    if any(c["blocked"] for c in counts.values()):
        overall_status = "partial_blocked"
    if any(c["missing"] for c in counts.values()):
        overall_status = "partial"
    if all(c["non_null"] == 0 for c in counts.values()):
        overall_status = "no_results"

    return {
        "paper_id": paper_id,
        "generated_at": now_iso(),
        "conditions": conditions,
        "result_preference_order": preferred_names,
        "overall_status": overall_status,
        "condition_counts": counts,
        "tables": gt.get("tables", []),
        "expected_metrics": expected,
        "final_cells": final_cells,
        "blocked_cells": list(blocked.values()),
    }


def write_csv(summary: dict[str, Any], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "table_id",
        "row_id",
        "col_id",
        "condition",
        "value",
        "status",
        "reason",
        "source_path",
        "source_status",
        "candidate_match_scope",
        "paper_reported_value",
        "dataset_availability",
    ]
    with out_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for cell in summary["final_cells"]:
            writer.writerow({field: cell.get(field) for field in fields})


def write_markdown(summary: dict[str, Any], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    conditions = summary["conditions"]
    lines = [
        f"# Reproduction Summary: {summary['paper_id']}",
        "",
        f"Generated: `{summary['generated_at']}`",
        f"Overall status: `{summary['overall_status']}`",
        "",
        "## Condition Coverage",
        "",
        markdown_table(
            ["Condition", "Expected", "Non-null", "Blocked", "Missing"],
            [
                [
                    condition,
                    str(counts["expected"]),
                    str(counts["non_null"]),
                    str(counts["blocked"]),
                    str(counts["missing"]),
                ]
                for condition, counts in summary["condition_counts"].items()
            ],
        ),
    ]

    if summary["blocked_cells"]:
        lines.extend(["", "## Unrunnable Dataset Cells", ""])
        blocked_rows = []
        for cell in summary["blocked_cells"]:
            blocked_rows.append([
                normalize_text(cell.get("table_id")),
                normalize_text(cell.get("row_id")),
                normalize_text(cell.get("col_id")),
                normalize_text(cell.get("availability_status")),
                normalize_text(cell.get("blocked_reason") or cell.get("block_reason") or cell.get("suggested_next_action")),
            ])
        lines.append(markdown_table(["Table", "Row", "Column", "Availability", "Reason"], blocked_rows))

    cells_by_table: dict[str, list[dict[str, Any]]] = {}
    for metric in summary["expected_metrics"]:
        cells_by_table.setdefault(normalize_text(metric.get("table_id") or "Results"), []).append(metric)
    final_lookup = {
        (cell["condition"], cell["table_id"], cell["row_id"], cell["col_id"]): cell
        for cell in summary["final_cells"]
    }

    for table in summary.get("tables", []) or [{"table_id": key} for key in cells_by_table]:
        table_id = normalize_text(table.get("table_id") or "Results")
        metrics = cells_by_table.get(table_id, [])
        if not metrics:
            continue
        rows, cols = table_dimensions(metrics)
        caption = normalize_text(table.get("table_caption"))
        lines.extend(["", f"## {table_id}", ""])
        if caption:
            lines.extend([caption, ""])
        for condition in conditions:
            body = []
            for row_id in rows:
                row = [row_id]
                for col_id in cols:
                    cell = final_lookup.get((condition, table_id, row_id, col_id))
                    if cell is None:
                        row.append("")
                    elif cell["status"] == "blocked":
                        row.append("BLOCKED")
                    else:
                        row.append(format_value(cell.get("value")))
                body.append(row)
            lines.extend([f"### Condition {condition}", "", markdown_table(["Row", *cols], body), ""])

    lines.extend(["", "## Notes", ""])
    missing_or_blocked = [c for c in summary["final_cells"] if c["status"] != "success"]
    if not missing_or_blocked:
        lines.append("All expected cells have non-null reproduced values.")
    else:
        for cell in missing_or_blocked:
            lines.append(
                f"- condition {cell['condition']} `{cell['table_id']}` / `{cell['row_id']}` / `{cell['col_id']}`: "
                f"{cell['status']} - {cell.get('reason') or 'no reason recorded'}"
            )
    out_path.write_text("\n".join(lines).rstrip() + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper-id", required=True)
    parser.add_argument("--conditions", default="A,B,C")
    parser.add_argument("--result-names", default=",".join(DEFAULT_RESULT_NAMES))
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    conditions = [c.strip().upper() for c in args.conditions.split(",") if c.strip()]
    result_names = [name.strip() for name in args.result_names.split(",") if name.strip()]
    summary = build_summary(args.paper_id, conditions, result_names)

    paper_root = CODEX_ROOT / "papers" / args.paper_id
    out_dir = Path(args.out_dir) if args.out_dir else paper_root / "final_summary"
    write_json(out_dir / "summary.json", summary)
    write_csv(summary, out_dir / "summary.csv")
    write_markdown(summary, out_dir / "summary.md")

    print(f"[summarize-results] paper={args.paper_id} status={summary['overall_status']}")
    for condition, counts in summary["condition_counts"].items():
        print(
            f"[summarize-results] condition_{condition}: "
            f"{counts['non_null']}/{counts['expected']} non-null, "
            f"blocked={counts['blocked']}, missing={counts['missing']}"
        )
    print(f"[summarize-results] json={out_dir / 'summary.json'}")
    print(f"[summarize-results] csv={out_dir / 'summary.csv'}")
    print(f"[summarize-results] markdown={out_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
