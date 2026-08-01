#!/usr/bin/env python3
"""Merge one or more sharded run archives, preferring later non-null retries."""

from __future__ import annotations

import argparse
import json
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path

from audit_contracts import classify_result, sanitize_error_text

CODEX_ROOT = Path(__file__).resolve().parent


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))


def metric_map(result_path: Path) -> dict[tuple[str, str], dict]:
    if not result_path.exists():
        return {}
    try:
        result = read_json(result_path)
    except Exception:
        return {}
    out = {}
    for metric in result.get("metrics", []):
        if isinstance(metric, dict):
            out[(metric.get("row_id"), metric.get("col_id"))] = metric
    return out


def archive_shards(archive: Path) -> list[dict]:
    manifest_path = archive / "sharded_run_manifest.json"
    if manifest_path.exists():
        return read_json(manifest_path).get("shards", [])
    return [{"name": p.name, "archive": str(p)} for p in sorted(archive.iterdir()) if p.is_dir() and (p / "ground_truth.json").exists()]


def collect_ground_truth(archives: list[Path]) -> dict:
    first = None
    metrics = OrderedDict()
    tables = OrderedDict()
    for archive in archives:
        for shard in archive_shards(archive):
            shard_dir = Path(shard.get("archive", archive / shard["name"]))
            gt_path = shard_dir / "ground_truth.json"
            if not gt_path.exists():
                continue
            gt = read_json(gt_path)
            if first is None:
                first = dict(gt)
            for table in gt.get("tables", []):
                table_id = table.get("table_id") or f"table_{len(tables)}"
                tables.setdefault(table_id, table)
            for metric in gt.get("expected_metrics", []):
                key = (metric.get("row_id"), metric.get("col_id"))
                metrics.setdefault(key, metric)
    base = first or {}
    base["tables"] = list(tables.values())
    base["expected_metrics"] = list(metrics.values())
    base["archive_merge"] = {
        "merged_at": now_iso(),
        "archives": [str(a) for a in archives],
        "n_expected_metrics": len(metrics),
    }
    return base


def merge_condition(paper_id: str, condition: str, archives: list[Path], result_name: str, model: str | None) -> dict:
    expected_gt = collect_ground_truth(archives)
    expected_order = [(m.get("row_id"), m.get("col_id")) for m in expected_gt.get("expected_metrics", [])]
    merged_metrics = OrderedDict()
    sources = {}
    assumptions = []
    summaries = []
    failures = []

    for archive in archives:
        for shard in archive_shards(archive):
            shard_name = shard["name"]
            shard_dir = Path(shard.get("archive", archive / shard_name))
            result_path = shard_dir / f"condition_{condition}" / result_name
            validation = classify_result(result_path, shard_dir / "ground_truth.json")
            if not result_path.exists():
                failures.append(f"[{archive.name}/{shard_name}] {validation.get('failure_class')}: result file missing")
                continue
            try:
                result = read_json(result_path)
            except Exception as exc:
                failures.append(f"[{archive.name}/{shard_name}] invalid_json: {exc}")
                continue
            for item in result.get("assumptions", []):
                text = f"[{archive.name}/{shard_name}] {item}"
                if text not in assumptions:
                    assumptions.append(text)
            if result.get("execution_summary"):
                summaries.append(f"[{archive.name}/{shard_name}] {sanitize_error_text(result.get('execution_summary'))}")
            if result.get("failure_diagnosis"):
                failures.append(f"[{archive.name}/{shard_name}] {sanitize_error_text(result.get('failure_diagnosis'))}")

            for key, metric in metric_map(result_path).items():
                metric = dict(metric)
                current = merged_metrics.get(key)
                current_value = current.get("value") if current else None
                new_value = metric.get("value")
                if current is None or (current_value is None and new_value is not None) or new_value is not None:
                    note = sanitize_error_text(metric.get("notes", ""))
                    metric["notes"] = f"[{archive.name}/{shard_name}] {note}".strip()
                    if new_value is None:
                        metric.setdefault("failure_class", validation.get("failure_class", "null_metric"))
                    merged_metrics[key] = metric
                    sources[f"{key[0]} | {key[1]}"] = {
                        "archive": str(archive),
                        "shard": shard_name,
                        "result_path": str(result_path),
                        "source_failure_class": validation.get("failure_class"),
                        "source_failure_classes": validation.get("failure_classes", []),
                    }

    output_metrics = []
    for row_id, col_id in expected_order:
        metric = merged_metrics.get((row_id, col_id))
        if metric is None:
            metric = {
                "row_id": row_id,
                "col_id": col_id,
                "value": None,
                "notes": "No shard archive produced this cell.",
                "failure_class": "missing_result_file",
            }
        output_metrics.append(metric)

    non_null = sum(m.get("value") is not None for m in output_metrics)
    status = "success" if output_metrics and non_null == len(output_metrics) else ("partial" if non_null else "execution_failed")
    failure_classes = []
    for metric in output_metrics:
        if metric.get("value") is None:
            failure_classes.append(metric.get("failure_class") or "null_metric")
    if status == "success":
        failure_classes = ["success"]
    elif status == "partial":
        failure_classes = ["partial"] + [c for c in failure_classes if c != "partial"]
    elif output_metrics and len(failure_classes) == len(output_metrics):
        failure_classes = ["all_null_metrics"] + [c for c in failure_classes if c != "all_null_metrics"]
    failure_classes = list(dict.fromkeys(failure_classes or ["implementation_error"]))
    return {
        "paper_id": paper_id,
        "status": status,
        "failure_class": failure_classes[0],
        "failure_classes": failure_classes,
        "seed": 42,
        "metrics": output_metrics,
        "assumptions": assumptions,
        "execution_summary": (
            f"Merged {len(archives)} sharded archive(s) for condition {condition} using model {model or 'default'}. "
            f"Produced {non_null}/{len(output_metrics)} non-null cells. "
            + " ".join(summaries[:3])
        ).strip(),
        "failure_diagnosis": None if status == "success" else " | ".join(failures) or f"{len(output_metrics) - non_null} cells remain null.",
        "retry_recommended": status != "success",
        "retry_scope": "cell" if status == "partial" else ("condition" if status != "success" else "none"),
        "retry_hint": "Retry only null cells with targeted shards." if status == "partial" else ("Retry this condition archive merge inputs." if status != "success" else "No retry needed."),
        "archive_merge": {
            "merged_at": now_iso(),
            "archives": [str(a) for a in archives],
            "condition": condition,
            "model": model or "default",
            "cell_sources": sources,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper-id", required=True)
    parser.add_argument("--archive", action="append", required=True)
    parser.add_argument("--conditions", default="A,B,C")
    parser.add_argument("--result-name", default="result.json")
    parser.add_argument("--merged-result-name", default="result_merged.json")
    parser.add_argument("--model", default=None)
    args = parser.parse_args()

    paper_root = CODEX_ROOT / "papers" / args.paper_id
    archives = [Path(a).resolve() for a in args.archive]
    conditions = [c.strip().upper() for c in args.conditions.split(",") if c.strip()]
    gt = collect_ground_truth(archives)
    for condition in conditions:
        result = merge_condition(args.paper_id, condition, archives, args.result_name, args.model)
        cond_dir = paper_root / "conditions" / f"condition_{condition}"
        write_json(cond_dir / "ground_truth_merged.json", gt)
        write_json(cond_dir / args.merged_result_name, result)
        print(
            f"[merge-archives] condition_{condition}: {result['status']} "
            f"{sum(m.get('value') is not None for m in result['metrics'])}/{len(result['metrics'])} non-null -> "
            f"{cond_dir / args.merged_result_name}"
        )


if __name__ == "__main__":
    main()
