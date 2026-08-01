"""Immutable run inputs and append-only cost tracking for paper experiments."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable


SCHEMA_VERSION = "cellpress.paper_run.v1"
UNUSABLE_FINISH_REASONS = {"error"}


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def stable_json_hash(payload: Any) -> str:
    raw = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(errors="replace") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def freeze_run_inputs(
    *,
    out_dir: Path,
    config: dict[str, Any],
    planned_items_path: Path,
    source_paths: Iterable[Path],
    study_stage: str,
) -> dict[str, Any]:
    """Freeze the exact configuration, plan, and runner source hashes.

    Resuming into an existing output directory is allowed only when the
    scientific configuration and planned request file are byte-for-byte
    equivalent. This prevents predictions from different protocols from being
    silently mixed in one paper result.
    """

    out_dir.mkdir(parents=True, exist_ok=True)
    config_snapshot = out_dir / "config.snapshot.json"
    if config_snapshot.exists():
        existing = json.loads(config_snapshot.read_text())
        if stable_json_hash(existing) != stable_json_hash(config):
            raise RuntimeError(
                "Refusing to resume: config.snapshot.json differs from the "
                "current experimental configuration. Use a new output directory."
            )
    else:
        write_json(config_snapshot, config)

    source_hashes = {
        str(path.resolve()): file_sha256(path)
        for path in source_paths
        if path.exists() and path.is_file()
    }
    payload = {
        "schema_version": SCHEMA_VERSION,
        "study_stage": study_stage,
        "config_sha256": stable_json_hash(config),
        "planned_items_sha256": file_sha256(planned_items_path),
        "source_sha256": source_hashes,
        "artifacts": {
            "config_snapshot": str(config_snapshot),
            "planned_items": str(planned_items_path),
        },
    }
    frozen_path = out_dir / "frozen_run_inputs.json"
    if frozen_path.exists():
        existing = json.loads(frozen_path.read_text())
        comparable = {
            key: value for key, value in existing.items() if key != "frozen_at"
        }
        if comparable != payload:
            raise RuntimeError(
                "Refusing to resume: frozen run inputs differ from the existing "
                "paper run. Use a new output directory."
            )
        return existing

    payload = {"frozen_at": now_iso(), **payload}
    write_json(frozen_path, payload)
    return payload


def usage_counts(usage: dict[str, Any]) -> dict[str, int]:
    input_tokens = int(
        usage.get("prompt_tokens", usage.get("input_tokens", 0)) or 0
    )
    output_tokens = int(
        usage.get("completion_tokens", usage.get("output_tokens", 0)) or 0
    )
    prompt_details = usage.get("prompt_tokens_details", {}) or {}
    completion_details = usage.get("completion_tokens_details", {}) or {}
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "reasoning_tokens": int(completion_details.get("reasoning_tokens", 0) or 0),
        "cached_input_tokens": int(prompt_details.get("cached_tokens", 0) or 0),
        "cache_write_tokens": int(prompt_details.get("cache_write_tokens", 0) or 0),
    }


def is_successful(row: dict[str, Any]) -> bool:
    return (
        not row.get("api_error")
        and not row.get("truncated", False)
        and row.get("finish_reason") not in UNUSABLE_FINISH_REASONS
    )


def _aggregate(
    rows: list[dict[str, Any]], cost_fn: Callable[[dict[str, Any]], float]
) -> dict[str, Any]:
    totals = {
        "calls": len(rows),
        "input_tokens": 0,
        "output_tokens": 0,
        "reasoning_tokens": 0,
        "cached_input_tokens": 0,
        "cache_write_tokens": 0,
        "cost_usd": 0.0,
    }
    for row in rows:
        counts = usage_counts(row.get("usage", {}) or {})
        for key, value in counts.items():
            totals[key] += value
        totals["cost_usd"] += cost_fn(row)
    return totals


def _breakdown(
    rows: list[dict[str, Any]],
    *,
    fields: tuple[str, ...],
    cost_fn: Callable[[dict[str, Any]], float],
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(str(row.get(field, "")) for field in fields)].append(row)
    output = []
    for key, group in sorted(groups.items()):
        output.append(
            {
                **dict(zip(fields, key)),
                **_aggregate(group, cost_fn),
            }
        )
    return output


def _write_ledger_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "created_at",
        "study_stage",
        "checkpoint_kind",
        "checkpoint_target",
        "completed_successful_rows",
        "planned_rows",
        "raw_prediction_rows",
        "failed_or_incomplete_rows",
        "input_tokens",
        "output_tokens",
        "reasoning_tokens",
        "cached_input_tokens",
        "cache_write_tokens",
        "observed_cost_usd",
        "raw_api_charges_usd",
        "incremental_raw_cost_usd",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            cumulative = row["canonical_usage"]
            writer.writerow({
                "created_at": row["created_at"],
                "study_stage": row["study_stage"],
                "checkpoint_kind": row["checkpoint_kind"],
                "checkpoint_target": row.get("checkpoint_target"),
                "completed_successful_rows": row["completed_successful_rows"],
                "planned_rows": row["planned_rows"],
                "raw_prediction_rows": row["raw_prediction_rows"],
                "failed_or_incomplete_rows": row["failed_or_incomplete_rows"],
                "input_tokens": cumulative["input_tokens"],
                "output_tokens": cumulative["output_tokens"],
                "reasoning_tokens": cumulative["reasoning_tokens"],
                "cached_input_tokens": cumulative["cached_input_tokens"],
                "cache_write_tokens": cumulative["cache_write_tokens"],
                "observed_cost_usd": row["observed_cost_usd"],
                "raw_api_charges_usd": row["raw_api_charges_usd"],
                "incremental_raw_cost_usd": row["incremental_raw_cost_usd"],
            })


def record_cost_checkpoint(
    *,
    out_dir: Path,
    study_stage: str,
    planned_rows: int,
    raw_rows: list[dict[str, Any]],
    canonical_rows: list[dict[str, Any]],
    cost_fn: Callable[[dict[str, Any]], float],
    interval: int,
    force_final: bool = False,
) -> dict[str, Any] | None:
    """Append a cost checkpoint at each exact success interval and at final."""

    if interval < 1:
        raise ValueError("Cost checkpoint interval must be positive")
    successful = [row for row in canonical_rows if is_successful(row)]
    completed = len(successful)
    is_interval = completed > 0 and completed % interval == 0
    if not is_interval and not force_final:
        return None

    ledger_path = out_dir / "cost_checkpoints.jsonl"
    ledger = read_jsonl(ledger_path)
    checkpoint_kind = (
        "interval"
        if is_interval
        else ("final" if completed >= planned_rows else "pause")
    )
    checkpoint_target = completed if is_interval else None
    identity = (checkpoint_kind, checkpoint_target, completed, len(raw_rows))
    if ledger:
        last = ledger[-1]
        last_identity = (
            last.get("checkpoint_kind"),
            last.get("checkpoint_target"),
            last.get("completed_successful_rows"),
            last.get("raw_prediction_rows"),
        )
        if last_identity == identity:
            return last

    canonical_usage = _aggregate(successful, cost_fn)
    billed_raw = [row for row in raw_rows if not row.get("api_error")]
    raw_usage = _aggregate(billed_raw, cost_fn)
    previous_raw_cost = ledger[-1]["raw_api_charges_usd"] if ledger else 0.0
    payload = {
        "schema_version": SCHEMA_VERSION,
        "created_at": now_iso(),
        "study_stage": study_stage,
        "checkpoint_kind": checkpoint_kind,
        "checkpoint_target": checkpoint_target,
        "completed_successful_rows": completed,
        "planned_rows": planned_rows,
        "raw_prediction_rows": len(raw_rows),
        "failed_or_incomplete_rows": len(canonical_rows) - completed,
        "canonical_usage": canonical_usage,
        "raw_usage": raw_usage,
        "observed_cost_usd": canonical_usage["cost_usd"],
        "raw_api_charges_usd": raw_usage["cost_usd"],
        "incremental_raw_cost_usd": raw_usage["cost_usd"] - previous_raw_cost,
        "by_dataset": _breakdown(
            successful, fields=("dataset",), cost_fn=cost_fn
        ),
        "by_dataset_and_variant": _breakdown(
            successful, fields=("dataset", "variant"), cost_fn=cost_fn
        ),
    }
    append_jsonl(ledger_path, payload)
    ledger.append(payload)
    _write_ledger_csv(out_dir / "cost_checkpoints.csv", ledger)
    write_json(out_dir / "cost_status.json", payload)
    return payload
