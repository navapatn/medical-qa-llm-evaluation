#!/usr/bin/env python3
"""Dataset feasibility normalization and cell-level runnable planning."""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


AVAILABILITY_STATUSES = {
    "available",
    "missing_downloadable",
    "license_gated",
    "unavailable",
    "too_large",
    "unknown",
    "partial",
}
SOURCE_RUNNABLE_DATASET_STATUSES = {"available", "missing_downloadable", "partial"}
REPRODUCTION_READY_DATASET_STATUSES = {"available"}
# Backward-compatible name used by older planning code. This means "source can
# support a future run", not "local files are already ready for execution".
RUNNABLE_DATASET_STATUSES = SOURCE_RUNNABLE_DATASET_STATUSES
DEFAULT_LOCAL_MAX_BYTES = 10 * 1024 * 1024 * 1024


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))


def dataset_tokens(name: str) -> list[str]:
    generic = {
        "dataset",
        "data",
        "corpus",
        "benchmark",
        "review",
        "reviews",
        "news",
        "topics",
        "topic",
        "payoff",
        "novelty",
        "rule",
        "tournament",
        "simulation",
        "generated",
    }
    words = [
        w for w in re.findall(r"[a-z0-9]+", (name or "").lower())
        if w not in generic and len(w) > 1
    ]
    return words or re.findall(r"[a-z0-9]+", (name or "").lower())


def names_match(left: str, right: str) -> bool:
    left_tokens = dataset_tokens(left)
    right_tokens = dataset_tokens(right)
    return bool(left_tokens and right_tokens and left_tokens == right_tokens)


def dataset_aliases(name: str) -> set[str]:
    words = dataset_tokens(name)
    aliases = {
        re.sub(r"[^a-z0-9]+", " ", (name or "").lower()).strip(),
        re.sub(r"[^a-z0-9]+", "", (name or "").lower()),
        " ".join(words),
        "".join(words),
    }
    if words:
        aliases.add(words[0])
        if len(words[0]) >= 3:
            aliases.add(words[0][:3])
        common_prefixes = {
            "amazon": "amz",
            "yahoo": "yah",
            "dbpedia": "dbp",
        }
        if words[0] in common_prefixes:
            aliases.add(common_prefixes[words[0]])
        if len(words) >= 2:
            aliases.add(f"{words[0]} {words[-1]}")
            aliases.add(f"{words[0]} {words[-1][0]}")
            if len(words[0]) >= 3:
                aliases.add(f"{words[0][:3]} {words[-1][0]}")
                aliases.add(f"{words[0][:3]}{words[-1][0]}")
            if words[0] in common_prefixes:
                prefix = common_prefixes[words[0]]
                aliases.add(f"{prefix} {words[-1][0]}")
                aliases.add(f"{prefix}{words[-1][0]}")
        aliases.add("".join(w[0] for w in words if w))
        if {"iterated", "prisoner", "dilemma"} <= set(words):
            aliases.add("ipd")
        if "poker" in words:
            aliases.add("poker")
    return {a for a in aliases if a}


def dataset_match_score(dataset_name: str, text: str) -> int:
    haystack = re.sub(r"[^a-z0-9]+", " ", (text or "").lower()).strip()
    compact = re.sub(r"[^a-z0-9]+", "", (text or "").lower())
    best = 0
    for alias in dataset_aliases(dataset_name):
        alias_norm = re.sub(r"[^a-z0-9]+", " ", alias).strip()
        alias_compact = re.sub(r"[^a-z0-9]+", "", alias)
        if not alias_norm:
            continue
        if " " in alias_norm and alias_norm in haystack:
            best = max(best, 100 + len(alias_norm))
        elif len(alias_norm) <= 4:
            if re.search(rf"(?<![a-z0-9]){re.escape(alias_norm)}(?![a-z0-9])", haystack):
                best = max(best, 50 + len(alias_norm))
        elif alias_norm in haystack:
            best = max(best, 20 + len(alias_norm))
        if alias_compact and len(alias_compact) > 2 and alias_compact in compact:
            best = max(best, 10 + len(alias_compact))
    return best


def _metric_result_context(metric: dict[str, Any], ground_truth: dict[str, Any] | None) -> str:
    if not ground_truth:
        return ""

    metric_text = " ".join(
        str(metric.get(key, ""))
        for key in [
            "table_id",
            "source",
            "target_component_id",
            "component_id",
            "artifact_id",
            "domain",
            "task_family",
            "metric",
        ]
    )
    metric_compact = re.sub(r"[^a-z0-9]+", "", metric_text.lower())
    context_parts: list[str] = []

    for artifact in ground_truth.get("expected_artifacts", []):
        if not isinstance(artifact, dict):
            continue
        identifiers = [
            artifact.get("component_id"),
            artifact.get("artifact_id"),
        ]
        artifact_source = str(artifact.get("source", ""))
        matched = False
        for identifier in identifiers:
            if not identifier:
                continue
            identifier_compact = re.sub(r"[^a-z0-9]+", "", str(identifier).lower())
            if identifier_compact and identifier_compact in metric_compact:
                matched = True
                break
        if not matched and artifact_source:
            source_compact = re.sub(r"[^a-z0-9]+", "", artifact_source.lower())
            matched = bool(metric_compact and metric_compact in source_compact)
        if not matched:
            continue

        context_parts.extend(
            str(artifact.get(key, ""))
            for key in [
                "component_id",
                "artifact_id",
                "target_type",
                "artifact_kind",
                "metric",
                "domain",
                "task_family",
                "source",
                "rationale_for_inclusion",
            ]
        )
        scoring = artifact.get("scoring_guidance")
        if isinstance(scoring, dict):
            context_parts.extend(str(value) for value in scoring.values())

    return " ".join(part for part in context_parts if part)


def infer_dataset_for_cell(
    metric: dict[str, Any],
    datasets: list[dict[str, Any]],
    ground_truth: dict[str, Any] | None = None,
) -> str | None:
    text = " ".join(
        str(metric.get(key, ""))
        for key in [
            "row_id",
            "col_id",
            "table_id",
            "source",
            "target_type",
            "target_component_id",
            "component_id",
            "artifact_id",
            "domain",
            "task_family",
            "metric",
            "notes",
        ]
    )
    extra_context = _metric_result_context(metric, ground_truth)
    if extra_context:
        text = f"{text} {extra_context}"
    best_name = None
    best_score = 0
    for dataset in datasets:
        name = dataset.get("dataset_name", "")
        score = dataset_match_score(name, text)
        if score > best_score:
            best_name = name
            best_score = score
    return best_name


def _entry_text(entry: dict[str, Any]) -> str:
    values = [
        entry.get("status"),
        entry.get("availability_status"),
        entry.get("canonical_source"),
        entry.get("kind"),
        entry.get("user_instructions"),
        entry.get("reason"),
        entry.get("block_reason"),
        " ".join(str(e) for e in entry.get("errors_encountered", []) if e),
    ]
    return " ".join(str(v) for v in values if v).lower()


def _bytes_from_entry(entry: dict[str, Any]) -> int | None:
    for key in ["estimated_download_size_bytes", "size_bytes", "download_size_bytes"]:
        value = entry.get(key)
        if isinstance(value, int):
            return value
        if isinstance(value, str) and value.isdigit():
            return int(value)
    evidence = entry.get("probe_evidence") if isinstance(entry.get("probe_evidence"), dict) else {}
    value = evidence.get("content_length")
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return None


def max_local_dataset_bytes() -> int:
    raw = os.environ.get("CODEX_MAX_LOCAL_DATASET_BYTES")
    if raw and raw.isdigit():
        return int(raw)
    return DEFAULT_LOCAL_MAX_BYTES


def local_path_stats(path: Path) -> dict[str, Any]:
    stats: dict[str, Any] = {
        "path": str(path),
        "exists": path.exists(),
        "is_file": path.is_file(),
        "is_dir": path.is_dir(),
        "file_count": 0,
        "size_bytes": 0,
        "non_empty": False,
        "errors": [],
    }
    if not path.exists():
        return stats
    try:
        if path.is_file():
            stats["size_bytes"] = path.stat().st_size
            stats["file_count"] = 1
        elif path.is_dir():
            total = 0
            count = 0
            for item in path.rglob("*"):
                if item.is_file():
                    count += 1
                    total += item.stat().st_size
            stats["size_bytes"] = total
            stats["file_count"] = count
        stats["non_empty"] = bool(stats["file_count"] and stats["size_bytes"] > 0)
    except OSError as exc:
        stats["errors"].append(str(exc))
    return stats


def prepared_dataset_map(data_manifest: dict[str, Any], paper_root: Path) -> dict[str, dict[str, Any]]:
    prepared: dict[str, dict[str, Any]] = {}
    for item in data_manifest.get("datasets_prepared", []):
        if not isinstance(item, dict) or not item.get("dataset_name"):
            continue
        local_path = item.get("local_path")
        path = paper_root / local_path if local_path else paper_root / "data" / item["dataset_name"]
        prepared[item["dataset_name"]] = {
            **item,
            "local_validation": local_path_stats(path),
        }
    return prepared


def find_prepared(name: str, prepared: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    for prepared_name, item in prepared.items():
        if names_match(name, prepared_name):
            return item
    return None


def classify_availability(entry: dict[str, Any], *, local_prepared: dict[str, Any] | None = None) -> tuple[str, str, str]:
    local_valid = bool(
        local_prepared
        and local_prepared.get("status") == "success"
        and local_prepared.get("local_validation", {}).get("non_empty")
    )
    if local_valid:
        return "available", "Local dataset files are present and non-empty.", "use_local_data"

    explicit = str(entry.get("availability_status") or "").strip().lower()
    if explicit in AVAILABILITY_STATUSES:
        if explicit == "too_large":
            return explicit, "Dataset exceeds the configured local size limit.", "run_on_azure_or_prepare_remote"
        return explicit, entry.get("reason") or entry.get("user_instructions") or f"Dataset is {explicit}.", _default_action(explicit)

    text = _entry_text(entry)
    size_bytes = _bytes_from_entry(entry)
    if size_bytes is not None and size_bytes > max_local_dataset_bytes():
        return "too_large", f"Estimated dataset size {size_bytes} bytes exceeds local limit.", "run_on_azure_or_prepare_remote"
    if any(term in text for term in ["license", "gated", "auth", "401", "403", "kaggle", "imagenet", "token", "credential"]):
        return "license_gated", entry.get("user_instructions") or "Dataset requires credentials, account access, or license acceptance.", "obtain_license_or_credentials"

    status = str(entry.get("status") or "").strip().lower()
    if status in {"feasible", "ready"}:
        return "missing_downloadable", "Dataset source was probed successfully but local bytes are not present yet.", "prepare_data"
    if status in {"manual_preparation_required", "manual_required"}:
        return "missing_downloadable", entry.get("user_instructions") or "Dataset must be placed locally before running.", "manual_prepare_data"
    if status in {"unable_to_obtain", "unavailable", "download_failed"}:
        return "unavailable", entry.get("user_instructions") or entry.get("reason") or "Dataset could not be obtained from the stated source.", "find_alternative_or_drop_cells"
    if status in {"unknown", ""}:
        return "unknown", entry.get("reason") or "Dataset availability was not established.", "rerun_feasibility_probe"
    return "unknown", f"Unrecognized dataset feasibility status: {status}", "rerun_feasibility_probe"


def reproduction_readiness_for_dataset(availability: str, local_validation: dict[str, Any]) -> tuple[bool, str, str]:
    """Return strict execution readiness for a dataset.

    Source availability and reproduction readiness are intentionally separate:
    `missing_downloadable` is useful for setup screening, but the reproduction
    agent cannot execute until local bytes exist and validation passes.
    """
    if availability == "available" and local_validation.get("non_empty"):
        return True, "ready", "Local dataset files are present and non-empty."
    if availability in {"missing_downloadable", "partial"}:
        return False, "needs_data_preparation", "Dataset source is available, but local prepared files are missing."
    if availability == "available":
        return False, "needs_local_validation", "Dataset was marked available, but local validation did not find non-empty files."
    return False, "blocked", f"Dataset availability is {availability}."


def _default_action(status: str) -> str:
    return {
        "available": "use_local_data",
        "missing_downloadable": "prepare_data",
        "license_gated": "obtain_license_or_credentials",
        "unavailable": "find_alternative_or_drop_cells",
        "too_large": "run_on_azure_or_prepare_remote",
        "unknown": "rerun_feasibility_probe",
        "partial": "run_available_cells_and_prepare_missing_parts",
    }.get(status, "inspect_dataset")


def normalize_feasibility_manifest(
    *,
    paper_id: str,
    paper_root: Path,
    constraints: dict[str, Any],
    ground_truth: dict[str, Any],
    feasibility: dict[str, Any],
    data_manifest: dict[str, Any] | None = None,
) -> dict[str, Any]:
    datasets = [d for d in constraints.get("datasets", []) if isinstance(d, dict)]
    checks_by_name = {
        item.get("dataset_name", ""): item
        for item in feasibility.get("datasets_checked", [])
        if isinstance(item, dict) and item.get("dataset_name")
    }
    prepared = prepared_dataset_map(data_manifest or {}, paper_root)
    dataset_entries = []
    dataset_status: dict[str, str] = {}
    status_counts = {status: 0 for status in sorted(AVAILABILITY_STATUSES)}

    for dataset in datasets:
        name = dataset.get("dataset_name", "")
        check = checks_by_name.get(name) or next(
            (item for item_name, item in checks_by_name.items() if names_match(name, item_name)),
            {},
        )
        local_prepared = find_prepared(name, prepared)
        availability, reason, action = classify_availability(check or dataset, local_prepared=local_prepared)
        status_counts[availability] += 1
        dataset_status[name] = availability
        local_validation = (local_prepared or {}).get("local_validation") or local_path_stats(paper_root / "data" / name)
        reproduction_ready, readiness_status, readiness_reason = reproduction_readiness_for_dataset(
            availability,
            local_validation,
        )
        entry = {
            **check,
            "dataset_name": name,
            "canonical_source": check.get("canonical_source") or dataset.get("canonical_source"),
            "availability_status": availability,
            "source_available": availability in SOURCE_RUNNABLE_DATASET_STATUSES,
            "runnable": availability in RUNNABLE_DATASET_STATUSES,
            "reproduction_ready": reproduction_ready,
            "readiness_status": readiness_status,
            "readiness_reason": readiness_reason,
            "requires_data_preparation": readiness_status in {"needs_data_preparation", "needs_local_validation"},
            "block_reason": None if availability in SOURCE_RUNNABLE_DATASET_STATUSES else reason,
            "suggested_next_action": action,
            "local_validation": local_validation,
        }
        if not entry.get("reason"):
            entry["reason"] = reason
        dataset_entries.append(entry)

    runnable_cells = []
    reproduction_ready_cells = []
    preparation_required_cells = []
    blocked_cells = []
    unknown_dataset_cells = []
    runnable_dataset_entries = [entry for entry in dataset_entries if entry.get("runnable")]
    for metric in ground_truth.get("expected_metrics", []):
        if not isinstance(metric, dict):
            continue
        dataset_name = infer_dataset_for_cell(metric, datasets, ground_truth)
        dataset_inference = None
        if not dataset_name and len(datasets) == 1:
            dataset_name = datasets[0].get("dataset_name")
            dataset_inference = "single_dataset_fallback" if dataset_name else None
        elif not dataset_name and len(runnable_dataset_entries) == 1:
            dataset_name = runnable_dataset_entries[0].get("dataset_name")
            dataset_inference = "single_runnable_dataset_fallback" if dataset_name else None
        cell = {
            "table_id": metric.get("table_id"),
            "row_id": metric.get("row_id"),
            "col_id": metric.get("col_id"),
            "dataset_name": dataset_name,
        }
        if dataset_inference:
            cell["dataset_inference"] = dataset_inference
        if not dataset_name:
            unknown_dataset_cells.append({
                **cell,
                "availability_status": "unknown",
                "blocked_reason": "Could not infer the required dataset for this result cell.",
                "suggested_next_action": "add_dataset_mapping",
            })
            continue
        availability = dataset_status.get(dataset_name, "unknown")
        dataset_entry = next((d for d in dataset_entries if d["dataset_name"] == dataset_name), {})
        if availability in RUNNABLE_DATASET_STATUSES:
            source_cell = {
                **cell,
                "availability_status": availability,
                "reproduction_ready": bool(dataset_entry.get("reproduction_ready")),
                "readiness_status": dataset_entry.get("readiness_status"),
            }
            runnable_cells.append(source_cell)
            if dataset_entry.get("reproduction_ready"):
                reproduction_ready_cells.append(source_cell)
            else:
                preparation_required_cells.append({
                    **source_cell,
                    "blocked_reason": dataset_entry.get("readiness_reason") or "Dataset must be prepared locally before reproduction.",
                    "suggested_next_action": dataset_entry.get("suggested_next_action") or "prepare_data",
                })
        else:
            blocked_cells.append({
                **cell,
                "availability_status": availability,
                "blocked_reason": dataset_entry.get("block_reason") or dataset_entry.get("reason") or "Dataset is not runnable.",
                "suggested_next_action": dataset_entry.get("suggested_next_action") or _default_action(availability),
            })

    n_cells = len(runnable_cells) + len(blocked_cells) + len(unknown_dataset_cells)
    all_dataset_statuses = set(dataset_status.values())
    if n_cells and not blocked_cells and not unknown_dataset_cells:
        source_overall = "ready"
    elif n_cells and runnable_cells and (blocked_cells or unknown_dataset_cells):
        source_overall = "partial"
    elif n_cells and (blocked_cells or unknown_dataset_cells):
        source_overall = "blocked"
    elif all_dataset_statuses and all_dataset_statuses <= {"available"}:
        source_overall = "ready"
    elif all_dataset_statuses and all_dataset_statuses <= RUNNABLE_DATASET_STATUSES:
        source_overall = "ready"
    elif not datasets:
        source_overall = "ready"
    else:
        source_overall = "blocked"

    if n_cells == 0 and not ground_truth.get("expected_artifacts") and not datasets:
        reproduction_overall = "no_targets"
    elif n_cells and len(reproduction_ready_cells) == n_cells:
        reproduction_overall = "ready"
    elif n_cells and reproduction_ready_cells:
        reproduction_overall = "partial"
    elif n_cells and runnable_cells:
        reproduction_overall = "needs_data_preparation"
    elif n_cells:
        reproduction_overall = "blocked"
    elif all_dataset_statuses and all_dataset_statuses <= {"available"}:
        reproduction_overall = "ready"
    elif not datasets:
        reproduction_overall = "no_targets"
    else:
        reproduction_overall = "blocked"

    return {
        "paper_id": paper_id,
        "generated_at": now_iso(),
        "schema_version": "dataset_gate.v2",
        "overall_status": source_overall,
        "source_availability_status": source_overall,
        "reproduction_readiness_status": reproduction_overall,
        "ready_for_reproduction": reproduction_overall == "ready",
        "allow_partial_execution": bool(runnable_cells),
        "allow_source_screening_execution": bool(runnable_cells),
        "requires_data_preparation": bool(preparation_required_cells),
        "availability_statuses": sorted(AVAILABILITY_STATUSES),
        "dataset_status_counts": status_counts,
        "datasets": dataset_entries,
        "runnable_cells": runnable_cells,
        "reproduction_ready_cells": reproduction_ready_cells,
        "preparation_required_cells": preparation_required_cells,
        "blocked_cells": blocked_cells + unknown_dataset_cells,
        "n_ground_truth_cells": n_cells,
        "n_runnable_cells": len(runnable_cells),
        "n_reproduction_ready_cells": len(reproduction_ready_cells),
        "n_preparation_required_cells": len(preparation_required_cells),
        "n_blocked_cells": len(blocked_cells) + len(unknown_dataset_cells),
        "no_reproducible_targets": n_cells == 0 and not ground_truth.get("expected_artifacts") and not datasets,
        "blocked_dataset_reasons": [
            {
                "dataset_name": d["dataset_name"],
                "availability_status": d["availability_status"],
                "reason": d.get("block_reason") or d.get("reason"),
                "suggested_next_action": d.get("suggested_next_action"),
            }
            for d in dataset_entries
            if not d.get("runnable")
        ],
    }


def write_dataset_gate_files(paper_root: Path, gate: dict[str, Any]) -> None:
    write_json(paper_root / "data_feasibility_plan.json", gate)
    write_json(
        paper_root / "runnable_cells.json",
        {
            "paper_id": gate["paper_id"],
            "generated_at": gate["generated_at"],
            "cells": gate["runnable_cells"],
            "n_cells": gate["n_runnable_cells"],
        },
    )
    write_json(
        paper_root / "blocked_cells.json",
        {
            "paper_id": gate["paper_id"],
            "generated_at": gate["generated_at"],
            "cells": gate["blocked_cells"],
            "n_cells": gate["n_blocked_cells"],
            "blocked_dataset_reasons": gate["blocked_dataset_reasons"],
        },
    )
    write_json(
        paper_root / "reproduction_ready_cells.json",
        {
            "paper_id": gate["paper_id"],
            "generated_at": gate["generated_at"],
            "cells": gate.get("reproduction_ready_cells", []),
            "n_cells": gate.get("n_reproduction_ready_cells", 0),
        },
    )
    write_json(
        paper_root / "preparation_required_cells.json",
        {
            "paper_id": gate["paper_id"],
            "generated_at": gate["generated_at"],
            "cells": gate.get("preparation_required_cells", []),
            "n_cells": gate.get("n_preparation_required_cells", 0),
        },
    )
