#!/usr/bin/env python3
"""Run a frozen LLM-as-a-judge plan for the medical-QA extensions."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from llm_api import UNUSABLE_FINISH_REASONS, build_provider_clients, stable_hash
from paper_run_tracking import (
    append_jsonl,
    file_sha256,
    freeze_run_inputs,
    record_cost_checkpoint,
)
from replicate_medical_reasoning import row_cost


CODEX_ROOT = Path(__file__).resolve().parent
PAPERS_ROOT = CODEX_ROOT / "papers"
TASKS = {
    "question_reconstruction_fidelity",
    "free_answer_option_mapping",
    "free_answer_semantic_correctness",
}
MAPPED_OPTIONS = {"A", "B", "C", "D", "NONE", "AMBIGUOUS"}


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
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    )


def is_transport_usable(row: dict[str, Any]) -> bool:
    return (
        not row.get("api_error")
        and not row.get("truncated", False)
        and row.get("finish_reason") not in UNUSABLE_FINISH_REASONS
    )


def latest_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return the first usable response per judge/model key."""

    chosen: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        key = (str(row.get("model")), str(row.get("judge_item_id")))
        existing = chosen.get(key)
        if existing is None or (
            not is_transport_usable(existing) and is_transport_usable(row)
        ):
            chosen[key] = row
    return list(chosen.values())


def _json_object(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start < 0 or end < start:
        raise ValueError("No JSON object found")
    payload = json.loads(cleaned[start:end + 1])
    if not isinstance(payload, dict):
        raise ValueError("Judge output is not a JSON object")
    return payload


def parse_judgment(task: str, output: str) -> dict[str, Any]:
    payload = _json_object(output)
    rationale = str(payload.get("rationale", "")).strip()
    if task == "question_reconstruction_fidelity":
        target_match = payload.get("same", payload.get("target_match"))
        if isinstance(target_match, str):
            normalized = target_match.strip().lower()
            if normalized in {"true", "false"}:
                target_match = normalized == "true"
        if not isinstance(target_match, bool):
            raise ValueError("target_match must be boolean")
        return {
            "target_match": target_match,
            "fidelity": None,
            "mapped_option": None,
            "judge_correct": None,
            "rationale": rationale,
        }
    if task == "free_answer_option_mapping":
        mapped_option = str(payload.get("mapped_option", "")).strip().upper()
        mapped_option = re.sub(r"[^A-Z]", "", mapped_option)
        if mapped_option not in MAPPED_OPTIONS:
            raise ValueError(f"Unsupported mapped_option: {mapped_option!r}")
        return {
            "target_match": None,
            "fidelity": None,
            "mapped_option": mapped_option,
            "semantic_correct": None,
            "judge_correct": None,
            "rationale": rationale,
        }
    if task == "free_answer_semantic_correctness":
        semantic_correct = payload.get("correct", payload.get("semantic_correct"))
        if isinstance(semantic_correct, str):
            normalized = semantic_correct.strip().lower()
            if normalized in {"true", "false"}:
                semantic_correct = normalized == "true"
        if not isinstance(semantic_correct, bool):
            raise ValueError("semantic_correct must be boolean")
        return {
            "target_match": None,
            "fidelity": None,
            "mapped_option": None,
            "semantic_correct": semantic_correct,
            "judge_correct": semantic_correct,
            "rationale": rationale,
        }
    raise ValueError(f"Unknown judge task: {task}")


def shard_items(
    items: list[dict[str, Any]], *, shard_count: int, shard_index: int
) -> list[dict[str, Any]]:
    if shard_count < 1:
        raise ValueError("item_shard_count must be at least 1")
    if not 0 <= shard_index < shard_count:
        raise ValueError(
            "item_shard_index must be between 0 and item_shard_count - 1"
        )
    if shard_count == 1:
        return items
    return [
        item for position, item in enumerate(items)
        if position % shard_count == shard_index
    ]


def checkpoint(
    *,
    config: dict[str, Any],
    items: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    out_dir: Path,
    model: dict[str, Any],
    stopped_for_cost: bool = False,
) -> dict[str, Any]:
    canonical = latest_rows(rows)
    usable = [row for row in canonical if is_transport_usable(row)]
    parsed = [row for row in usable if not row.get("parse_error")]
    manifest = {
        "schema_version": "cellpress.judge_run.v1",
        "generated_at": now_iso(),
        "config_sha256": stable_hash(config),
        "source_model": config.get("source_model"),
        "judge_model": model,
        "planned_requests": len(items),
        "completed_requests": len(usable),
        "parsed_requests": len(parsed),
        "parse_error_requests": len(usable) - len(parsed),
        "raw_rows": len(rows),
        "observed_cost_usd": sum(
            row_cost(row, model) for row in usable
        ),
        "raw_api_charges_usd": sum(
            row_cost(row, model) for row in rows if not row.get("api_error")
        ),
        "max_cost_usd": float(config.get("max_cost_usd", 15)),
        "stopped_for_cost": stopped_for_cost,
        "task_counts": dict(sorted(Counter(item["task"] for item in items).items())),
        "response_models": sorted({
            str(row.get("response_model")) for row in usable
        }),
        "outputs": {
            "planned_items": str(out_dir / "planned_items.jsonl"),
            "predictions": str(out_dir / "predictions.jsonl"),
            "config_snapshot": str(out_dir / "config.snapshot.json"),
            "frozen_run_inputs": str(out_dir / "frozen_run_inputs.json"),
            "cost_checkpoints_csv": str(out_dir / "cost_checkpoints.csv"),
        },
    }
    write_json(out_dir / "run_manifest.json", manifest)
    return manifest


def run(
    config: dict[str, Any],
    *,
    paper_root: Path,
    out_dir: Path,
    execute_api: bool,
    max_new_requests: int | None = None,
) -> dict[str, Any]:
    enabled = [model for model in config.get("models", []) if model.get("enabled", True)]
    if len(enabled) != 1:
        raise RuntimeError("Each judge run requires exactly one enabled model")
    model = enabled[0]

    source_path = paper_root / str(config["judge_items_path"])
    expected_sha = str(config.get("judge_items_sha256", ""))
    actual_sha = file_sha256(source_path)
    if expected_sha and expected_sha != actual_sha:
        raise RuntimeError(
            f"Judge source hash mismatch: expected={expected_sha}, actual={actual_sha}"
        )
    full_items = read_jsonl(source_path)
    if len(full_items) != int(config.get("expected_judge_items", len(full_items))):
        raise RuntimeError("Judge item count does not match the frozen configuration")
    if len({item["judge_item_id"] for item in full_items}) != len(full_items):
        raise RuntimeError("Judge item IDs are not unique")
    if {item["task"] for item in full_items} - TASKS:
        raise RuntimeError("Judge plan contains an unsupported task")
    if {item["source_model"] for item in full_items} != {config["source_model"]}:
        raise RuntimeError("Judge plan source model differs from configuration")

    items = shard_items(
        full_items,
        shard_count=int(config.get("item_shard_count", 1)),
        shard_index=int(config.get("item_shard_index", 0)),
    )
    if len(items) > int(config.get("max_requests", len(full_items))):
        raise RuntimeError("Judge plan exceeds max_requests")

    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / "planned_items.jsonl", items)
    freeze_run_inputs(
        out_dir=out_dir,
        config=config,
        planned_items_path=out_dir / "planned_items.jsonl",
        source_paths=[Path(__file__), CODEX_ROOT / "llm_api.py", CODEX_ROOT / "paper_run_tracking.py"],
        study_stage="llm_judge",
    )

    providers = build_provider_clients(config["providers"], out_dir / "api_cache")
    client = providers[model["provider"]]
    predictions_path = out_dir / "predictions.jsonl"
    rows = read_jsonl(predictions_path)
    completed = {
        (row.get("model"), row.get("judge_item_id"))
        for row in latest_rows(rows)
        if is_transport_usable(row)
    }
    pending = [
        item for item in items
        if (model["model"], item["judge_item_id"]) not in completed
    ]
    if max_new_requests is not None:
        if max_new_requests < 1:
            raise ValueError("max_new_requests must be positive")
        pending = pending[:max_new_requests]
    if not execute_api:
        return checkpoint(
            config=config, items=items, rows=rows, out_dir=out_dir, model=model
        )

    workers = int(config.get("max_workers", 8))
    progress_every = int(config.get("progress_every", 200))
    max_cost = float(config.get("max_cost_usd", 15))
    stopped_for_cost = False

    def call(item: dict[str, Any]) -> dict[str, Any]:
        try:
            output, meta = client.complete(
                model=model["model"],
                messages=item["messages"],
                temperature=float(model.get("temperature", 0)),
                max_tokens=int(model.get("max_tokens", 192)),
                max_tokens_field=str(model.get("max_tokens_field", "max_tokens")),
                extra_body=model.get("extra_body"),
            )
            finish_reason = meta.get("finish_reason")
            truncated = finish_reason in {"length", "max_tokens"}
            parsed: dict[str, Any] = {
                "target_match": None,
                "fidelity": None,
                "mapped_option": None,
                "semantic_correct": None,
                "judge_correct": None,
                "rationale": "",
            }
            parse_error = None
            if not truncated:
                try:
                    parsed = parse_judgment(item["task"], output)
                    if item["task"] == "free_answer_option_mapping":
                        parsed["judge_correct"] = (
                            parsed["mapped_option"] == item["gold_option"]
                        )
                except Exception as exc:  # noqa: BLE001
                    parse_error = str(exc)
            api_error = None
        except Exception as exc:  # noqa: BLE001
            output = ""
            meta = {"usage": {}, "cache_hit": False, "finish_reason": None}
            finish_reason = None
            truncated = False
            parsed = {
                "target_match": None,
                "fidelity": None,
                "mapped_option": None,
                "semantic_correct": None,
                "judge_correct": None,
                "rationale": "",
            }
            parse_error = None
            api_error = str(exc)
        return {
            "created_at": now_iso(),
            "provider": model["provider"],
            "model_label": model.get("label") or model["model"],
            "model": model["model"],
            **{key: item.get(key) for key in [
                "judge_item_id", "task", "dataset", "example_id",
                "source_generation_item_id", "source_model", "source_response_id",
                "messages_sha256", "gold_option",
            ]},
            "output": output,
            **parsed,
            "parse_error": parse_error,
            "usage": meta.get("usage", {}),
            "cache_hit": meta.get("cache_hit", False),
            "finish_reason": finish_reason,
            "response_id": meta.get("response_id"),
            "response_model": meta.get("response_model"),
            "truncated": truncated,
            "api_error": api_error,
        }

    index = 0
    while index < len(pending):
        current_cost = sum(
            row_cost(row, model) for row in latest_rows(rows)
            if not row.get("api_error")
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
                if is_transport_usable(row):
                    completed.add((row["model"], row["judge_item_id"]))
                if len(completed) > 0 and len(completed) % progress_every == 0:
                    manifest = checkpoint(
                        config=config, items=items, rows=rows, out_dir=out_dir,
                        model=model,
                    )
                    record_cost_checkpoint(
                        out_dir=out_dir,
                        study_stage="llm_judge",
                        planned_rows=len(items),
                        raw_rows=rows,
                        canonical_rows=latest_rows(rows),
                        cost_fn=lambda prediction: row_cost(prediction, model),
                        interval=progress_every,
                    )
                    print(
                        f"[medical-judge] progress={manifest['completed_requests']}/"
                        f"{manifest['planned_requests']} cost="
                        f"${manifest['observed_cost_usd']:.4f}",
                        flush=True,
                    )
        index += len(batch)

    manifest = checkpoint(
        config=config, items=items, rows=rows, out_dir=out_dir, model=model,
        stopped_for_cost=stopped_for_cost,
    )
    record_cost_checkpoint(
        out_dir=out_dir,
        study_stage="llm_judge",
        planned_rows=len(items),
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
    parser.add_argument("--max-new-requests", type=int)
    parser.add_argument("--max-cost-usd", type=float)
    parser.add_argument("--item-shard-count", type=int)
    parser.add_argument("--item-shard-index", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = read_json(args.config)
    if args.max_cost_usd is not None:
        config["max_cost_usd"] = args.max_cost_usd
    if args.item_shard_count is not None:
        config["item_shard_count"] = args.item_shard_count
    if args.item_shard_index is not None:
        config["item_shard_index"] = args.item_shard_index
    paper_root = PAPERS_ROOT / args.paper_id
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_dir or paper_root / "runs" / f"medical_judge_{stamp}"
    manifest = run(
        config,
        paper_root=paper_root,
        out_dir=out_dir,
        execute_api=args.execute_api,
        max_new_requests=args.max_new_requests,
    )
    print(
        f"[medical-judge] mode={'api' if args.execute_api else 'plan'} "
        f"calls={manifest['planned_requests']} completed="
        f"{manifest['completed_requests']} cost="
        f"${manifest['observed_cost_usd']:.4f}"
    )


if __name__ == "__main__":
    main()
