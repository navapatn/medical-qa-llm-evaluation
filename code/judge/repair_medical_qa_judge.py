#!/usr/bin/env python3
"""Retry only length-truncated rows from a frozen medical-QA judge run.

The original prompt, source item, judge model, and temperature are retained.
Only the completion ceiling changes, so a length-limited response can finish
without rerunning (or mixing) the rest of the paper-production judge batch.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from llm_api import build_provider_clients
from paper_run_tracking import append_jsonl, freeze_run_inputs
from run_medical_qa_judge import (
    is_transport_usable,
    latest_rows,
    parse_judgment,
    read_json,
    read_jsonl,
    write_json,
    write_jsonl,
)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def repair(source_run: Path, out_dir: Path, max_tokens: int) -> dict[str, Any]:
    source_config = read_json(source_run / "config.snapshot.json")
    source_model = [
        model for model in source_config["models"] if model.get("enabled", True)
    ]
    if len(source_model) != 1:
        raise RuntimeError("The source run must have exactly one enabled judge")
    model = {**source_model[0], "max_tokens": max_tokens}

    source_items = read_jsonl(source_run / "planned_items.jsonl")
    by_id = {str(item["judge_item_id"]): item for item in source_items}
    raw_rows = [
        row
        for path in sorted(source_run.rglob("predictions.jsonl"))
        for row in read_jsonl(path)
    ]
    canonical = latest_rows(raw_rows)
    truncated = [
        row for row in canonical
        if row.get("truncated") and str(row.get("judge_item_id")) in by_id
    ]
    if not truncated:
        raise RuntimeError("The source run has no length-truncated rows to repair")
    items = [by_id[str(row["judge_item_id"])] for row in truncated]

    config = {
        "schema_version": "cellpress.judge_repair.v1",
        "repair_of": str(source_run),
        "repair_reason": "length_truncation",
        "source_config_sha256": source_config,
        "models": [model],
        "providers": source_config["providers"],
        "max_tokens": max_tokens,
        "expected_repairs": len(items),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / "planned_items.jsonl", items)
    freeze_run_inputs(
        out_dir=out_dir,
        config=config,
        planned_items_path=out_dir / "planned_items.jsonl",
        source_paths=[Path(__file__), Path(__file__).parent / "llm_api.py"],
        study_stage="llm_judge_length_repair",
    )

    client = build_provider_clients(
        source_config["providers"], out_dir / "api_cache"
    )[model["provider"]]
    predictions_path = out_dir / "predictions.jsonl"
    rows = read_jsonl(predictions_path)
    done = {
        str(row.get("judge_item_id"))
        for row in latest_rows(rows)
        if is_transport_usable(row)
    }
    for item in items:
        if str(item["judge_item_id"]) in done:
            continue
        try:
            output, meta = client.complete(
                model=model["model"],
                messages=item["messages"],
                temperature=float(model.get("temperature", 0)),
                max_tokens=max_tokens,
                max_tokens_field=str(model.get("max_tokens_field", "max_tokens")),
                extra_body=model.get("extra_body"),
            )
            finish_reason = meta.get("finish_reason")
            truncated_now = finish_reason in {"length", "max_tokens"}
            parsed: dict[str, Any] = {
                "target_match": None,
                "fidelity": None,
                "mapped_option": None,
                "semantic_correct": None,
                "judge_correct": None,
                "rationale": "",
            }
            parse_error = None
            if not truncated_now:
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
            truncated_now = False
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
        row = {
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
            "truncated": truncated_now,
            "api_error": api_error,
            "repair_reason": "length_truncation",
            "max_tokens_requested": max_tokens,
        }
        rows.append(row)
        append_jsonl(predictions_path, row)

    result = {
        "created_at": now_iso(),
        "source_run": str(source_run),
        "max_tokens_requested": max_tokens,
        "planned_repairs": len(items),
        "completed_repairs": sum(
            is_transport_usable(row) for row in latest_rows(rows)
        ),
        "parsed_repairs": sum(
            is_transport_usable(row) and not row.get("parse_error")
            for row in latest_rows(rows)
        ),
        "raw_rows": len(rows),
        "raw_api_cost_usd": sum(
            float((row.get("usage") or {}).get("cost") or 0)
            for row in rows if not row.get("api_error")
        ),
    }
    write_json(out_dir / "repair_manifest.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_run", type=Path)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--max-tokens", type=int, default=4096)
    args = parser.parse_args()
    if args.max_tokens < 1:
        raise ValueError("--max-tokens must be positive")
    print(json.dumps(repair(args.source_run, args.out_dir, args.max_tokens), indent=2))


if __name__ == "__main__":
    main()
