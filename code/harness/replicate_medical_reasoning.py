#!/usr/bin/env python3
"""Reproduce the main medical-reasoning benchmark with current chat models.

The runner intentionally implements one primary condition: five-shot
chain-of-thought with local answer extraction and nested self-consistency.
Generating the largest configured sample count once is sufficient to report all
smaller k values.  Transport is provider-neutral and supports mixing Foundry and
OpenRouter deployments in the same run.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

CODEX_ROOT = Path(__file__).resolve().parent
if str(CODEX_ROOT) not in sys.path:
    sys.path.insert(0, str(CODEX_ROOT))

from llm_api import (  # noqa: E402
    UNUSABLE_FINISH_REASONS,
    build_provider_clients,
    stable_hash,
)
from paper_run_tracking import (  # noqa: E402
    freeze_run_inputs,
    record_cost_checkpoint,
)
from robustness.medical_qa import (  # noqa: E402
    CHOICE_LABELS,
    load_records,
    normalize_example,
    normalize_text_value,
)


PAPERS_ROOT = CODEX_ROOT / "papers"
DEFAULT_SYSTEM_PROMPT = (
    "As a healthcare professional, provide an expert response to each question. "
    "Exactly one answer option is the most correct."
)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(errors="replace") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def resolve_path(paper_root: Path, configured_path: str) -> Path:
    if configured_path.startswith("repo://"):
        return CODEX_ROOT / configured_path.removeprefix("repo://")
    path = Path(configured_path)
    return path if path.is_absolute() else paper_root / path


def extract_reasoning(raw: dict[str, Any]) -> str:
    for key in [
        "reasoning",
        "explanation",
        "exp",
        "long_answer",
        "LONG_ANSWER",
        "rationale",
    ]:
        if raw.get(key) not in [None, ""]:
            return normalize_text_value(raw[key])
    return ""


def load_normalized(path: Path, dataset: str) -> list[dict[str, Any]]:
    examples = []
    for index, raw in enumerate(load_records(path)):
        if not isinstance(raw, dict):
            continue
        example = normalize_example(raw, dataset, index)
        if example:
            example["reasoning"] = extract_reasoning(raw)
            examples.append(example)
    return examples


def require_numpy() -> Any:
    """Import NumPy only for legacy RNG compatibility.

    The framework's experiment requirements include NumPy, but keeping this
    import lazy lets provider and prompt unit tests run in minimal Python
    environments.
    """
    try:
        import numpy as np  # type: ignore[import-not-found]
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "selection_method=legacy_numpy reproduces the authors' RNG and "
            "requires NumPy. Install codex_version/requirements_experiment.txt."
        ) from exc
    return np


def percentile(values: list[int], percent: float) -> float:
    """Match NumPy's default linear percentile without a hard dependency."""
    if not values:
        raise ValueError("Cannot compute a percentile of an empty list.")
    ordered = sorted(values)
    position = (len(ordered) - 1) * percent / 100
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def filter_shots(
    shots: list[dict[str, Any]], dataset_cfg: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    configured = dataset_cfg.get("shot_filter_percentiles")
    if not configured:
        return shots, None
    if len(configured) != 2:
        raise ValueError("shot_filter_percentiles must contain [low, high].")
    low_percent, high_percent = (float(value) for value in configured)
    lengths = [len(shot["reasoning"].split()) for shot in shots]
    low = int(percentile(lengths, low_percent))
    high = int(percentile(lengths, high_percent))
    filtered = [
        shot for shot in shots if low <= len(shot["reasoning"].split()) <= high
    ]
    return filtered, {
        "percentiles": [low_percent, high_percent],
        "word_length_bounds": [low, high],
        "before": len(shots),
        "after": len(filtered),
    }


def select_examples(
    examples: list[dict[str, Any]], dataset_cfg: dict[str, Any]
) -> list[dict[str, Any]]:
    sample_size = dataset_cfg.get("sample_size")
    if sample_size not in [None, "all", "full"]:
        sample_size = int(sample_size)
        if sample_size > len(examples):
            raise RuntimeError(
                f"Requested {sample_size} examples from {dataset_cfg['name']}, "
                f"but only {len(examples)} normalized examples are available."
            )
        seed = int(dataset_cfg.get("selection_seed", 0))
        method = dataset_cfg.get("selection_method", "legacy_numpy")
        indices = list(range(len(examples)))
        if method == "legacy_numpy":
            rng = require_numpy().random.RandomState(seed)
            rng.shuffle(indices)
        elif method == "python_random":
            random.Random(seed).shuffle(indices)
        elif method != "head":
            raise ValueError(f"Unknown selection_method: {method}")
        examples = [examples[index] for index in indices[:sample_size]]

    max_examples = dataset_cfg.get("max_examples")
    if max_examples not in [None, "all", "full"]:
        max_examples = int(max_examples)
        method = dataset_cfg.get("max_examples_method", "head")
        if method == "head":
            examples = examples[:max_examples]
        elif method == "stratified_answer":
            if max_examples > len(examples):
                raise RuntimeError(
                    f"Requested {max_examples} stratified examples from "
                    f"{dataset_cfg['name']}, but only {len(examples)} "
                    "normalized examples are available."
                )
            seed = int(dataset_cfg.get("max_examples_seed", 0))
            rng = random.Random(seed)
            buckets: dict[int, list[tuple[int, dict[str, Any]]]] = {}
            for position, example in enumerate(examples):
                buckets.setdefault(int(example["answer_index"]), []).append(
                    (position, example)
                )
            labels = sorted(buckets)
            if not labels:
                return []
            base, remainder = divmod(max_examples, len(labels))
            selected: list[tuple[int, dict[str, Any]]] = []
            for label_position, label in enumerate(labels):
                bucket = buckets[label][:]
                rng.shuffle(bucket)
                quota = base + (1 if label_position < remainder else 0)
                if len(bucket) < quota:
                    raise RuntimeError(
                        f"Dataset {dataset_cfg['name']} has only {len(bucket)} "
                        f"examples for answer_index={label}; {quota} are "
                        "required for stratified max_examples."
                    )
                selected.extend(bucket[:quota])
            # Preserve evaluation order after sampling. This makes prompt order
            # stable while preventing label-grouped request batches.
            examples = [example for _, example in sorted(selected)]
        else:
            raise ValueError(f"Unknown max_examples_method: {method}")
    return examples


def load_dataset_bundle(
    paper_root: Path, dataset_cfg: dict[str, Any], n_shots: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    eval_path = resolve_path(paper_root, dataset_cfg["path"])
    if not eval_path.exists():
        raise RuntimeError(f"Dataset path does not exist: {eval_path}")
    eval_examples = select_examples(
        load_normalized(eval_path, dataset_cfg["name"]), dataset_cfg
    )
    if not eval_examples:
        raise RuntimeError(f"No usable examples found in {eval_path}")

    shots: list[dict[str, Any]] = []
    if n_shots:
        shots_path_raw = dataset_cfg.get("shots_path")
        if not shots_path_raw:
            raise RuntimeError(
                f"Dataset {dataset_cfg['name']} needs shots_path for the "
                f"configured {n_shots}-shot condition."
            )
        shots_path = resolve_path(paper_root, shots_path_raw)
        if not shots_path.exists():
            raise RuntimeError(f"Shots path does not exist: {shots_path}")
        shots = load_normalized(shots_path, dataset_cfg["name"])
        # The legacy implementation calculated length percentiles before
        # filtering empty rationales, so empty strings must remain in the
        # percentile population for exact MedMCQA compatibility.
        shots, shot_filter = filter_shots(shots, dataset_cfg)
        shots = [shot for shot in shots if shot.get("reasoning")]
        if len(shots) < n_shots:
            raise RuntimeError(
                f"Dataset {dataset_cfg['name']} has only {len(shots)} examples "
                f"with explanations; {n_shots} are required."
            )

    fingerprint = {
        "dataset": dataset_cfg["name"],
        "path": str(eval_path),
        "n": len(eval_examples),
        "selected_examples_sha256": stable_hash(
            [
                {
                    "id": example["id"],
                    "answer_index": example["answer_index"],
                    "choices": example["choices"],
                }
                for example in eval_examples
            ]
        ),
        "selection_method": dataset_cfg.get("selection_method", "legacy_numpy"),
        "selection_seed": dataset_cfg.get("selection_seed", 0),
        "sample_size": dataset_cfg.get("sample_size"),
        "max_examples": dataset_cfg.get("max_examples"),
        "max_examples_method": dataset_cfg.get("max_examples_method", "head"),
        "max_examples_seed": dataset_cfg.get("max_examples_seed", 0),
        "selected_answer_counts": dict(
            sorted(Counter(example["answer_index"] for example in eval_examples).items())
        ),
        "shots_path": str(shots_path) if n_shots else None,
        "shot_pool_n": len(shots),
        "shot_filter": shot_filter if n_shots else None,
    }
    return eval_examples, shots, fingerprint


def format_question(example: dict[str, Any]) -> str:
    options = " ".join(
        f"({CHOICE_LABELS[index]}) {choice}"
        for index, choice in enumerate(example["choices"])
    )
    return f"Question: {example['question']}\nChoices: {options}."


def format_shot_answer(example: dict[str, Any]) -> str:
    label = CHOICE_LABELS[example["answer_index"]]
    return (
        f"Explanation: {example['reasoning']}\n"
        f"Answer: Therefore, the answer is ({label}) {example['answer_text']}."
    )


def select_shots(
    shots: list[dict[str, Any]],
    *,
    n_shots: int,
    example_position: int,
    dataset_cfg: dict[str, Any],
) -> list[dict[str, Any]]:
    if not n_shots:
        return []
    mode = dataset_cfg.get("shot_selection", "per_example")
    if mode == "fixed":
        seed = int(dataset_cfg.get("shot_seed", 0))
        indices = list(range(len(shots)))
        require_numpy().random.RandomState(seed).shuffle(indices)
        chosen = indices[:n_shots]
    elif mode == "per_example":
        # Mirrors the legacy repository: each evaluation row deterministically
        # selects five training explanations with RandomState(row_index).
        chosen = require_numpy().random.RandomState(example_position).choice(
            len(shots), size=n_shots, replace=False
        )
    elif mode == "head":
        chosen = list(range(n_shots))
    else:
        raise ValueError(f"Unknown shot_selection: {mode}")
    return [shots[int(index)] for index in chosen]


def build_messages(
    example: dict[str, Any],
    shots: list[dict[str, Any]],
    *,
    system_prompt: str,
    answer_instruction: str = (
        "Think through the question step by step. End your response "
        "with exactly `Answer: (X)`, replacing X with the option letter."
    ),
) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    for shot in shots:
        messages.append({"role": "user", "content": format_question(shot)})
        messages.append({"role": "assistant", "content": format_shot_answer(shot)})
    messages.append(
        {
            "role": "user",
            "content": (
                f"{format_question(example)}\n"
                f"{answer_instruction}"
            ),
        }
    )
    return messages


def parse_answer(output: str, choices: list[str]) -> str | None:
    labels = CHOICE_LABELS[: len(choices)]
    patterns = [
        rf"(?i)(?:final\s+)?answer\s*(?::|is)?\s*\(?([{labels}])\)?",
        rf"(?i)therefore[^\n]{{0,120}}?answer[^\n]{{0,40}}?\(?([{labels}])\)?",
    ]
    for pattern in patterns:
        matches = re.findall(pattern, output)
        if matches:
            return matches[-1].upper()
    return None


def majority_vote(labels: list[str | None]) -> tuple[str | None, float, int]:
    valid = [label for label in labels if label]
    if not valid:
        return None, 0.0, 0
    counts = Counter(valid)
    max_count = max(counts.values())
    # Stable label-order tie break makes re-runs comparable.
    winner = sorted(label for label, count in counts.items() if count == max_count)[0]
    entropy = 0.0
    for count in counts.values():
        probability = count / len(valid)
        entropy -= probability * math.log2(probability)
    return winner, entropy, len(counts)


def prediction_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row.get("provider"),
        row.get("model"),
        row.get("item_id"),
        row.get("sample_index"),
    )


def latest_prediction_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep the most recently recorded attempt for each logical prediction.

    Sharded paper runs can be resumed in a different directory (for example,
    after a transient transport failure).  Filesystem traversal order is not a
    valid definition of "latest" in that case, so choose explicitly by the
    UTC timestamp written into every prediction row.  The input position is a
    deterministic fallback for legacy rows with a missing timestamp.
    """
    latest: dict[tuple[Any, ...], tuple[int, str, int, dict[str, Any]]] = {}
    for index, row in enumerate(rows):
        key = prediction_key(row)
        # A late retry that is itself truncated/error must never replace an
        # already usable answer from an earlier shard.  Among attempts with
        # equal usability, retain the chronologically latest record.
        usable = int(
            not row.get("api_error")
            and not row.get("truncated", False)
            and row.get("finish_reason") not in UNUSABLE_FINISH_REASONS
        )
        candidate = (usable, str(row.get("created_at", "")), index, row)
        previous = latest.get(key)
        if previous is None or candidate[:3] >= previous[:3]:
            latest[key] = candidate
    return [entry[3] for entry in latest.values()]


def summarize(
    rows: list[dict[str, Any]],
    items: list[dict[str, Any]],
    models: list[dict[str, Any]],
    sample_counts: list[int],
) -> list[dict[str, Any]]:
    item_lookup = {item["item_id"]: item for item in items}
    latest = [
        row
        for row in latest_prediction_rows(rows)
        if not row.get("api_error")
        and row.get("finish_reason") not in UNUSABLE_FINISH_REASONS
    ]
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in latest:
        grouped.setdefault((row["model_label"], row["item_id"]), []).append(row)

    metric_rows = []
    for model in models:
        label = model.get("label") or model["model"]
        for dataset in sorted({item["dataset"] for item in items}):
            dataset_items = [item for item in items if item["dataset"] == dataset]
            dataset_has_predictions = any(
                grouped.get((label, item["item_id"])) for item in dataset_items
            )
            for k in sample_counts:
                if not dataset_has_predictions:
                    metric_rows.append(
                        {
                            "model_label": label,
                            "provider": model["provider"],
                            "dataset": dataset,
                            "k": k,
                            "n": len(dataset_items),
                            "accuracy": None,
                            "coverage": None,
                            "mean_vote_entropy": None,
                            "mean_unique_answers": None,
                        }
                    )
                    continue
                correct = 0
                covered = 0
                entropies = []
                diversities = []
                for item in dataset_items:
                    samples = sorted(
                        grouped.get((label, item["item_id"]), []),
                        key=lambda row: row["sample_index"],
                    )
                    selected = [row.get("parsed_answer") for row in samples[:k]]
                    winner, entropy, diversity = majority_vote(selected)
                    if winner is not None:
                        covered += 1
                    if winner == item_lookup[item["item_id"]]["expected_letter"]:
                        correct += 1
                    entropies.append(entropy)
                    diversities.append(diversity)
                n = len(dataset_items)
                metric_rows.append(
                    {
                        "model_label": label,
                        "provider": model["provider"],
                        "dataset": dataset,
                        "k": k,
                        "n": n,
                        "accuracy": correct / n if n else None,
                        "coverage": covered / n if n else None,
                        "mean_vote_entropy": sum(entropies) / n if n else None,
                        "mean_unique_answers": sum(diversities) / n if n else None,
                    }
                )
    return metric_rows


def usage_tokens(usage: dict[str, Any]) -> tuple[int, int]:
    input_tokens = int(usage.get("prompt_tokens", usage.get("input_tokens", 0)) or 0)
    output_tokens = int(
        usage.get("completion_tokens", usage.get("output_tokens", 0)) or 0
    )
    return input_tokens, output_tokens


def row_cost(row: dict[str, Any], model: dict[str, Any]) -> float:
    usage = row.get("usage", {})
    # OpenRouter reports the charge applied after provider routing. Prefer it
    # over a catalog-price estimate because routed provider rates can differ
    # substantially from the model's headline price. Foundry responses do not
    # currently include this field, so they retain the configured proxy.
    reported_cost = usage.get("cost")
    if reported_cost is not None:
        return float(reported_cost)
    input_tokens, output_tokens = usage_tokens(usage)
    return (
        input_tokens * float(model.get("input_price_per_million", 0))
        + output_tokens * float(model.get("output_price_per_million", 0))
    ) / 1_000_000


def estimate_plan_cost(
    n_items: int, max_k: int, models: list[dict[str, Any]], config: dict[str, Any]
) -> tuple[float, list[dict[str, Any]]]:
    assumed_input = int(config.get("estimated_input_tokens", 2000))
    assumed_output = int(config.get("estimated_output_tokens", 250))
    breakdown = []
    total = 0.0
    for model in models:
        cost = n_items * max_k * (
            assumed_input * float(model.get("input_price_per_million", 0))
            + assumed_output * float(model.get("output_price_per_million", 0))
        ) / 1_000_000
        breakdown.append(
            {
                "model_label": model.get("label") or model["model"],
                "provider": model["provider"],
                "planned_cost_usd": cost,
            }
        )
        total += cost
    return total, breakdown


def write_metrics_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "model_label",
        "provider",
        "dataset",
        "k",
        "n",
        "accuracy",
        "coverage",
        "mean_vote_entropy",
        "mean_unique_answers",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field) for field in fields} for row in rows)


def write_summary(path: Path, manifest: dict[str, Any], metrics: list[dict[str, Any]]) -> None:
    def money(value: float) -> str:
        precision = 4 if 0 < value < 0.01 else 2
        return f"${value:.{precision}f}"

    lines = [
        "# Medical Reasoning Replication",
        "",
        f"Generated: `{manifest['generated_at']}`",
        f"Mode: `{'api' if manifest['execute_api'] else 'plan'}`",
        f"Condition: `{manifest['condition']}`",
        f"Questions: {manifest['planned_items']}",
        f"Maximum self-consistency k: {manifest['max_k']}",
        f"Planned requests: {manifest['planned_requests']}",
        f"Estimated maximum cost: {money(manifest['estimated_cost_usd'])}",
        f"Observed cost from reported token usage: {money(manifest['observed_cost_usd'])}",
        "",
        "## Metrics",
        "",
        "| Model | Provider | Dataset | k | N | Accuracy | Coverage | Vote entropy | Unique answers |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in metrics:
        def fmt(value: Any) -> str:
            return "NA" if value is None else f"{value:.3f}"

        lines.append(
            f"| {row['model_label']} | {row['provider']} | {row['dataset']} | "
            f"{row['k']} | {row['n']} | {fmt(row['accuracy'])} | "
            f"{fmt(row['coverage'])} | {fmt(row['mean_vote_entropy'])} | "
            f"{fmt(row['mean_unique_answers'])} |"
        )
    path.write_text("\n".join(lines).rstrip() + "\n")


def checkpoint(
    *,
    out_dir: Path,
    config: dict[str, Any],
    execute_api: bool,
    items: list[dict[str, Any]],
    models: list[dict[str, Any]],
    providers: dict[str, Any],
    sample_counts: list[int],
    rows: list[dict[str, Any]],
    fingerprints: list[dict[str, Any]],
    estimated_cost: float,
    cost_breakdown: list[dict[str, Any]],
    stopped_for_cost: bool = False,
) -> dict[str, Any]:
    metrics = summarize(rows, items, models, sample_counts)
    write_json(out_dir / "metrics.json", metrics)
    write_metrics_csv(out_dir / "metrics.csv", metrics)

    model_lookup = {
        (model["provider"], model["model"]): model for model in models
    }
    observed_cost = sum(
        row_cost(row, model_lookup[(row["provider"], row["model"])])
        for row in latest_prediction_rows(rows)
        if not row.get("api_error")
        and (row.get("provider"), row.get("model")) in model_lookup
    )
    successful = sum(
        1
        for row in latest_prediction_rows(rows)
        if not row.get("api_error")
        and row.get("finish_reason") not in UNUSABLE_FINISH_REASONS
    )
    manifest = {
        "generated_at": now_iso(),
        "execute_api": execute_api,
        "condition": config.get("condition", "five_shot_cot_local_extract"),
        "config_sha256": stable_hash(config),
        "sample_counts": sample_counts,
        "max_k": max(sample_counts),
        "planned_items": len(items),
        "planned_requests": len(items) * len(models) * max(sample_counts),
        "successful_requests": successful,
        "estimated_cost_usd": estimated_cost,
        "estimated_cost_by_model": cost_breakdown,
        "observed_cost_usd": observed_cost,
        "stopped_for_cost": stopped_for_cost,
        "datasets": fingerprints,
        "models": [
            {
                key: model.get(key)
                for key in ["label", "provider", "model", "temperature", "max_tokens"]
            }
            for model in models
        ],
        "providers": {
            name: client.provider.redacted() for name, client in providers.items()
        },
        "outputs": {
            "planned_items": str(out_dir / "planned_items.jsonl"),
            "predictions": str(out_dir / "predictions.jsonl"),
            "metrics": str(out_dir / "metrics.json"),
            "summary": str(out_dir / "summary.md"),
            "config_snapshot": str(out_dir / "config.snapshot.json"),
            "frozen_run_inputs": str(out_dir / "frozen_run_inputs.json"),
            "cost_checkpoints_jsonl": str(out_dir / "cost_checkpoints.jsonl"),
            "cost_checkpoints_csv": str(out_dir / "cost_checkpoints.csv"),
        },
    }
    write_json(out_dir / "run_manifest.json", manifest)
    write_summary(out_dir / "summary.md", manifest, metrics)
    return manifest


def run(
    config: dict[str, Any],
    *,
    paper_root: Path,
    out_dir: Path,
    execute_api: bool,
    max_new_requests: int | None = None,
    resume_from: Path | None = None,
    retry_unparsed: bool = False,
) -> dict[str, Any]:
    n_shots = int(config.get("n_shots", 5))
    sample_counts = sorted({int(value) for value in config.get("sample_counts", [1, 5, 10])})
    if not sample_counts or sample_counts[0] < 1:
        raise ValueError("sample_counts must contain positive integers")
    max_k = max(sample_counts)

    enabled_models = [
        model for model in config.get("models", []) if model.get("enabled", True)
    ]
    if not enabled_models:
        raise RuntimeError("No enabled models are configured.")
    if max_new_requests is not None and max_new_requests < 1:
        raise ValueError("max_new_requests must be positive")
    model_lookup = {
        (model["provider"], model["model"]): model for model in enabled_models
    }

    dataset_state = {}
    items = []
    fingerprints = []
    system_prompt = str(config.get("system_prompt", DEFAULT_SYSTEM_PROMPT))
    answer_instruction = str(
        config.get(
            "answer_instruction",
            "Think through the question step by step. End your response "
            "with exactly `Answer: (X)`, replacing X with the option letter.",
        )
    )
    for dataset_cfg in config.get("datasets", []):
        examples, shots, fingerprint = load_dataset_bundle(
            paper_root, dataset_cfg, n_shots
        )
        fingerprints.append(fingerprint)
        dataset_state[dataset_cfg["name"]] = (dataset_cfg, shots)
        for position, example in enumerate(examples):
            chosen_shots = select_shots(
                shots,
                n_shots=n_shots,
                example_position=position,
                dataset_cfg=dataset_cfg,
            )
            messages = build_messages(
                example,
                chosen_shots,
                system_prompt=system_prompt,
                answer_instruction=answer_instruction,
            )
            items.append(
                {
                    **example,
                    "item_id": stable_hash(
                        {
                            "dataset": dataset_cfg["name"],
                            "example_id": example["id"],
                            "condition": config.get(
                                "condition", "five_shot_cot_local_extract"
                            ),
                        }
                    )[:20],
                    "position": position,
                    "shot_ids": [shot["id"] for shot in chosen_shots],
                    "messages": messages,
                    "messages_sha256": stable_hash(messages),
                    "expected_letter": CHOICE_LABELS[example["answer_index"]],
                }
            )

    item_shard_count = int(config.get("item_shard_count", 1))
    item_shard_index = int(config.get("item_shard_index", 0))
    if item_shard_count < 1:
        raise ValueError("item_shard_count must be at least 1")
    if not 0 <= item_shard_index < item_shard_count:
        raise ValueError(
            "item_shard_index must be between 0 and item_shard_count - 1"
        )
    if item_shard_count > 1:
        # Filter only after full-dataset positions and demonstrations have been
        # assigned. This preserves exactly the same prompt that an item receives
        # in an unsharded run while allowing safe concurrent execution.
        items = [
            item
            for item in items
            if int(item["position"]) % item_shard_count == item_shard_index
        ]

    planned_requests = len(items) * len(enabled_models) * max_k
    max_requests = int(config.get("max_requests", 200))
    if planned_requests > max_requests:
        raise RuntimeError(
            f"Plan requires {planned_requests} requests, exceeding "
            f"max_requests={max_requests}. Increase the explicit cap only after "
            "reviewing the plan and cost estimate."
        )

    estimated_cost, cost_breakdown = estimate_plan_cost(
        len(items), max_k, enabled_models, config
    )
    max_cost = float(config.get("max_cost_usd", 10))
    if execute_api and estimated_cost > max_cost:
        raise RuntimeError(
            f"Estimated cost ${estimated_cost:.2f} exceeds "
            f"max_cost_usd=${max_cost:.2f}."
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    planned_path = out_dir / "planned_items.jsonl"
    planned_path.write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in items)
    )
    freeze_run_inputs(
        out_dir=out_dir,
        config=config,
        planned_items_path=planned_path,
        source_paths=[
            Path(__file__),
            CODEX_ROOT / "llm_api.py",
            CODEX_ROOT / "paper_run_tracking.py",
        ],
        study_stage="main_replication",
    )
    providers = build_provider_clients(
        config.get("providers", {}), out_dir / "api_cache"
    )
    for model in enabled_models:
        if model["provider"] not in providers:
            raise RuntimeError(
                f"Model {model['model']} references unknown provider "
                f"{model['provider']}."
            )

    predictions_path = out_dir / "predictions.jsonl"
    rows = read_jsonl(predictions_path)
    # A higher-cap retry is intentionally written to a fresh directory so the
    # initial frozen configuration remains immutable.  Reference predictions
    # provide completion state only; the retry directory records only the new
    # attempts and is later consolidated with the original shard by timestamp.
    reference_rows: list[dict[str, Any]] = []
    if resume_from is not None:
        if resume_from.is_dir():
            for source in sorted(resume_from.rglob("predictions.jsonl")):
                reference_rows.extend(read_jsonl(source))
        else:
            reference_rows.extend(read_jsonl(resume_from))
    rows_for_completion = [*reference_rows, *rows]
    completed = {
        prediction_key(row)
        for row in latest_prediction_rows(rows_for_completion)
        if not row.get("api_error")
        and not row.get("truncated", False)
        and row.get("finish_reason") not in UNUSABLE_FINISH_REASONS
        and (not retry_unparsed or row.get("parsed_answer") is not None)
    }

    stopped_for_cost = False
    if execute_api:
        progress_every = int(config.get("progress_every", 25))
        max_consecutive_errors = int(config.get("max_consecutive_errors", 10))
        consecutive_errors = 0
        new_requests = 0
        stop_requested = False
        for model in enabled_models:
            client = providers[model["provider"]]
            for item in items:
                dataset_cfg, shots = dataset_state[item["dataset"]]
                shot_lookup = {shot["id"]: shot for shot in shots}
                chosen_shots = [shot_lookup[shot_id] for shot_id in item["shot_ids"]]
                messages = build_messages(
                    item,
                    chosen_shots,
                    system_prompt=system_prompt,
                    answer_instruction=answer_instruction,
                )
                for sample_index in range(max_k):
                    if max_new_requests is not None and new_requests >= max_new_requests:
                        stop_requested = True
                        break
                    key = (
                        model["provider"],
                        model["model"],
                        item["item_id"],
                        sample_index,
                    )
                    if key in completed:
                        continue
                    current_cost = sum(
                        row_cost(row, model_lookup[(row["provider"], row["model"])])
                        for row in rows
                        if not row.get("api_error")
                        and (row.get("provider"), row.get("model"))
                        in model_lookup
                    )
                    if current_cost >= max_cost:
                        stopped_for_cost = True
                        stop_requested = True
                        break
                    try:
                        temperature_raw = model.get("temperature", 0.5)
                        output, meta = client.complete(
                            model=model["model"],
                            messages=messages,
                            temperature=(
                                None
                                if temperature_raw is None
                                else float(temperature_raw)
                            ),
                            max_tokens=int(model.get("max_tokens", 512)),
                            max_tokens_field=str(
                                model.get("max_tokens_field", "max_tokens")
                            ),
                            seed=(
                                int(model.get("seed", 0)) + sample_index
                                if model.get("send_seed", False)
                                else None
                            ),
                            extra_body=model.get("extra_body"),
                        )
                        finish_reason = meta.get("finish_reason")
                        truncated = finish_reason in {"length", "max_tokens"}
                        parsed = (
                            None
                            if truncated
                            else parse_answer(output, item["choices"])
                        )
                        api_error = None
                        consecutive_errors = 0
                    except Exception as exc:  # noqa: BLE001
                        output = ""
                        meta = {"usage": {}, "cache_hit": False}
                        parsed = None
                        finish_reason = None
                        truncated = False
                        api_error = str(exc)
                        consecutive_errors += 1
                    new_requests += 1
                    row = {
                        "created_at": now_iso(),
                        "provider": model["provider"],
                        "model_label": model.get("label") or model["model"],
                        "model": model["model"],
                        "dataset": item["dataset"],
                        "example_id": item["id"],
                        "item_id": item["item_id"],
                        "sample_index": sample_index,
                        "output": output,
                        "reasoning": meta.get("reasoning", ""),
                        "reasoning_details": meta.get("reasoning_details"),
                        "assistant_message": meta.get("assistant_message"),
                        "parsed_answer": parsed,
                        "expected_letter": item["expected_letter"],
                        "correct": parsed == item["expected_letter"],
                        "usage": meta.get("usage", {}),
                        "cache_hit": meta.get("cache_hit", False),
                        "finish_reason": finish_reason,
                        "response_id": meta.get("response_id"),
                        "response_model": meta.get("response_model"),
                        "truncated": truncated,
                        "api_error": api_error,
                    }
                    rows.append(row)
                    append_jsonl(predictions_path, row)
                    success_added = False
                    if (
                        api_error is None
                        and not truncated
                        and finish_reason not in UNUSABLE_FINISH_REASONS
                    ):
                        completed.add(key)
                        success_added = True
                    if (
                        success_added
                        and len(completed) > 0
                        and len(completed) % progress_every == 0
                    ):
                        manifest = checkpoint(
                            out_dir=out_dir,
                            config=config,
                            execute_api=execute_api,
                            items=items,
                            models=enabled_models,
                            providers=providers,
                            sample_counts=sample_counts,
                            rows=rows,
                            fingerprints=fingerprints,
                            estimated_cost=estimated_cost,
                            cost_breakdown=cost_breakdown,
                        )
                        record_cost_checkpoint(
                            out_dir=out_dir,
                            study_stage="main_replication",
                            planned_rows=planned_requests,
                            raw_rows=rows,
                            canonical_rows=latest_prediction_rows(rows),
                            cost_fn=lambda prediction: row_cost(
                                prediction,
                                model_lookup[(prediction["provider"], prediction["model"])],
                            ),
                            interval=progress_every,
                        )
                        print(
                            f"[medical-replication] progress successful="
                            f"{len(completed)}/{planned_requests} cost="
                            f"${manifest['observed_cost_usd']:.4f} errors="
                            f"{consecutive_errors}",
                            flush=True,
                        )
                    if consecutive_errors >= max_consecutive_errors:
                        raise RuntimeError(
                            f"Stopped after {consecutive_errors} consecutive API errors."
                        )
                if stop_requested:
                    break
            if stop_requested:
                break

    manifest = checkpoint(
        out_dir=out_dir,
        config=config,
        execute_api=execute_api,
        items=items,
        models=enabled_models,
        providers=providers,
        sample_counts=sample_counts,
        rows=rows,
        fingerprints=fingerprints,
        estimated_cost=estimated_cost,
        cost_breakdown=cost_breakdown,
        stopped_for_cost=stopped_for_cost,
    )
    if execute_api:
        record_cost_checkpoint(
            out_dir=out_dir,
            study_stage="main_replication",
            planned_rows=planned_requests,
            raw_rows=rows,
            canonical_rows=latest_prediction_rows(rows),
            cost_fn=lambda prediction: row_cost(
                prediction,
                model_lookup[(prediction["provider"], prediction["model"])],
            ),
            interval=int(config.get("progress_every", 25)),
            force_final=True,
        )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper-id", default="cellpress-medical-reasoning")
    parser.add_argument("--paper-root", type=Path)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--execute-api", action="store_true")
    parser.add_argument(
        "--sample-counts",
        type=int,
        nargs="+",
        help="Override configured nested k values, for example: --sample-counts 1 5.",
    )
    parser.add_argument("--max-requests", type=int)
    parser.add_argument("--max-cost-usd", type=float)
    parser.add_argument(
        "--max-new-requests",
        type=int,
        help="Execute at most this many currently pending calls, then checkpoint.",
    )
    parser.add_argument(
        "--resume-from",
        type=Path,
        help=(
            "Read completed prediction rows from this file or directory while "
            "writing any pending retries to --out-dir."
        ),
    )
    parser.add_argument(
        "--retry-unparsed",
        action="store_true",
        help=(
            "When resuming, treat otherwise usable records without a parsed "
            "option letter as pending recovery calls."
        ),
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        help="Override the response-token cap for every selected model.",
    )
    parser.add_argument(
        "--model",
        action="append",
        dest="models",
        help=(
            "Run only this exact model ID. Repeat the option to select multiple "
            "models. This is useful for resumable model/dataset shards."
        ),
    )
    parser.add_argument(
        "--dataset",
        action="append",
        dest="datasets",
        help=(
            "Run only this exact dataset name. Repeat the option to select "
            "multiple datasets."
        ),
    )
    parser.add_argument(
        "--item-shard-count",
        type=int,
        help="Partition fully constructed items into this many modulo shards.",
    )
    parser.add_argument(
        "--item-shard-index",
        type=int,
        help="Zero-based modulo shard index selected by --item-shard-count.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paper_root = args.paper_root or PAPERS_ROOT / args.paper_id
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_dir or paper_root / "runs" / f"main_replication_{stamp}"
    config = read_json(args.config)
    if args.models:
        selected = set(args.models)
        config["models"] = [
            model for model in config.get("models", [])
            if model.get("model") in selected
        ]
        missing = selected - {
            model.get("model") for model in config.get("models", [])
        }
        if missing:
            raise ValueError(f"Unknown model ID(s): {sorted(missing)}")
    if args.datasets:
        selected = set(args.datasets)
        config["datasets"] = [
            dataset for dataset in config.get("datasets", [])
            if dataset.get("name") in selected
        ]
        missing = selected - {
            dataset.get("name") for dataset in config.get("datasets", [])
        }
        if missing:
            raise ValueError(f"Unknown dataset name(s): {sorted(missing)}")
    if args.sample_counts:
        config["sample_counts"] = args.sample_counts
    if args.max_requests is not None:
        config["max_requests"] = args.max_requests
    if args.max_cost_usd is not None:
        config["max_cost_usd"] = args.max_cost_usd
    if args.max_tokens is not None:
        if args.max_tokens < 1:
            raise ValueError("max_tokens must be positive")
        for model in config.get("models", []):
            model["max_tokens"] = args.max_tokens
    if args.item_shard_count is not None:
        config["item_shard_count"] = args.item_shard_count
    if args.item_shard_index is not None:
        config["item_shard_index"] = args.item_shard_index
    manifest = run(
        config,
        paper_root=paper_root,
        out_dir=out_dir,
        execute_api=args.execute_api,
        max_new_requests=args.max_new_requests,
        resume_from=args.resume_from,
        retry_unparsed=args.retry_unparsed,
    )
    print(
        f"[medical-replication] mode={'api' if args.execute_api else 'plan'} "
        f"items={manifest['planned_items']} requests={manifest['planned_requests']} "
        f"estimated_cost=${manifest['estimated_cost_usd']:.4f}"
    )
    print(f"[medical-replication] summary={manifest['outputs']['summary']}")


if __name__ == "__main__":
    main()
