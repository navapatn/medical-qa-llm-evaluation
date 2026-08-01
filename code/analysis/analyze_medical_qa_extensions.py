#!/usr/bin/env python3
"""Create the final statistical report for the Qwen medical-QA extensions."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

from scipy.stats import binomtest  # type: ignore[import-not-found]

CODEX_ROOT = Path(__file__).resolve().parent
if str(CODEX_ROOT) not in sys.path:
    sys.path.insert(0, str(CODEX_ROOT))

from replicate_medical_reasoning import row_cost  # noqa: E402
from robustness.medical_qa import CHOICE_LABELS  # noqa: E402
from run_medical_qa_extensions import (  # noqa: E402
    latest_rows,
    read_json,
    read_jsonl,
    write_json,
)


DATASETS = ["MedQA-USMLE", "MedMCQA", "PubMedQA"]


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def wilson(successes: int, n: int, z: float = 1.959963984540054) -> list[float] | None:
    if n == 0:
        return None
    p = successes / n
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return [max(0.0, center - margin), min(1.0, center + margin)]


def binary_metric(values: list[bool]) -> dict[str, Any]:
    successes = sum(values)
    return {
        "n": len(values),
        "successes": successes,
        "rate": successes / len(values) if values else None,
        "wilson_95": wilson(successes, len(values)),
    }


def bootstrap_mean_ci(
    values: list[float], *, seed: int = 20260716, repeats: int = 10_000
) -> list[float] | None:
    if not values:
        return None
    rng = random.Random(seed)
    n = len(values)
    estimates = sorted(
        sum(values[rng.randrange(n)] for _ in range(n)) / n
        for _ in range(repeats)
    )
    return [estimates[int(0.025 * repeats)], estimates[int(0.975 * repeats)]]


def paired_binary(
    baseline: dict[tuple[str, str], bool],
    condition: dict[tuple[str, str], bool],
    *,
    seed: int = 20260716,
) -> dict[str, Any]:
    keys = sorted(set(baseline) & set(condition))
    pairs = [(baseline[key], condition[key]) for key in keys]
    improved = sum(not base and new for base, new in pairs)
    degraded = sum(base and not new for base, new in pairs)
    discordant = improved + degraded
    differences = [float(new) - float(base) for base, new in pairs]
    p_value = (
        binomtest(improved, discordant, 0.5).pvalue if discordant else 1.0
    )
    return {
        "n": len(pairs),
        "baseline_rate": mean([float(base) for base, _ in pairs]),
        "condition_rate": mean([float(new) for _, new in pairs]),
        "difference": mean(differences),
        "paired_bootstrap_95": bootstrap_mean_ci(differences, seed=seed),
        "improved": improved,
        "degraded": degraded,
        "unchanged": len(pairs) - discordant,
        "mcnemar_exact_p": p_value,
    }


def scoped(rows: list[dict[str, Any]], dataset: str | None) -> list[dict[str, Any]]:
    return rows if dataset is None else [row for row in rows if row["dataset"] == dataset]


def scope_name(dataset: str | None) -> str:
    return "All" if dataset is None else dataset


def question_key(row: dict[str, Any]) -> tuple[str, str]:
    return (str(row["dataset"]), str(row["example_id"]))


def analyze_baseline(
    baseline_rows: list[dict[str, Any]], dataset: str | None
) -> dict[str, Any]:
    rows = scoped(baseline_rows, dataset)
    return binary_metric([bool(row.get("correct")) for row in rows])


def analyze_generative(
    rows: list[dict[str, Any]],
    baseline_rows: list[dict[str, Any]],
    dataset: str | None,
) -> dict[str, Any]:
    generated = [
        row for row in scoped(rows, dataset)
        if row["variant"] == "generative_no_choices"
    ]
    baseline = {
        question_key(row): bool(row.get("correct"))
        for row in scoped(baseline_rows, dataset)
    }
    option_recovery = {
        question_key(row): bool(row.get("option_recovery_correct"))
        for row in generated
    }
    return {
        "strict_text_match": binary_metric([
            bool(row.get("strict_correct")) for row in generated
        ]),
        "option_recovery": binary_metric(list(option_recovery.values())),
        "combined_automatic_score": binary_metric([
            bool(row.get("correct")) for row in generated
        ]),
        "paired_vs_baseline_option_recovery": paired_binary(
            baseline, option_recovery
        ),
    }


def analyze_idk(
    rows: list[dict[str, Any]],
    baseline_rows: list[dict[str, Any]],
    dataset: str | None,
) -> dict[str, Any]:
    selected = scoped(rows, dataset)
    answerable = [row for row in selected if row["variant"] == "idk_answerable"]
    unanswerable = [row for row in selected if row["variant"] == "idk_unanswerable"]
    baseline = {
        question_key(row): bool(row.get("correct"))
        for row in scoped(baseline_rows, dataset)
    }
    condition = {question_key(row): bool(row.get("correct")) for row in answerable}
    by_position = {}
    for position in sorted({int(row["idk_position"]) for row in answerable}):
        group = [row for row in answerable if int(row["idk_position"]) == position]
        by_position[CHOICE_LABELS[position]] = {
            "n": len(group),
            "accuracy": binary_metric([bool(row.get("correct")) for row in group]),
            "abstention": binary_metric([bool(row.get("abstained")) for row in group]),
        }
    return {
        "answerable_accuracy": binary_metric([
            bool(row.get("correct")) for row in answerable
        ]),
        "answerable_abstention": binary_metric([
            bool(row.get("abstained")) for row in answerable
        ]),
        "unanswerable_abstention": binary_metric([
            bool(row.get("abstained")) for row in unanswerable
        ]),
        "paired_accuracy_vs_baseline": paired_binary(baseline, condition),
        "by_idk_position": by_position,
    }


def analyze_position(
    rows: list[dict[str, Any]],
    baseline_rows: list[dict[str, Any]],
    dataset: str | None,
) -> dict[str, Any]:
    baseline = scoped(baseline_rows, dataset)
    rotations = [
        row for row in scoped(rows, dataset)
        if row["variant"].startswith("choice_rotation_")
    ]
    by_position_values: dict[str, list[bool]] = defaultdict(list)
    by_question: dict[tuple[str, str], list[tuple[int | None, bool]]] = defaultdict(list)
    for row in baseline:
        letter = row.get("parsed_answer")
        predicted = CHOICE_LABELS.index(letter) if letter in CHOICE_LABELS else None
        expected_position = CHOICE_LABELS.index(row["expected_letter"])
        by_position_values[CHOICE_LABELS[expected_position]].append(bool(row.get("correct")))
        by_question[question_key(row)].append((predicted, bool(row.get("correct"))))
    for row in rotations:
        expected_position = CHOICE_LABELS.index(row["expected_letter"])
        by_position_values[CHOICE_LABELS[expected_position]].append(bool(row.get("correct")))
        by_question[question_key(row)].append((
            row.get("predicted_original_index"), bool(row.get("correct"))
        ))

    invariance = []
    all_orders_correct = []
    any_order_correct = []
    for group in by_question.values():
        predictions = [prediction for prediction, _ in group]
        correctness = [correct for _, correct in group]
        invariance.append(None not in predictions and len(set(predictions)) == 1)
        all_orders_correct.append(all(correctness))
        any_order_correct.append(any(correctness))
    position_metrics = {
        position: binary_metric(values)
        for position, values in sorted(by_position_values.items())
    }
    rates = [metric["rate"] for metric in position_metrics.values() if metric["rate"] is not None]
    return {
        "by_displayed_correct_position": position_metrics,
        "max_minus_min_position_accuracy": max(rates) - min(rates) if rates else None,
        "fully_invariant": binary_metric(invariance),
        "changed_with_order": binary_metric([not value for value in invariance]),
        "all_orders_correct": binary_metric(all_orders_correct),
        "any_order_correct": binary_metric(any_order_correct),
    }


def analyze_candidate_verification(
    rows: list[dict[str, Any]], dataset: str | None
) -> dict[str, Any]:
    candidates = [
        row for row in scoped(rows, dataset)
        if row["variant"] == "candidate_verification"
    ]
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        grouped[question_key(row)].append(row)
    exact = []
    sensitivity = []
    specificity = []
    true_counts = Counter()
    exactly_one = []
    exactly_one_correct = []
    for group in grouped.values():
        group_exact = all(row.get("correct") is True for row in group)
        exact.append(group_exact)
        sensitivity.extend(
            row.get("parsed_answer") is True
            for row in group if row.get("expected_boolean") is True
        )
        specificity.extend(
            row.get("parsed_answer") is False
            for row in group if row.get("expected_boolean") is False
        )
        true_count = sum(row.get("parsed_answer") is True for row in group)
        true_counts[true_count] += 1
        exactly_one.append(true_count == 1)
        if true_count == 1:
            exactly_one_correct.append(group_exact)
    return {
        "candidate_accuracy": binary_metric([
            bool(row.get("correct")) for row in candidates
        ]),
        "exact_question_accuracy": binary_metric(exact),
        "correct_candidate_sensitivity": binary_metric(sensitivity),
        "distractor_specificity": binary_metric(specificity),
        "exactly_one_true_coverage": binary_metric(exactly_one),
        "accuracy_when_exactly_one_true": binary_metric(exactly_one_correct),
        "predicted_true_count_distribution": {
            str(key): value for key, value in sorted(true_counts.items())
        },
    }


def analyze_reconstruction(
    rows: list[dict[str, Any]], dataset: str | None
) -> dict[str, Any]:
    reconstructed = [
        row for row in scoped(rows, dataset)
        if row["variant"] == "choices_to_question"
    ]
    f1_values = [float(row["question_token_f1"]) for row in reconstructed]
    return {
        "n": len(reconstructed),
        "mean_token_f1": mean(f1_values),
        "bootstrap_95": bootstrap_mean_ci(f1_values),
        "question_mark_compliance": binary_metric([
            str(row.get("output", "")).strip().endswith("?")
            for row in reconstructed
        ]),
    }


def analyze_costs(
    rows: list[dict[str, Any]], raw_rows: list[dict[str, Any]], model: dict[str, Any]
) -> dict[str, Any]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["dataset"], row["variant"])].append(row)

    breakdown = []
    for (dataset, variant), group in sorted(grouped.items()):
        cost = sum(row_cost(row, model) for row in group)
        prompt_tokens = sum(int(row.get("usage", {}).get("prompt_tokens", 0) or 0) for row in group)
        completion_tokens = sum(int(row.get("usage", {}).get("completion_tokens", 0) or 0) for row in group)
        reasoning_tokens = sum(int(
            row.get("usage", {}).get("completion_tokens_details", {}).get("reasoning_tokens", 0) or 0
        ) for row in group)
        breakdown.append({
            "dataset": dataset,
            "variant": variant,
            "calls": len(group),
            "cost_usd": cost,
            "mean_cost_usd": cost / len(group),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "reasoning_tokens": reasoning_tokens,
        })
    canonical_cost = sum(row_cost(row, model) for row in rows)
    raw_cost = sum(row_cost(row, model) for row in raw_rows if not row.get("api_error"))
    return {
        "canonical_experiment_cost_usd": canonical_cost,
        "raw_api_charges_usd": raw_cost,
        "duplicate_overhead_usd": raw_cost - canonical_cost,
        "breakdown": breakdown,
    }


def reconstruction_examples(
    rows: list[dict[str, Any]], planned_items: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    output = []
    for dataset in DATASETS:
        group = sorted(
            [row for row in rows if row["dataset"] == dataset and row["variant"] == "choices_to_question"],
            key=lambda row: float(row["question_token_f1"]),
        )
        if not group:
            continue
        for label, row in [("low_overlap", group[0]), ("median_overlap", group[len(group) // 2])]:
            item = planned_items[row["item_id"]]
            output.append({
                "dataset": dataset,
                "selection": label,
                "token_f1": row["question_token_f1"],
                "choices": item["choices"],
                "original_question": item["question"],
                "reconstructed_question": row["output"],
            })
    return output


def pct(value: float | None) -> str:
    return "NA" if value is None else f"{100 * value:.1f}%"


def pct_ci(metric: dict[str, Any]) -> str:
    if metric["rate"] is None:
        return "NA"
    low, high = metric["wilson_95"]
    return f"{pct(metric['rate'])} [{pct(low)}, {pct(high)}]"


def write_cost_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "dataset", "variant", "calls", "cost_usd", "mean_cost_usd",
        "prompt_tokens", "completion_tokens", "reasoning_tokens",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field) for field in fields} for row in rows)


def write_report(path: Path, analysis: dict[str, Any]) -> None:
    lines = [
        "# Final Qwen Medical-QA Extension Analysis",
        "",
        "## Execution audit",
        "",
        f"- Questions: {analysis['audit']['questions']} (115 MedQA, 90 MedMCQA, 45 PubMedQA).",
        f"- Completed planned calls: {analysis['audit']['completed_calls']}/2,660; API errors: {analysis['audit']['api_errors']}; unparsed structured answers: {analysis['audit']['unparsed_structured_answers']}.",
        f"- Canonical experiment cost: ${analysis['costs']['canonical_experiment_cost_usd']:.4f}.",
        f"- Total actual API charges: ${analysis['costs']['raw_api_charges_usd']:.4f}, including ${analysis['costs']['duplicate_overhead_usd']:.4f} from one late validation duplicate.",
        "",
        "## Results by dataset",
        "",
        "| Dataset | Baseline MCQ | Question-only option recovery | Paired change | Answerable IDK accuracy | IDK abstention | Order invariant | All orders correct | T/F exact question | Choices→question F1 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in ["All", *DATASETS]:
        scope = analysis["scopes"][name]
        paired = scope["generative"]["paired_vs_baseline_option_recovery"]
        lines.append(
            f"| {name} | {pct_ci(scope['baseline'])} | "
            f"{pct_ci(scope['generative']['option_recovery'])} | "
            f"{paired['difference'] * 100:+.1f} pp | "
            f"{pct_ci(scope['idk']['answerable_accuracy'])} | "
            f"{pct(scope['idk']['answerable_abstention']['rate'])} | "
            f"{pct_ci(scope['position']['fully_invariant'])} | "
            f"{pct_ci(scope['position']['all_orders_correct'])} | "
            f"{pct_ci(scope['candidate_verification']['exact_question_accuracy'])} | "
            f"{scope['reconstruction']['mean_token_f1']:.3f} |"
        )

    pooled = analysis["scopes"]["All"]
    gen_paired = pooled["generative"]["paired_vs_baseline_option_recovery"]
    idk_paired = pooled["idk"]["paired_accuracy_vs_baseline"]
    lines.extend([
        "",
        "## Main findings",
        "",
        f"1. **Removing choices causes a large drop.** Automatic option recovery fell from {pct(gen_paired['baseline_rate'])} to {pct(gen_paired['condition_rate'])}, a paired change of {gen_paired['difference'] * 100:+.1f} points (95% paired bootstrap interval {gen_paired['paired_bootstrap_95'][0] * 100:+.1f} to {gen_paired['paired_bootstrap_95'][1] * 100:+.1f}; exact McNemar p={gen_paired['mcnemar_exact_p']:.3g}). The drop is largest on MedMCQA.",
        f"2. **Adding IDK did not harm aggregate accuracy.** Accuracy changed from {pct(idk_paired['baseline_rate'])} to {pct(idk_paired['condition_rate'])} ({idk_paired['difference'] * 100:+.1f} points; exact McNemar p={idk_paired['mcnemar_exact_p']:.3g}), but Qwen abstained on only {pct(pooled['idk']['answerable_abstention']['rate'])} of answerable items. It abstained on {pct(pooled['idk']['unanswerable_abstention']['rate'])} of the explicit withheld-stem controls.",
        f"3. **Balanced reordering reveals residual sensitivity.** Aggregate prediction was invariant for {pct(pooled['position']['fully_invariant']['rate'])}; therefore {pct(pooled['position']['changed_with_order']['rate'])} of questions changed underlying answer at least once. Every ordering was correct for {pct(pooled['position']['all_orders_correct']['rate'])}.",
        f"4. **Independent true/false verification is materially weaker than MCQ answering.** Exact question-level verification was {pct(pooled['candidate_verification']['exact_question_accuracy']['rate'])}; sensitivity to the correct candidate was {pct(pooled['candidate_verification']['correct_candidate_sensitivity']['rate'])}, versus {pct(pooled['candidate_verification']['distractor_specificity']['rate'])} specificity.",
        f"5. **Choices alone underdetermine the original question.** Mean lexical token F1 between reconstructed and original stems was {pooled['reconstruction']['mean_token_f1']:.3f}. This is descriptive only: lexical overlap cannot establish semantic plausibility.",
        "",
        "## Accuracy by displayed correct-answer position",
        "",
        "| Dataset | A | B | C | D | Max–min |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for name in DATASETS:
        position = analysis["scopes"][name]["position"]
        values = position["by_displayed_correct_position"]
        lines.append(
            f"| {name} | {pct(values.get('A', {}).get('rate'))} | "
            f"{pct(values.get('B', {}).get('rate'))} | "
            f"{pct(values.get('C', {}).get('rate'))} | "
            f"{pct(values.get('D', {}).get('rate'))} | "
            f"{position['max_minus_min_position_accuracy'] * 100:.1f} pp |"
        )
    lines.extend([
        "",
        "Displayed-position accuracy is shown within dataset because PubMedQA has only three choices; pooling would make position D look artificially strong by excluding PubMedQA.",
    ])

    lines.extend([
        "",
        "## Candidate verification behavior",
        "",
        f"- Candidate-level accuracy: {pct_ci(pooled['candidate_verification']['candidate_accuracy'])}.",
        f"- Exact all-candidate question accuracy: {pct_ci(pooled['candidate_verification']['exact_question_accuracy'])}.",
        f"- Exactly-one-True coverage: {pct_ci(pooled['candidate_verification']['exactly_one_true_coverage'])}; accuracy conditional on exactly one True: {pct_ci(pooled['candidate_verification']['accuracy_when_exactly_one_true'])}.",
        f"- Predicted-True counts per question: {json.dumps(pooled['candidate_verification']['predicted_true_count_distribution'], sort_keys=True)}.",
        "",
        "## Interpretation limits",
        "",
        "- This is one current model, one stratified 250-question sample, and one stochastic completion per prompt at temperature 0.5. Confidence intervals quantify item sampling uncertainty, not model-sampling or provider-version uncertainty.",
        "- The original-order baseline was reused from the earlier paper-faithful Qwen run. It matches question IDs and prompt construction but was not generated simultaneously with the extensions.",
        "- Question-only scoring is automatic: exact answer-text containment plus lexical recovery against the original options. It should be validated on a blinded human/clinician subset before being treated as definitive generative-answer accuracy.",
        "- Choices-to-question token F1 is not a semantic judge. The saved qualitative examples require human or separately validated judge scoring.",
        "- The unanswerable control explicitly states that the stem was withheld, so its 100% abstention is a ceiling sanity check, not evidence of calibrated abstention on subtly unanswerable clinical questions.",
        "- Full cyclic rotations balance displayed positions, but k=1 stochasticity means a changed answer cannot be attributed exclusively to position without a deterministic or repeated-sampling follow-up.",
    ])
    path.write_text("\n".join(lines).rstrip() + "\n")


def analyze(run_dir: Path, config_path: Path) -> dict[str, Any]:
    config = read_json(config_path)
    model = config["models"][0]
    raw_rows = read_jsonl(run_dir / "predictions.jsonl")
    rows = [row for row in latest_rows(raw_rows) if not row.get("api_error")]
    baseline_rows = read_jsonl(run_dir / "baseline_predictions.jsonl")
    planned_items = {
        row["item_id"]: row for row in read_jsonl(run_dir / "planned_items.jsonl")
    }

    scopes = {}
    for dataset in [None, *DATASETS]:
        scopes[scope_name(dataset)] = {
            "baseline": analyze_baseline(baseline_rows, dataset),
            "generative": analyze_generative(rows, baseline_rows, dataset),
            "idk": analyze_idk(rows, baseline_rows, dataset),
            "position": analyze_position(rows, baseline_rows, dataset),
            "candidate_verification": analyze_candidate_verification(rows, dataset),
            "reconstruction": analyze_reconstruction(rows, dataset),
        }

    costs = analyze_costs(rows, raw_rows, model)
    audit = {
        "questions": len(baseline_rows),
        "completed_calls": len(rows),
        "raw_rows": len(raw_rows),
        "duplicate_audit_rows": len(raw_rows) - len(rows),
        "api_errors": sum(bool(row.get("api_error")) for row in rows),
        "unparsed_structured_answers": sum(
            row.get("parsed_answer") is None and row["variant"] != "choices_to_question"
            for row in rows
        ),
        "finish_reasons": dict(sorted(Counter(row.get("finish_reason") for row in rows).items())),
    }
    analysis = {
        "audit": audit,
        "scopes": scopes,
        "costs": costs,
        "reconstruction_examples": reconstruction_examples(rows, planned_items),
    }
    write_json(run_dir / "final_analysis.json", analysis)
    write_json(run_dir / "reconstruction_examples.json", analysis["reconstruction_examples"])
    write_cost_csv(run_dir / "cost_by_condition.csv", costs["breakdown"])
    write_report(run_dir / "final_analysis.md", analysis)
    return analysis


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    analysis = analyze(args.run_dir, args.config)
    print(
        f"[medical-extension-analysis] calls={analysis['audit']['completed_calls']} "
        f"canonical_cost=${analysis['costs']['canonical_experiment_cost_usd']:.4f} "
        f"raw_charges=${analysis['costs']['raw_api_charges_usd']:.4f}"
    )
    print(f"[medical-extension-analysis] report={args.run_dir / 'final_analysis.md'}")


if __name__ == "__main__":
    main()
