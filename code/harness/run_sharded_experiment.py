#!/usr/bin/env python3
"""Run a paper experiment as independent result-column shards and merge outputs."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from run_codex_condition import validate_result


CODEX_ROOT = Path(__file__).resolve().parent
PAPERS_ROOT = Path(os.environ.get("CODEX_PAPERS_ROOT", CODEX_ROOT / "papers")).resolve()

ARCHIVE_OUTPUT_IGNORE = shutil.ignore_patterns(
    ".cache",
    ".venv",
    ".venv*",
    "__pycache__",
    "cache",
    "dist-packages",
    "env",
    "envs",
    "node_modules",
    "site-packages",
    "venv",
    "venv*",
)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def run_cmd(cmd: list[str], cwd: Path, env: dict[str, str] | None = None) -> None:
    print("+ " + " ".join(shlex_quote(part) for part in cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd), check=True, env=env)


def shlex_quote(text: str) -> str:
    if not text:
        return "''"
    safe = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_+-=.,/:")
    if all(ch in safe for ch in text):
        return text
    return "'" + text.replace("'", "'\"'\"'") + "'"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))


def copy_if_exists(src: Path, dst: Path, ignore=None) -> None:
    if not src.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.is_dir():
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst, symlinks=True, ignore=ignore)
    else:
        shutil.copy2(src, dst)


def clone_paper_for_shard(src_root: Path, dst_root: Path) -> None:
    """Create an isolated paper workspace without duplicating large datasets."""
    if dst_root.exists():
        shutil.rmtree(dst_root)
    dst_root.mkdir(parents=True)
    symlink_names = {"source", "data"}
    skip_names = {"conditions", "sharded_runs"}
    for item in src_root.iterdir():
        if item.name in skip_names:
            continue
        target = dst_root / item.name
        if item.name in symlink_names and item.exists():
            target.symlink_to(item.resolve(), target_is_directory=item.is_dir())
        elif item.is_dir():
            shutil.copytree(item, target, symlinks=True)
        else:
            shutil.copy2(item, target)


def isolated_env(workspace: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["CODEX_PAPERS_ROOT"] = str(workspace / "papers")
    return env


def write_retry_context(paper_root: Path, shard: dict) -> None:
    """Expose retry guidance inside the isolated paper/condition workspace."""
    retry_of = shard.get("retry_of")
    context_path = paper_root / "retry_context.json"
    if not isinstance(retry_of, dict):
        if context_path.exists():
            context_path.unlink()
        return
    context = {
        "is_retry": True,
        "source_shard": retry_of.get("source_shard"),
        "previous_attempt_outputs": "outputs/previous_attempts",
        "failure_class": retry_of.get("failure_class"),
        "failure_classes": retry_of.get("failure_classes", []),
        "reason": retry_of.get("reason"),
        "retry_count": retry_of.get("retry_count", 0),
        "repair_strategy": retry_of.get("repair_strategy", {}),
    }
    write_json(context_path, context)


def import_retry_outputs(paper_root: Path, shard: dict, conditions: list[str]) -> None:
    """Make prior same-condition outputs visible to the next isolated retry agent."""
    retry_of = shard.get("retry_of")
    if not isinstance(retry_of, dict):
        return
    archive_root = retry_of.get("archive_root")
    source_shard = retry_of.get("source_shard")
    if not archive_root or not source_shard:
        return
    source_archive = Path(archive_root)
    source_shard_names = [str(source_shard)]
    try:
        for candidate in source_archive.iterdir():
            if candidate.is_dir() and not candidate.name.startswith("_") and candidate.name not in source_shard_names:
                source_shard_names.append(candidate.name)
    except OSError:
        pass
    for label in conditions:
        for shard_name in source_shard_names:
            src = source_archive / shard_name / f"condition_{label}" / "outputs"
            if not src.exists():
                continue
            dst = paper_root / "conditions" / f"condition_{label}" / "outputs" / "previous_attempts" / shard_name
            copy_if_exists(src, dst)


def archive_shard(
    *,
    paper_root: Path,
    archive_root: Path,
    shard_name: str,
    conditions: list[str],
    result_name: str,
) -> dict:
    shard_dir = archive_root / shard_name
    shard_dir.mkdir(parents=True, exist_ok=True)
    copy_if_exists(paper_root / "ground_truth.json", shard_dir / "ground_truth.json")
    copy_if_exists(paper_root / "reproduction_constraints.json", shard_dir / "reproduction_constraints.json")
    copy_if_exists(paper_root / "conditions_manifest.json", shard_dir / "conditions_manifest.json")
    expected = read_json(paper_root / "ground_truth.json").get("expected_metrics", [])

    archived = {
        "shard": shard_name,
        "expected_metrics": expected,
        "conditions": {},
    }
    for label in conditions:
        cond_dir = paper_root / "conditions" / f"condition_{label}"
        out_dir = shard_dir / f"condition_{label}"
        out_dir.mkdir(parents=True, exist_ok=True)
        for pattern in [
            result_name,
            "ground_truth.json",
            "reproduction_constraints.json",
            "data_manifest.json",
            "retry_context.json",
            "codex_manifest_*.json",
            "codex_stdout_*.log",
            "codex_stderr_*.log",
            "codex_prompt_*.txt",
        ]:
            for src in cond_dir.glob(pattern):
                copy_if_exists(src, out_dir / src.name)
        copy_if_exists(cond_dir / "outputs", out_dir / "outputs", ignore=ARCHIVE_OUTPUT_IGNORE)
        result_path = out_dir / result_name
        summary = validate_result(result_path, out_dir / "ground_truth.json")
        archived["conditions"][label] = {
            "result_path": str(result_path),
            "validation": summary,
        }
    write_json(shard_dir / "shard_archive_manifest.json", archived)
    return archived


def result_metric_map(result_path: Path) -> dict[tuple[str, str], dict]:
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


def merge_condition(
    *,
    paper_id: str,
    condition: str,
    archive_root: Path,
    shards: list[dict],
    result_name: str,
    model: str | None,
) -> dict:
    metrics = []
    assumptions = []
    summaries = []
    failures = []
    for shard in shards:
        shard_name = shard["name"]
        shard_dir = archive_root / shard_name
        shard_conditions = shard.get("_conditions_for_run", shard.get("conditions"))
        if shard_conditions and condition not in shard_conditions:
            continue
        expected = read_json(shard_dir / "ground_truth.json").get("expected_metrics", [])
        result_path = shard_dir / f"condition_{condition}" / result_name
        validation = validate_result(result_path, shard_dir / "ground_truth.json")
        result_metrics = result_metric_map(result_path)
        try:
            result = read_json(result_path) if result_path.exists() else {}
        except Exception:
            result = {}
        for item in result.get("assumptions", []):
            text = f"[{shard_name}] {item}"
            if text not in assumptions:
                assumptions.append(text)
        if result.get("execution_summary"):
            summaries.append(f"[{shard_name}] {result.get('execution_summary')}")
        if result.get("failure_diagnosis"):
            failures.append(f"[{shard_name}] {result.get('failure_diagnosis')}")

        for expected_metric in expected:
            key = (expected_metric.get("row_id"), expected_metric.get("col_id"))
            metric = result_metrics.get(key)
            if metric is None:
                metric = {
                    "row_id": key[0],
                    "col_id": key[1],
                    "value": None,
                    "notes": f"Shard {shard_name} did not produce this expected cell.",
                    "failure_class": validation.get("failure_class", "missing_result_file"),
                }
            else:
                metric = dict(metric)
                note = metric.get("notes", "")
                metric["notes"] = f"[{shard_name}] {note}".strip()
                if metric.get("value") is None:
                    metric.setdefault("failure_class", validation.get("failure_class", "null_metric"))
            metrics.append(metric)

    non_null = sum(m.get("value") is not None for m in metrics)
    status = "success" if non_null == len(metrics) else ("partial" if non_null else "execution_failed")
    failure_classes = []
    for metric in metrics:
        if metric.get("value") is None:
            failure_classes.append(metric.get("failure_class") or "null_metric")
    if status == "success":
        failure_classes = ["success"]
    elif status == "partial":
        failure_classes = ["partial"] + [c for c in failure_classes if c != "partial"]
    elif metrics and len(failure_classes) == len(metrics):
        failure_classes = ["all_null_metrics"] + [c for c in failure_classes if c != "all_null_metrics"]
    failure_classes = list(dict.fromkeys(failure_classes or ["implementation_error"]))
    return {
        "paper_id": paper_id,
        "status": status,
        "failure_class": failure_classes[0],
        "failure_classes": failure_classes,
        "seed": 42,
        "metrics": metrics,
        "assumptions": assumptions,
        "execution_summary": (
            f"Merged {len(shards)} sharded Codex reproduction runs for condition {condition} "
            f"using model {model or 'default'}. Produced {non_null}/{len(metrics)} non-null cells. "
            + " ".join(summaries[:3])
        ).strip(),
        "failure_diagnosis": None if status == "success" else " | ".join(failures) or f"{len(metrics) - non_null} cells were missing or null across shards.",
        "retry_recommended": status != "success",
        "retry_scope": "cell" if status == "partial" else ("condition" if status != "success" else "none"),
        "retry_hint": "Retry only null cells with targeted shards." if status == "partial" else ("Retry this condition shard set." if status != "success" else "No retry needed."),
        "sharded_merge": {
            "merged_at": now_iso(),
            "archive_root": str(archive_root),
            "shards": [s["name"] for s in shards],
            "condition": condition,
            "model": model or "default",
            "result_name": result_name,
        },
    }


def merged_ground_truth(archive_root: Path, shards: list[dict]) -> dict:
    base = None
    metrics = []
    tables_by_id = {}
    for shard in shards:
        gt = read_json(archive_root / shard["name"] / "ground_truth.json")
        if base is None:
            base = dict(gt)
        for table in gt.get("tables", []):
            tables_by_id[table.get("table_id", f"table_{len(tables_by_id)}")] = table
        metrics.extend(gt.get("expected_metrics", []))
    if base is None:
        base = {}
    base["tables"] = list(tables_by_id.values())
    base["expected_metrics"] = metrics
    base["sharded_merge"] = {
        "merged_at": now_iso(),
        "archive_root": str(archive_root),
        "shards": [s["name"] for s in shards],
        "n_expected_metrics": len(metrics),
    }
    notes = base.get("notes", "")
    merge_note = f"Merged ground truth from {len(shards)} sharded result-column runs."
    if merge_note not in notes:
        base["notes"] = (notes + " " + merge_note).strip()
    return base


def run_one_shard(
    *,
    index: int,
    total: int,
    shard: dict,
    args: argparse.Namespace,
    archive_root: Path,
    base_paper_root: Path,
    isolated: bool,
) -> dict:
    name = shard["name"]
    columns = shard.get("columns", [])
    rows = shard.get("rows", [])
    datasets = shard.get("datasets", [])
    shard_conditions = shard["_conditions_for_run"]

    if isolated:
        workspace = archive_root / "_workspaces" / name
        workspace_paper_root = workspace / "papers" / args.paper_id
        clone_paper_for_shard(base_paper_root, workspace_paper_root)
        paper_root = workspace_paper_root
        env = isolated_env(workspace)
        print(f"[sharded] shard {name}: isolated workspace={workspace_paper_root}", flush=True)
    else:
        paper_root = base_paper_root
        env = os.environ.copy()

    write_retry_context(paper_root, shard)

    print("\n" + "-" * 60, flush=True)
    print(
        f"[sharded] shard {index}/{total}: {name} "
        f"conditions={shard_conditions} columns={columns} rows={rows or '[all]'} datasets={datasets}",
        flush=True,
    )
    print("-" * 60, flush=True)

    setup_base = [
        sys.executable,
        str(CODEX_ROOT / "setup_codex_paper.py"),
        "--paper-id",
        args.paper_id,
        "--title",
        args.title,
        "--model",
        args.model or "",
    ]
    if not args.model:
        setup_base = [part for part in setup_base if part != "--model" and part != ""]

    filter_cmd = [
        sys.executable,
        str(CODEX_ROOT / "pipeline" / "filter_result_columns.py"),
        "--paper-id",
        args.paper_id,
        "--reason",
        f"Sharded reproduction run {archive_root.name}, shard {name}.",
    ]
    for col in columns:
        filter_cmd.extend(["--column", col])
    for row in rows:
        filter_cmd.extend(["--row", row])
    for dataset in datasets:
        filter_cmd.extend(["--dataset", dataset])
    run_cmd(filter_cmd, CODEX_ROOT.parent, env=env)

    force_stage = "conditions" if set(shard_conditions) <= {"A"} else "fake"
    setup_conditions = setup_base + [
        "--stop-at",
        "conditions",
        "--force-stage",
        force_stage,
        "--allow-partial-data",
    ]
    if args.allow_network_data:
        setup_conditions.append("--allow-network-data")
    run_cmd(setup_conditions, CODEX_ROOT.parent, env=env)
    import_retry_outputs(paper_root, shard, shard_conditions)

    run_experiment_cmd = [
        sys.executable,
        str(CODEX_ROOT / "run_codex_experiment.py"),
        "--paper",
        args.paper_id,
        "--conditions",
        ",".join(shard_conditions),
        "--parallel",
        str(max(1, args.parallel)),
        "--timeout-seconds",
        str(args.timeout_seconds),
        "--result-name",
        args.result_name,
        "--force",
    ]
    if args.model:
        run_experiment_cmd.extend(["--model", args.model])
    run_cmd(run_experiment_cmd, CODEX_ROOT.parent, env=env)

    archived = archive_shard(
        paper_root=paper_root,
        archive_root=archive_root,
        shard_name=name,
        conditions=shard_conditions,
        result_name=args.result_name,
    )
    return {
        "name": name,
        "columns": columns,
        "rows": rows,
        "datasets": datasets,
        "conditions_requested": shard_conditions,
        "archive": str(archive_root / name),
        "conditions": archived["conditions"],
        "isolated_workspace": str(paper_root) if isolated else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper-id", required=True)
    parser.add_argument("--title", required=True)
    parser.add_argument("--shard-plan", required=True)
    parser.add_argument("--model", default=None)
    parser.add_argument("--conditions", default="A,B,C")
    parser.add_argument("--parallel", type=int, default=3)
    parser.add_argument("--timeout-seconds", type=int, default=21600)
    parser.add_argument("--result-name", default="result.json")
    parser.add_argument("--merged-result-name", default="result_merged.json")
    parser.add_argument("--allow-network-data", action="store_true")
    parser.add_argument(
        "--shard-parallel",
        type=int,
        default=1,
        help="Number of shards to run concurrently. Values >1 require --isolated-shard-workspaces.",
    )
    parser.add_argument(
        "--isolated-shard-workspaces",
        action="store_true",
        help="Run each shard against a private paper workspace under the archive directory.",
    )
    args = parser.parse_args()

    paper_root = PAPERS_ROOT / args.paper_id
    plan = read_json(Path(args.shard_plan))
    shards = plan.get("shards", [])
    if not shards:
        raise RuntimeError(f"no shards found in {args.shard_plan}")
    conditions = [c.strip().upper() for c in args.conditions.split(",") if c.strip()]
    if not conditions:
        raise RuntimeError("no conditions requested")
    shard_parallel = max(1, args.shard_parallel)
    isolated = bool(args.isolated_shard_workspaces)
    if shard_parallel > 1 and not isolated:
        raise RuntimeError("--shard-parallel > 1 requires --isolated-shard-workspaces")

    archive_root = paper_root / "sharded_runs" / f"{Path(args.shard_plan).stem}_{stamp()}"
    archive_root.mkdir(parents=True)
    run_manifest = {
        "paper_id": args.paper_id,
        "title": args.title,
        "model": args.model or "default",
        "started_at": now_iso(),
        "conditions": conditions,
        "parallel": args.parallel,
        "timeout_seconds": args.timeout_seconds,
        "result_name": args.result_name,
        "merged_result_name": args.merged_result_name,
        "shard_parallel": shard_parallel,
        "isolated_shard_workspaces": isolated,
        "shard_plan": str(Path(args.shard_plan).resolve()),
        "archive_root": str(archive_root),
        "shards": [],
        "status": "running",
    }
    write_json(archive_root / "sharded_run_manifest.json", run_manifest)

    print(
        f"[sharded] paper={args.paper_id} model={args.model or 'default'} archive={archive_root} "
        f"shard_parallel={shard_parallel} isolated={isolated}",
        flush=True,
    )
    planned_shards = []
    for shard in shards:
        shard_conditions = [
            c.strip().upper()
            for c in shard.get("conditions", conditions)
            if str(c).strip()
        ]
        if not shard_conditions:
            raise RuntimeError(f"shard {shard['name']} has no runnable conditions")
        planned = dict(shard)
        planned["_conditions_for_run"] = shard_conditions
        planned_shards.append(planned)

    if shard_parallel == 1:
        for i, shard in enumerate(planned_shards, 1):
            entry = run_one_shard(
                index=i,
                total=len(planned_shards),
                shard=shard,
                args=args,
                archive_root=archive_root,
                base_paper_root=paper_root,
                isolated=isolated,
            )
            run_manifest["shards"].append(entry)
            write_json(archive_root / "sharded_run_manifest.json", run_manifest)
    else:
        completed = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=shard_parallel) as pool:
            futures = {
                pool.submit(
                    run_one_shard,
                    index=i,
                    total=len(planned_shards),
                    shard=shard,
                    args=args,
                    archive_root=archive_root,
                    base_paper_root=paper_root,
                    isolated=isolated,
                ): i
                for i, shard in enumerate(planned_shards, 1)
            }
            for future in concurrent.futures.as_completed(futures):
                index = futures[future]
                completed[index] = future.result()
                run_manifest["shards"] = [completed[i] for i in sorted(completed)]
                write_json(archive_root / "sharded_run_manifest.json", run_manifest)

    print("\n" + "-" * 60, flush=True)
    print("[sharded] merging shard results", flush=True)
    print("-" * 60, flush=True)
    merged = {}
    gt_merged = merged_ground_truth(archive_root, planned_shards)
    write_json(archive_root / "merged" / "ground_truth_merged.json", gt_merged)
    for condition in conditions:
        merged_result = merge_condition(
            paper_id=args.paper_id,
            condition=condition,
            archive_root=archive_root,
            shards=planned_shards,
            result_name=args.result_name,
            model=args.model,
        )
        merged[condition] = merged_result
        gt_merged_path = paper_root / "conditions" / f"condition_{condition}" / "ground_truth_merged.json"
        write_json(gt_merged_path, gt_merged)
        merged_path = paper_root / "conditions" / f"condition_{condition}" / args.merged_result_name
        write_json(merged_path, merged_result)
        copy_if_exists(merged_path, archive_root / "merged" / f"condition_{condition}_{args.merged_result_name}")
        copy_if_exists(gt_merged_path, archive_root / "merged" / f"condition_{condition}_ground_truth_merged.json")
        print(
            f"[sharded] condition_{condition}: {merged_result['status']} "
            f"{sum(m.get('value') is not None for m in merged_result['metrics'])}/{len(merged_result['metrics'])} non-null -> {merged_path}",
            flush=True,
        )

    run_manifest["completed_at"] = now_iso()
    run_manifest["status"] = "success" if all(m["status"] == "success" for m in merged.values()) else "partial_or_failed"
    write_json(archive_root / "sharded_run_manifest.json", run_manifest)
    print(f"[sharded] manifest={archive_root / 'sharded_run_manifest.json'}", flush=True)


if __name__ == "__main__":
    main()
