#!/usr/bin/env python3
"""Prepare the public dataset portion of the CellPress medical benchmark.

The script deliberately preserves the published splits.  It downloads MedQA
and MedMCQA through Hugging Face and the exact PubMedQA archive referenced by
the paper authors' dataset builder.  It does not download or reconstruct the
held-out MedMCQA test labels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


CODEX_ROOT = Path(__file__).resolve().parent
PAPERS_ROOT = CODEX_ROOT / "papers"
PUBMEDQA_AUTHORS_URL = (
    "https://f001.backblazeb2.com/file/FindZebraData/"
    "fz-openqa/datasets/pubmedqa.zip"
)
EXPECTED_COUNTS = {
    "MedQA-USMLE/test": 1273,
    "MedMCQA/train": 182822,
    "MedMCQA/validation": 4183,
    "PubMedQA/train": 450,
    "PubMedQA/validation": 50,
    "PubMedQA/test": 500,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_record_count(path: Path) -> int:
    if path.suffix == ".jsonl":
        with path.open(errors="replace") as handle:
            return sum(1 for line in handle if line.strip())
    payload = json.loads(path.read_text())
    return len(payload) if isinstance(payload, (dict, list)) else 0


def write_dataset_jsonl(dataset: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_json(str(path), orient="records", lines=True, force_ascii=False)


def prepare_huggingface(paper_root: Path, *, force: bool) -> list[Path]:
    try:
        from datasets import load_dataset  # type: ignore[import-not-found]
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "Dataset preparation requires the `datasets` and `pyarrow` packages. "
            "Install codex_version/requirements_experiment.txt first."
        ) from exc

    outputs: list[Path] = []
    medqa_path = paper_root / "data" / "MedQA-USMLE" / "test.jsonl"
    if force or not medqa_path.exists():
        medqa = load_dataset("GBaker/MedQA-USMLE-4-options", split="test")
        write_dataset_jsonl(medqa, medqa_path)
    outputs.append(medqa_path)

    medmcqa = None
    for split in ["train", "validation"]:
        path = paper_root / "data" / "MedMCQA" / f"{split}.jsonl"
        if force or not path.exists():
            if medmcqa is None:
                medmcqa = load_dataset("openlifescienceai/medmcqa")
            write_dataset_jsonl(medmcqa[split], path)
        outputs.append(path)
    return outputs


def prepare_pubmedqa(paper_root: Path, *, force: bool) -> list[Path]:
    output_dir = paper_root / "data" / "PubMedQA"
    outputs = [output_dir / f"{split}.json" for split in ["train", "validation", "test"]]
    if not force and all(path.exists() for path in outputs):
        return outputs

    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="pubmedqa-") as tmp:
        archive = Path(tmp) / "pubmedqa.zip"
        urllib.request.urlretrieve(PUBMEDQA_AUTHORS_URL, archive)
        with zipfile.ZipFile(archive) as handle:
            handle.extractall(Path(tmp) / "extracted")
        for split, destination in zip(["train", "validation", "test"], outputs):
            matches = list((Path(tmp) / "extracted").rglob(f"{split}_set.json"))
            if len(matches) != 1:
                raise RuntimeError(
                    f"Expected one {split}_set.json in PubMedQA archive; found {matches}."
                )
            shutil.copyfile(matches[0], destination)
    return outputs


def validate_outputs(paths: list[Path], *, allow_count_mismatch: bool) -> list[dict[str, Any]]:
    rows = []
    for path in paths:
        dataset = path.parent.name
        split = path.stem
        key = f"{dataset}/{split}"
        count = json_record_count(path)
        expected = EXPECTED_COUNTS.get(key)
        if expected is not None and count != expected and not allow_count_mismatch:
            raise RuntimeError(
                f"{key} contains {count} records; expected {expected}. "
                "Use --allow-count-mismatch only after auditing the source revision."
            )
        rows.append(
            {
                "dataset_split": key,
                "path": str(path),
                "records": count,
                "expected_records": expected,
                "sha256": sha256_file(path),
            }
        )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper-id", default="cellpress-medical-reasoning")
    parser.add_argument("--paper-root", type=Path)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--allow-count-mismatch", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paper_root = args.paper_root or PAPERS_ROOT / args.paper_id
    paths = prepare_huggingface(paper_root, force=args.force)
    paths.extend(prepare_pubmedqa(paper_root, force=args.force))
    records = validate_outputs(
        paths, allow_count_mismatch=args.allow_count_mismatch
    )
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scope": "public main-result datasets only",
        "sources": {
            "MedQA-USMLE": "GBaker/MedQA-USMLE-4-options",
            "MedMCQA": "openlifescienceai/medmcqa",
            "PubMedQA": PUBMEDQA_AUTHORS_URL,
        },
        "files": records,
        "limitations": [
            "MedMCQA test labels are held out and are not prepared.",
            "The main replication uses the public 1,000-question validation subset.",
        ],
    }
    manifest_path = paper_root / "data" / "dataset_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"[medical-data] manifest={manifest_path}")
    for row in records:
        print(f"[medical-data] {row['dataset_split']} records={row['records']}")


if __name__ == "__main__":
    main()
