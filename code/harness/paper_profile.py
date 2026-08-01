#!/usr/bin/env python3
"""Infer execution profile for a paper reproduction.

The setup agents extract authoritative reproduction constraints, but the shell
orchestrators also need a small deterministic contract: is this a local CPU/GPU
paper, or does the reproduction need API-backed language-model calls?  This
module keeps that decision auditable and reusable by setup, preflight, retry,
and tmux runners.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


CODEX_ROOT = Path(__file__).resolve().parent
PAPERS_ROOT = Path(os.environ.get("CODEX_PAPERS_ROOT", CODEX_ROOT / "papers")).resolve()

LLM_KEYWORDS = [
    "generated knowledge prompting",
    "chain-of-thought",
    "chain of thought",
    "few-shot prompting",
    "zero-shot prompting",
    "prompted language model",
    "large language model",
    "language model prompting",
    "in-context learning",
    "commonsense reasoning",
    "openai api",
    "gpt-3",
    "gpt3",
    "text-davinci",
    "instructgpt",
]

MODEL_PATTERNS = [
    r"\btext-davinci-\d{3}\b",
    r"\bcode-davinci-\d{3}\b",
    r"\bgpt-4(?:[-_.][A-Za-z0-9]+)*\b",
    r"\bgpt-3\.5(?:[-_.][A-Za-z0-9]+)*\b",
    r"\bgpt-?3\b",
    r"\binstructgpt\b",
    r"\bchatgpt\b",
    r"\bllama[- ]?\d+(?:[- ][A-Za-z0-9.]+)*\b",
    r"\bt5[- ]?(?:small|base|large|xl|xxl|11b|3b)?\b",
    r"\bul2\b",
]


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def read_source_text(source_dir: Path, limit_chars: int = 8_000_000) -> str:
    chunks: list[str] = []
    total = 0
    suffix_rank = {".tex": 0, ".bbl": 1, ".bib": 2, ".md": 3, ".txt": 4}
    paths = sorted(
        (p for p in source_dir.rglob("*") if p.is_file()),
        key=lambda p: (suffix_rank.get(p.suffix.lower(), 99), p.as_posix()),
    )
    for path in paths:
        if not path.is_file() or path.suffix.lower() not in {
            ".tex",
            ".bib",
            ".bbl",
            ".md",
            ".txt",
        }:
            continue
        try:
            text = path.read_text(errors="replace")
        except Exception:
            continue
        remaining = limit_chars - total
        if remaining <= 0:
            break
        chunks.append(text[:remaining])
        total += len(chunks[-1])
    return "\n".join(chunks)


def normalize_model_name(name: str) -> str:
    return re.sub(r"\s+", "-", name.strip()).strip(".,;:()[]{}").lower()


def extract_source_models(text: str) -> list[str]:
    models: set[str] = set()
    for pattern in MODEL_PATTERNS:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            model = normalize_model_name(match.group(0))
            if model and len(model) >= 3:
                models.add(model)
    return sorted(models)


def constraints_llm_models(constraints: dict[str, Any]) -> list[str]:
    llm_api = constraints.get("llm_api")
    if not isinstance(llm_api, dict):
        return []
    raw = llm_api.get("paper_models") or []
    models: set[str] = set()
    if isinstance(raw, list):
        for item in raw:
            if isinstance(item, str):
                models.add(normalize_model_name(item))
            elif isinstance(item, dict):
                for key in ["model", "model_name", "engine", "api_model", "paper_model"]:
                    value = item.get(key)
                    if isinstance(value, str) and value.strip():
                        models.add(normalize_model_name(value))
    elif isinstance(raw, str):
        models.add(normalize_model_name(raw))
    return sorted(m for m in models if m)


def infer_paper_profile(paper_id: str, paper_root: Path | None = None) -> dict[str, Any]:
    paper_root = paper_root or PAPERS_ROOT / paper_id
    source_text = read_source_text(paper_root / "source")
    lower = source_text.lower()
    constraints = read_json(paper_root / "reproduction_constraints.json")
    llm_api = constraints.get("llm_api") if isinstance(constraints.get("llm_api"), dict) else {}
    constraints_says_llm = bool(llm_api.get("uses_llm_api")) if isinstance(llm_api, dict) else False
    keyword_hits = [kw for kw in LLM_KEYWORDS if kw in lower]
    paper_models = constraints_llm_models(constraints) or extract_source_models(source_text)

    execution_kind = "llm_api" if constraints_says_llm or len(keyword_hits) >= 2 else "local_compute"
    confidence = "high" if constraints_says_llm else ("medium" if execution_kind == "llm_api" else "low")

    env: dict[str, str] = {}
    warnings: list[str] = []
    if execution_kind == "llm_api":
        env["EXPERIMENT_LLM_PROVIDER"] = "openrouter"
        env["EXPERIMENT_LLM_BASE_URL"] = "https://openrouter.ai/api/v1"
        env["ALLOW_NETWORK_DATA"] = "1"
        if not paper_models:
            warnings.append(
                "No paper-stated model was detected; EXPERIMENT_LLM_MODEL must be set as a fallback."
            )
        else:
            warnings.append(
                "Paper-stated model names are authoritative; EXPERIMENT_LLM_MODEL is only a fallback."
            )

    return {
        "schema_version": 1,
        "paper_id": paper_id,
        "created_at": now_iso(),
        "execution_kind": execution_kind,
        "confidence": confidence,
        "uses_llm_api": execution_kind == "llm_api",
        "source": {
            "constraints_llm_api": constraints_says_llm,
            "keyword_hits": keyword_hits,
            "paper_models": paper_models,
        },
        "recommended_environment": env,
        "warnings": warnings,
    }


def write_paper_profile(paper_id: str, paper_root: Path | None = None) -> Path:
    paper_root = paper_root or PAPERS_ROOT / paper_id
    profile = infer_paper_profile(paper_id, paper_root)
    path = paper_root / "paper_profile.json"
    path.write_text(json.dumps(profile, indent=2))
    return path


def shell_exports(profile: dict[str, Any], preserve_existing: bool = True) -> str:
    env = profile.get("recommended_environment") or {}
    lines: list[str] = []
    for key, value in sorted(env.items()):
        if preserve_existing:
            lines.append(f'if [[ -z "${{{key}:-}}" ]]; then export {key}={shlex.quote(str(value))}; fi')
        else:
            lines.append(f"export {key}={shlex.quote(str(value))}")
    if profile.get("uses_llm_api"):
        lines.append('export EXPERIMENT_LLM_PAPER_PROFILE="${EXPERIMENT_LLM_PAPER_PROFILE:-1}"')
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper-id", required=True)
    parser.add_argument("--paper-root")
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--format", choices=["json", "shell"], default="json")
    parser.add_argument("--overwrite-env", action="store_true")
    args = parser.parse_args()

    paper_root = Path(args.paper_root).resolve() if args.paper_root else PAPERS_ROOT / args.paper_id
    if args.write:
        write_paper_profile(args.paper_id, paper_root)
    profile = infer_paper_profile(args.paper_id, paper_root)
    if args.format == "shell":
        print(shell_exports(profile, preserve_existing=not args.overwrite_env))
    else:
        print(json.dumps(profile, indent=2))


if __name__ == "__main__":
    main()
