# Evaluating memorization and question-answering ability on medical benchmarks

Code and analysis artifacts for *Do Large Language Models know medical questions?
Evaluating memorization and question-answering ability on medical benchmarks.*

Ten language models from five providers, taken as a smaller/larger pair from
each, were evaluated on 2,773 questions from three medical benchmarks under
three settings:

1. **Multiple choice** — the standard format, five-shot chain of thought.
2. **Generative** — the same questions with the answer choices withheld, graded
   by a frozen LLM-as-a-judge rubric.
3. **Reconstruction** — the model sees only a question's answer choices and must
   reconstruct the question they belong to, used as a bounded proxy for prior
   exposure to the benchmark.

---

## What is in this repository, and what is in the archive

This repository holds the **code and the small artifacts**: every script, the
prompt templates, both judge rubrics, the run configurations, per-run manifests
and metrics, the derived tables, and the figures as they appear in the paper.

The **model generations and per-question judge verdicts** (~1.35 GB) are too
large for a code repository and are deposited separately:

> **Archive:** https://doi.org/10.5281/zenodo.21736709 (CC-BY-4.0)
>
> That is the concept DOI and always resolves to the latest version.
> The version deposited alongside this commit is
> [10.5281/zenodo.21736710](https://doi.org/10.5281/zenodo.21736710).

Both are needed to re-derive the paper's numbers from scratch. The code here
runs against the archive's `outputs/` directory.

```
code/
  harness/     runs the models under each setting
  judge/       runs the judge and finalizes its verdicts
  analysis/    per-benchmark extraction and result summaries
  figures/     regenerates every figure in the paper
  configs/     run configurations, including exploratory conditions not reported
prompts/       the five message templates, exactly as issued
derived/       small tables underlying the paper's numbers
figures/       the figures as they appear in the paper
outputs/       per-run manifests, configs, and metrics (generations are in the archive)
```

## Reproducing the figures and statistics

The analysis and figure scripts read stored outputs and make no network calls.

```bash
pip install numpy matplotlib
python code/figures/make_paper_figures.py
```

Paths at the top of each figure script point at a run directory; set them to the
`outputs/` directory from the archive. Nothing here needs API access.

Re-running the model queries is a separate matter and is not required to
reproduce any reported number. It needs an OpenRouter API key supplied through
the `OPENROUTER_API_KEY` environment variable. **No credentials are stored in
this repository**; the configs reference environment variable names only.

## The benchmarks

The three benchmarks are public and are not redistributed here.
`derived/question_set_fingerprints.csv` gives a SHA-256 over the sorted question
identifiers of each evaluation set, so the exact subsets used can be
reconstructed rather than approximated. MedQA-USMLE is the full 1,273-question
test split and PubMedQA the full 500-question expert-annotated test split.
MedMCQA is a fixed seed-0 1,000-question sample of the validation split, because
its test labels are released only through leaderboard submission.

## Three things worth knowing before reading the results

**Decoding temperature is not uniform.** Eight models were run with greedy
decoding at temperature 0. DeepSeek V4 Pro and Qwen 3.7 Plus were run at
temperature 0.5, from a single shared run that evaluated both. Every run's
recorded temperature is in its `run_manifest.json` and summarized in
`derived/run_manifest_summary.csv`.

**Qwen 3.5 9B was configured differently across settings.** Its multiple-choice
run used `enable_thinking: true`; its generative and reconstruction run used
`reasoning: {effort: none, exclude: true}`. The other nine models used the same
reasoning configuration in both. This matters when reading Qwen 3.5 9B's
between-setting difference, the largest in the cohort, because some share of it
may belong to the reasoning configuration rather than to the removal of the
answer choices. Both configurations are in the deposited config snapshots.

**Settings 2 and 3 are zero-shot; Setting 1 is five-shot.** The five worked
exemplars appear only in Setting 1, so the between-setting comparison reflects
removing the answer scaffold and removing the exemplars together. The templates
in `prompts/` show this directly.

## A caveat on re-running

All models were accessed through a commercial inference API rather than run
locally. Provider-hosted endpoints can be updated or withdrawn without notice,
so re-querying the same model identifiers later is not guaranteed to reproduce
these generations. The deposited outputs, not live re-querying, are the
reproducible record.

## Citation

Citation details will be added on publication. The archived outputs are cited as
https://doi.org/10.5281/zenodo.21736709.
