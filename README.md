# LatentDemandBench

A synthetic benchmark for predicting product–need pairings before commercial
activation. Includes the generator, rendered dossiers, model prompts, scoring,
baselines, experiment runners, regression tests, and recorded results with raw
model responses.

## Setup

Python 3.11 or newer is required. From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.lock
python -m pip install --no-deps -e .
```

`requirements.lock` records the environment tested for this release (Python
3.13); it is not a record of the original model-generation environment. For an
unpinned installation, use `python -m pip install -e '.[dev]'` instead.

## Reproduce offline

```bash
make reproduce
```

This runs the regression suite and rebuilds the result tables, numerical macros,
two figures, and exact prompt/schema export into `reproduced/`. Tests verify the
generated tables and prompt export against reference SHA-256 digests. No API
keys, response cache, LaTeX installation, or manuscript files are needed.

```bash
make baselines  # regenerate 100 baseline universes and 60 twin seeds
make rescore   # dry-run rescoring of archived responses; no files overwritten
```

`make baselines` writes fresh results into `reproduced/`. The independent random
control used in the reported tables is supplied by
`results/random_baseline_audit.json`; regenerate it with
`python tools/random_baseline_audit.py --out reproduced/random_baseline_audit.json`.
`make rescore` recomputes the migrated arms and preserves their original generation
metadata; current native runs are skipped by that migration command. Regression
tests also check scoring of the corrected large-economy responses.

The underlying commands are:

```bash
python -m pytest
python tools/paper_tables.py
python tools/figures.py
python tools/prompt_appendix.py
python tools/rescore_review.py
```

Additional CPU-only analysis tools regenerate the corresponding JSON artifacts
under `results/` by default (use `--out` to write elsewhere):

- `train_curve.py`, `train_resample.py`: held-out reference learning curves and resampling.
- `sensitivity.py`, `observation_loss.py`: generator sensitivity and observation loss.
- `probe_calibration.py`, `probe_and_subset.py`, `interactions.py`: probe and paired analyses.
- `parser_reference.py`: dossier parsing and parse-then-rank; use `--surface opaque`
  with `--out results/parser_reference_opaque.json` for the opaque surface.
- `spec_tables.py`, `random_baseline_audit.py`: output-contract accounting and random control.

All these scripts live in `tools/`. The canonical table and figure generators
consume the archived artifacts, including the independent random-control audit.

## Run new model experiments

New completions require model access and incur provider charges. The recorded
model identifiers and settings are in each artifact's `config.backend`; access
to those same identifiers is required to rerun them. Hosted model revisions may
change, so archived responses are the reproducible input for offline scoring.

With `OPENAI_API_KEY` set in your environment:

```bash
python tools/run_suite.py --backend openai --model gpt-5.6-luna \
  --effort medium --max-workers 2 --outdir reproduced/luna_full
python tools/run_suite.py --backend openai --model gpt-5.6-terra \
  --effort medium --max-workers 2 --outdir reproduced/terra_full
```

Each suite uses 100 baseline universes, 30 model universes, and the complete
surface/assistance/horizon/slope/size/replicate/twin experiment grid. It uses one
final top-K budget after averaging replicate completions. The large-economy
condition uses K=90. Inspect `--help` for smaller runs.

For the recorded open-weight configuration, install `.[openweight]`, authenticate
with Modal, and run:

```bash
python tools/run_open_weight.py \
  --universes 100 --llm-universes 30 --surface finance \
  --model Qwen/Qwen3-32B --gpu H100:2 --tensor-parallel 2 \
  --max-tokens 24000 --max-model-len 32768 --enable-thinking false \
  --out reproduced/openweight/model_finance.json
```

The remote vLLM/Transformers versions and decoding parameters are defined in
`tools/modal_batch.py`. Gemini and Anthropic backends are also available. The
Codex CLI backend is exploratory because it can read local files; reported
artifacts use isolated backends.

## Files and provenance

- `ldb/`: generator, observation layer, prompts, baselines, metrics, and backends.
- `run_v0.py`, `run_twins.py`: canonical evaluation runners.
- `tools/`: experiment orchestration, offline rescoring, and result generation.
- `tests/`: generator, protocol, artifact, and output regression checks.
- `results/`: recorded metrics, raw model responses, seeds, and configurations.
- `human/`: example dossier and hand-checking packet; no human baseline was collected.

See [results/README.md](results/README.md) for the artifact inventory and provenance.
The core source is preserved byte-for-byte, including `make_tables.py`, whose
formatting helpers are imported by the main table generator. Its legacy CLI
should be given an explicit `--out`; use `tools/paper_tables.py` for the complete
recorded analysis. The source hash covers `ldb/**/*.py` and root Python files;
changing those files invalidates existing artifact hashes.
