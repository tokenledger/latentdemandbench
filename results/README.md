# Recorded results

| Files | Purpose |
|---|---|
| `baselines.json`, `twins.json` | Baseline ranking and counterfactual results |
| `luna_full/`, `terra_full/` | Complete 16-condition model suites, including raw responses |
| `openweight/model_finance.json` | Qwen3-32B finance arm and raw responses |
| Root-level analysis JSON files | Learning curves, resampling, probe/subset analyses, interactions, parser checks, observation loss, and output-contract accounting |
| `random_baseline_audit.json` | Independent random-control rows applied by the table generator |
| `sol/model_finance.json`, `terra/model_finance.json` | Two historical artifacts required only for the explicitly historical AUC comparison |

The historical files use an earlier protocol. They must not be pooled with the
current results or used to support current auxiliary metrics. Other superseded
runs are excluded.

Current artifacts carry source hash `6b2630d26248899e`. For offline-rescored
artifacts, `config.code_hash` identifies the scoring source;
`config.generation` retains the original model-generation settings and source
hash. `config.rescoring` records the original artifact digest, raw-response
digest, generation/observation contract digest, and scoring-script digest.
Original commit IDs are provenance labels from the development repository;
that repository's history is not required to run the included scoring code.

Raw responses are embedded in `model_outputs` (both worlds for twins), so
rescoring needs no private cache. Seeds and configuration reconstruct the
synthetic universes. The two current `size_large.json` artifacts contain fresh
K=90 runs and record that distinction in `review_rerun` metadata.

The archived OpenAI runs preserve requested model names and reasoning effort,
but not immutable returned model revisions or request timing. Their legacy
`temperature: 0.0` metadata does not represent a sent temperature parameter.
The open-weight artifact records its runtime versions, decoding settings, and
batch timing. Offline scoring can reproduce the recorded responses; new model
calls are separate experiments and may produce different answers.
