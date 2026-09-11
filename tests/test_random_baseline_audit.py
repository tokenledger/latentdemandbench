"""Regression checks for the independently seeded random-control sidecar."""

import json
from pathlib import Path

import numpy as np
import pytest

from ldb.baselines import random_rank
from ldb.llm import apply_top_k_protocol
from ldb.metrics import evaluate
from ldb.task import build_instance
from tools.random_baseline_audit import RNG_NAMESPACE

ROOT = Path(__file__).resolve().parent.parent


def test_committed_random_rows_use_an_independent_seed_namespace():
    audit = json.loads((ROOT / "results/random_baseline_audit.json").read_text())
    assert audit["config"]["rng_namespace"] == RNG_NAMESPACE
    assert audit["config"]["seed_rule"] == (
        "SeedSequence([universe_seed, rng_namespace])"
    )

    for universe_seed in range(3):
        inst = build_instance(universe_seed)
        raw = random_rank(
            inst, seed=np.random.SeedSequence([universe_seed, RNG_NAMESPACE])
        )
        scores, _ = apply_top_k_protocol(raw, raw, 40)
        expected = evaluate(inst, scores, probe_scores=raw)
        stored = audit["per_universe"]["random"][universe_seed]
        assert stored["auc"] == pytest.approx(expected["auc"])
        assert stored["gap_auc"] == pytest.approx(expected["gap_auc"])
        assert stored["probe_auc"] == pytest.approx(expected["probe_auc"])


def test_namespaced_stream_is_not_the_historical_universe_seed_stream():
    historical = np.random.default_rng(0).random(16)
    audited = np.random.default_rng(
        np.random.SeedSequence([0, RNG_NAMESPACE])
    ).random(16)
    assert not np.array_equal(historical, audited)
