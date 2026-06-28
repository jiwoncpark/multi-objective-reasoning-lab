"""Tests for ``scripts/preflight_sweep.py`` (shrunk run on a tiny fixture pool).

The full preflight sweeps the real assets and is slow; here we prove the loop and
the metrics execute on a small synthetic pool with 2 strategies and 2 rounds.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch

from mobo_lab import config
from mobo_lab.oracle import AntibodyOracle
from mobo_lab.pool import VHSequencePool

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "preflight_sweep", REPO_ROOT / "scripts" / "preflight_sweep.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


pf = _load_module()


@pytest.fixture(scope="module")
def tiny():
    torch.manual_seed(0)
    n = 24
    X = torch.rand(n, config.LATENT_DIM, dtype=torch.double)
    pool = VHSequencePool(X, [f"SEQ{i}" for i in range(n)])
    # A smooth-ish true objective table so the GP has signal; mild trade-off.
    Y_true = torch.stack([X[:, 0], 1.0 - 0.7 * X[:, 0] + 0.3 * X[:, 1]], dim=-1)
    oracle = AntibodyOracle(Y_true, allow_true=True)
    initial_ids = [0, 1, 2, 3, 4, 5]
    return pool, oracle, initial_ids


def test_run_campaign_shapes_and_progress(tiny):
    pool, oracle, initial_ids = tiny
    rounds = [{"nehvi": config.BATCH_SIZE}, {"nehvi": config.BATCH_SIZE}]
    result = pf.run_campaign(rounds, pool, oracle, initial_ids, seed=0)
    assert len(result["hv_history"]) == len(rounds) + 1
    assert len(result["selected_ids"]) == config.BATCH_SIZE * len(rounds)
    assert len(set(result["selected_ids"])) == len(result["selected_ids"])
    assert not (set(result["selected_ids"]) & set(initial_ids))
    # hypervolume is non-decreasing along the campaign (re-query-stable oracle)
    hv = result["hv_history"]
    assert all(hv[i + 1] >= hv[i] - 1e-9 for i in range(len(hv) - 1))


def test_run_campaign_is_deterministic(tiny):
    pool, oracle, initial_ids = tiny
    rounds = [{"nehvi": 2, "parego": 2}, {"random": 4}]
    a = pf.run_campaign(rounds, pool, oracle, initial_ids, seed=1)
    b = pf.run_campaign(rounds, pool, oracle, initial_ids, seed=1)
    assert a["selected_ids"] == b["selected_ids"]
    assert a["hv_history"] == b["hv_history"]


def test_campaign_metrics_keys(tiny):
    pool, oracle, initial_ids = tiny
    result = pf.run_campaign([{"nehvi": 4}, {"nehvi": 4}], pool, oracle, initial_ids, seed=0)
    m = pf.campaign_metrics(result, pool=pool)
    for key in (
        "auc_hv",
        "final_hv",
        "hv_gain_per_round",
        "n_selected_nondominated",
        "selected_diversity",
        "angular_spread",
        "region_coverage",
    ):
        assert key in m
    assert len(m["hv_gain_per_round"]) == 2


def test_run_sweep_shrunk(tiny):
    pool, oracle, initial_ids = tiny
    strategies = {
        "all_nehvi": [{"nehvi": 4}, {"nehvi": 4}],
        "random_baseline": [{"random": 4}, {"random": 4}],
    }
    sweep = pf.run_sweep(strategies, pool, oracle, initial_ids, seeds=[0, 1])
    assert set(sweep) == {"all_nehvi", "random_baseline"}
    for stats in sweep.values():
        assert len(stats["auc_hv_by_seed"]) == 2
        assert "auc_hv_mean" in stats


def test_discrete_acq_margin_runs(tiny):
    pool, oracle, initial_ids = tiny
    margin = pf.discrete_acq_margin(pool, oracle, initial_ids, seed=0)
    assert margin["qth_value"] >= margin["next_value"]  # sorted descending
    assert "rel_margin" in margin


def test_write_plots(tiny, tmp_path):
    pool, oracle, initial_ids = tiny
    strategies = pf.default_strategies(n_rounds=2)
    n = pf.write_plots(strategies, pool, oracle, initial_ids, tmp_path, seed=0)
    assert n == 2
    assert (tmp_path / "hv_curves.png").exists()
    assert (tmp_path / "selection_coverage.png").exists()


def test_default_strategies_round_count():
    strategies = pf.default_strategies(n_rounds=3)
    assert all(len(rounds) == 3 for rounds in strategies.values())
    assert "all_nehvi" in strategies and "random_baseline" in strategies
