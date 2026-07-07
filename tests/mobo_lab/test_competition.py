"""Tests for ``mobo_lab/competition.py`` (campaign driver + leaderboard + reveal)."""

from __future__ import annotations

import pytest
import torch

from mobo_lab import competition, config
from mobo_lab.oracle import AntibodyOracle
from mobo_lab.pool import VHSequencePool


@pytest.fixture(scope="module")
def fixture_contest():
    torch.manual_seed(0)
    n = 28
    X = torch.rand(n, config.LATENT_DIM, dtype=torch.double)
    pool = VHSequencePool(X, [f"SEQ{i}" for i in range(n)])
    Y_true = torch.stack([X[:, 0], 1.0 - 0.6 * X[:, 0] + 0.4 * X[:, 1]], dim=-1)
    oracle = AntibodyOracle(Y_true, allow_true=False)
    initial_ids = [0, 1, 2, 3]
    return pool, oracle, initial_ids


def _run(strategy, name, fixture_contest, **kw):
    pool, oracle, initial_ids = fixture_contest
    return competition.run_campaign(
        strategy, name, pool=pool, oracle=oracle, initial_ids=initial_ids,
        n_rounds=len(strategy), **kw,
    )


def test_run_campaign_history_and_invariants(fixture_contest):
    pool, oracle, initial_ids = fixture_contest
    strategy = [{"nehvi": 4}, {"nehvi": 2, "parego": 2}]
    h = _run(strategy, "Alpha", fixture_contest)

    assert len(h["rounds"]) == 2
    assert len(h["hv_history"]) == 3  # initial + one per round
    hv = h["hv_history"]
    assert all(hv[i + 1] >= hv[i] - 1e-9 for i in range(len(hv) - 1))
    assert "auc_hv" in h and "final_hv" in h
    assert all(len(rd["ids"]) == config.BATCH_SIZE for rd in h["rounds"])
    assert len(set(h["selected_ids"])) == len(h["selected_ids"])
    assert not (set(h["selected_ids"]) & set(initial_ids))


def test_run_campaign_rejects_bad_plan_sum(fixture_contest):
    with pytest.raises(ValueError, match="sum"):
        _run([{"nehvi": 3}, {"nehvi": 4}], "BadSum", fixture_contest)


def test_run_campaign_rejects_wrong_round_count(fixture_contest):
    pool, oracle, initial_ids = fixture_contest
    with pytest.raises(ValueError, match="exactly 3 rounds"):
        competition.run_campaign(
            [{"nehvi": 4}, {"nehvi": 4}], "WrongLen",
            pool=pool, oracle=oracle, initial_ids=initial_ids, n_rounds=3,
        )


def _campaign(fixture_contest, name="Adaptive", **kw):
    pool, oracle, initial_ids = fixture_contest
    return competition.Campaign(
        name, pool=pool, oracle=oracle, initial_ids=initial_ids, **kw,
    )


def test_campaign_interactive_matches_run_campaign(fixture_contest):
    # Playing the same plans one-at-a-time (inspecting HV between) must reproduce the
    # one-shot open-loop history exactly -- that reproducibility is the whole contract.
    strategy = [{"nehvi": 4}, {"nehvi": 2, "parego": 2}, {"parego": 4}]
    batched = _run(strategy, "Twin", fixture_contest)

    c = _campaign(fixture_contest, name="Twin", n_rounds=3)
    for plan in strategy:
        c.play_round(plan, verbose=False)
        _ = c.current_hv  # inspecting between rounds must not perturb the RNG
    interactive = c.finalize()

    assert interactive == batched


def test_campaign_adapts_next_plan_from_observed_hv(fixture_contest):
    # The point of the class: choose round r+1 after seeing round r's hypervolume.
    c = _campaign(fixture_contest, n_rounds=3)
    assert c.round_index == 0 and c.rounds_left == 3
    hvs = [c.current_hv]
    for _ in range(3):
        plan = {"nehvi": 4} if c.current_hv < 0.99 else {"parego": 4}  # data-dependent choice
        c.play_round(plan, verbose=False)
        hvs.append(c.current_hv)
    assert c.round_index == 3 and c.rounds_left == 0
    assert all(hvs[i + 1] >= hvs[i] - 1e-9 for i in range(len(hvs) - 1))
    assert c.finalize()["n_rounds"] == 3


def test_campaign_plot_fronts_by_round(fixture_contest):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    c = _campaign(fixture_contest, n_rounds=2)
    c.play_round({"nehvi": 4}, verbose=False)
    c.play_round({"parego": 4}, verbose=False)
    ax = c.plot_fronts_by_round()
    labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert sum("after round" in lab for lab in labels) == 2  # one front per round played
    plt.close("all")


def test_campaign_play_round_rejects_bad_plan_sum(fixture_contest):
    c = _campaign(fixture_contest, n_rounds=2)
    with pytest.raises(ValueError, match="sum"):
        c.play_round({"nehvi": 3})


def test_campaign_play_beyond_budget_raises(fixture_contest):
    c = _campaign(fixture_contest, n_rounds=1)
    c.play_round({"nehvi": 4}, verbose=False)
    with pytest.raises(RuntimeError, match="already played"):
        c.play_round({"nehvi": 4})


def test_campaign_finalize_before_budget_raises(fixture_contest):
    c = _campaign(fixture_contest, n_rounds=2)
    c.play_round({"nehvi": 4}, verbose=False)
    with pytest.raises(RuntimeError, match="play 1 more"):
        c.finalize()


def test_oracle_blocks_true_objectives_during_run(fixture_contest):
    _, oracle, _ = fixture_contest
    with pytest.raises(PermissionError):
        oracle.evaluate_true([0])


def test_save_and_load_roundtrip(fixture_contest, tmp_path):
    h = _run([{"nehvi": 4}, {"random": 4}], "Round Trippers", fixture_contest)
    json_path = competition.save_run_outputs(h, output_dir=tmp_path)
    assert json_path.exists()
    assert (tmp_path / "round_trippers_history.csv").exists()
    assert (tmp_path / "round_trippers_pareto_plot.png").exists()
    assert (tmp_path / "round_trippers_hv_curve.png").exists()

    runs = competition.load_team_runs(tmp_path)
    assert len(runs) == 1
    assert runs[0]["team_name"] == "Round Trippers"
    assert runs[0]["selected_ids"] == h["selected_ids"]


def test_save_run_outputs_warns_on_slug_collision(fixture_contest, tmp_path):
    h1 = _run([{"nehvi": 4}], "Team A", fixture_contest)
    h2 = _run([{"random": 4}], "team a!", fixture_contest)  # -> same slug "team_a"
    competition.save_run_outputs(h1, output_dir=tmp_path)

    import warnings

    # Different team, same slug -> warn (about to clobber the earlier submission).
    with pytest.warns(UserWarning, match="slugifies"):
        competition.save_run_outputs(h2, output_dir=tmp_path)

    # Re-saving the *same* team must not warn ("latest wins" is intended).
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        competition.save_run_outputs(h2, output_dir=tmp_path)


def test_leaderboard_ranks_by_auc_hv(fixture_contest, tmp_path):
    h1 = _run([{"nehvi": 4}, {"nehvi": 4}], "Strong", fixture_contest)
    h2 = _run([{"random": 4}, {"random": 4}], "Weak", fixture_contest)
    competition.save_run_outputs(h1, output_dir=tmp_path)
    competition.save_run_outputs(h2, output_dir=tmp_path)

    board = competition.update_leaderboard(tmp_path)
    assert list(board["team_name"]) == sorted(
        ["Strong", "Weak"], key=lambda n: {"Strong": h1, "Weak": h2}[n]["auc_hv"], reverse=True
    )
    # AUC-HV must be sorted descending.
    assert board["auc_hv"].is_monotonic_decreasing


def test_build_final_debrief_report_writes_overlay(fixture_contest, tmp_path):
    pool, _, initial_ids = fixture_contest
    h = _run([{"nehvi": 4}, {"parego": 4}], "Revealed", fixture_contest)
    competition.save_run_outputs(h, output_dir=tmp_path)

    # Instructor reveal needs an allow_true oracle over the same true table.
    Y_true = torch.stack([pool.X[:, 0], 1.0 - 0.6 * pool.X[:, 0] + 0.4 * pool.X[:, 1]], dim=-1)
    reveal_oracle = AntibodyOracle(Y_true, allow_true=True)
    overlay = competition.build_final_debrief_report(tmp_path, reveal_oracle, initial_ids)
    assert overlay.exists()
    assert (tmp_path / "leaderboard.csv").exists()


def test_load_team_runs_empty_dir(tmp_path):
    assert competition.load_team_runs(tmp_path) == []
    assert competition.update_leaderboard(tmp_path).empty
