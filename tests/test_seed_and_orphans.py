"""MPNN seed offsets and orphaned run_status recovery."""
import sqlite3
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "api"))
sys.path.insert(0, str(ROOT / "pipeline"))

from mpnn_seed import diverse_mpnn_seed  # noqa: E402
import api as api_mod  # noqa: E402


def test_binder_key_is_the_tail_after_the_last_slash():
    from mpnn_seed import binder_key, binder_key_for_length

    target = "ACDEFGHIKLMNPQRSTVWY" * 3 + "ACDEFGHIKLMN"  # 72
    assert len(target) == 72
    binder_a = "CCCCCCCCCCCCCC"
    binder_b = "DDDDDDDDDDDDDD"
    assert binder_key(f"{target}/{binder_a}") == binder_a
    assert binder_key(f"{target}/{binder_b}") != binder_a
    assert binder_key(f"{'B' * 72}/{binder_a}") == binder_a
    assert binder_key(f" \n{target.lower()}/{binder_a.lower()} ") == binder_a
    assert binder_key(binder_a.lower()) == binder_a
    short = "C" * 10
    folded = binder_key_for_length(f"{target}/{short}", 14)
    flat = target + short
    assert folded == flat[-14:]
    assert binder_key_for_length(f"{target}/{binder_a}", 14) == binder_a
    assert binder_key(None) == ""


def test_diverse_seed_is_31_bits_and_deterministic():
    seeds = [diverse_mpnn_seed(527, attempt) for attempt in range(100_000)]
    assert min(seeds) >= 1
    assert max(seeds) <= 2**31 - 1
    assert diverse_mpnn_seed(None, 0) == diverse_mpnn_seed(42, 0)
    assert diverse_mpnn_seed(527, 3) == diverse_mpnn_seed(527, 3)
    assert diverse_mpnn_seed(527, 0) != diverse_mpnn_seed(527, 1)


def test_diverse_seed_shifts_with_base_so_minibatches_do_not_overlap():
    from mpnn_seed import binder_key

    target = "A" * 72
    binder_a = "CCCCCCCCCCCCCC"
    binder_b = "DDDDDDDDDDDDDD"
    assert len(binder_a) == 14 and len(binder_b) == 14
    assert binder_key(f"{target}/{binder_a}") == binder_a
    assert binder_key(f"{target}/{binder_a}") != binder_key(f"{target}/{binder_b}")
    assert binder_key(f"{target}/{binder_a}") == binder_key(f"{'B' * 72}/{binder_a}")
    assert binder_key(f"  {target.lower()} / {binder_a.lower()} \n") == binder_a
    assert binder_key("  acde  ") == "ACDE"
    assert binder_key(None) == ""


def _init_db(tmp_path, monkeypatch):
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    db = outputs / "run_status.db"
    monkeypatch.setattr(api_mod, "OUTPUTS", outputs)
    monkeypatch.setattr(api_mod, "RUN_STATUS_DB", db)
    api_mod.init_run_status_db()
    return db


def test_minibatch_run_ids_do_not_collide_on_composite_key(tmp_path, monkeypatch):
    """PK is (run_id, task). '{runId}_m{index}' is a different run_id, so it inserts."""
    _init_db(tmp_path, monkeypatch)
    parent = "20261008_1234"
    task = api_mod.TASK_MPNN_RF_DIFFUSION
    api_mod.run_status_insert(parent, task, api_mod.STATUS_RUNNING)
    api_mod.run_status_insert(f"{parent}_m0", task, api_mod.STATUS_RUNNING)
    api_mod.run_status_insert(f"{parent}_m1", task, api_mod.STATUS_RUNNING)
    api_mod.run_status_insert(parent, api_mod.TASK_RD_DIFFUSION, api_mod.STATUS_RUNNING)
    with pytest.raises(sqlite3.IntegrityError):
        api_mod.run_status_insert(f"{parent}_m0", task, api_mod.STATUS_RUNNING)


def test_startup_marks_dead_workers_and_keeps_live_ones(tmp_path, monkeypatch):
    db = _init_db(tmp_path, monkeypatch)
    task = api_mod.TASK_MPNN_RF_DIFFUSION
    api_mod.run_status_insert("dead_m0", task, api_mod.STATUS_RUNNING)
    api_mod.run_status_insert("live_m1", task, api_mod.STATUS_RUNNING)
    api_mod.run_status_set_worker_pid("dead_m0", task, 2**31 - 1)
    api_mod.run_status_set_worker_pid("live_m1", task, 4242)

    def alive(pid, run_id):
        return run_id == "live_m1" and pid == 4242

    monkeypatch.setattr(api_mod, "_worker_still_running", alive)
    marked = api_mod.mark_orphaned_runs(db)
    assert marked == ["dead_m0"]

    with sqlite3.connect(str(db)) as conn:
        rows = {
            row[0]: (row[1], row[2])
            for row in conn.execute("SELECT run_id, status, error_details FROM run_status")
        }
    assert rows["dead_m0"] == (api_mod.STATUS_ERROR, "orphaned: core restarted")
    assert rows["live_m1"][0] == api_mod.STATUS_RUNNING
