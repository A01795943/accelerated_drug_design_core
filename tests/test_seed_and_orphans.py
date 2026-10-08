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


def test_diverse_seed_keeps_historical_formula_when_unset():
    assert diverse_mpnn_seed(None, 0) == 42
    assert diverse_mpnn_seed(None, 2) == 2 * 12345 + 42


def test_diverse_seed_shifts_with_base_so_minibatches_do_not_overlap():
    first = [diverse_mpnn_seed(1000, n) for n in range(3)]
    second = [diverse_mpnn_seed(2000, n) for n in range(3)]
    assert first == [1000, 1001, 1002]
    assert set(first).isdisjoint(second)


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
