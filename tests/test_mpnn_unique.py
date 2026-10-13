"""Unique-sequence loop without ColabDesign."""
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "api"))
sys.path.insert(0, str(ROOT / "pipeline"))

from mpnn_unique import merge_run_stats, run_unique_design  # noqa: E402
import api as api_mod  # noqa: E402

AF_TERMS = ["plddt", "ptm", "i_ptm", "pae", "i_pae", "rmsd"]


def _seq(binder: str) -> str:
    return "A" * 72 + "/" + binder


class _Model:
    def __init__(self, seq: str):
        self.seq = seq

    def get_af_inputs(self, af_model):
        assert af_model.predicted == []

    def sample(self, num, batch, temperature):
        assert num == 1 and batch == 1
        return {"seq": [self.seq], "score": [0.5]}


class _Factory:
    def __init__(self, sequences):
        self.sequences = list(sequences)
        self.seeds = []

    def __call__(self, seed=None):
        self.seeds.append(seed)
        return _Model(self.sequences[len(self.seeds) - 1])


class _Af:
    def __init__(self):
        self._len = 14
        self._k = 0
        self.aux = {"log": {}}
        self.predicted = []
        self.saved = []

    def predict(self, seq, num_recycles, verbose):
        self.predicted.append(seq)
        self.aux = {
            "log": {
                "plddt": 0.8,
                "ptm": 0.7,
                "i_ptm": 0.6,
                "pae": 0.2,
                "i_pae": 0.1,
                "rmsd": 2.0 + len(self.predicted),
            }
        }

    def save_current_pdb(self, path):
        Path(path).write_text("ATOM", encoding="utf-8")
        self.saved.append(path)

    def _save_results(self, save_best, verbose):
        return None


def _run(tmp_path, sequences, num_seqs, max_attempts, known=None):
    af_model = _Af()
    factory = _Factory(sequences)
    stats = run_unique_design(
        loc=str(tmp_path),
        af_model=af_model,
        mk_mpnn_model=factory,
        num_seqs=num_seqs,
        max_attempts=max_attempts,
        temperature=0.4,
        base_seed=527,
        num_recycles=1,
        af_terms=AF_TERMS,
        known_keys=set(known or []),
    )
    return stats, af_model, factory


def _csv_rows(tmp_path):
    with open(tmp_path / "mpnn_results.csv", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_loop_stops_at_n_unique(tmp_path):
    sequences = [_seq("C" * 14), _seq("D" * 14), _seq("E" * 14), _seq("F" * 14)]
    stats, af_model, factory = _run(tmp_path, sequences, num_seqs=3, max_attempts=10)
    assert stats["accepted"] == 3
    assert stats["capped"] is False
    assert stats["attempts_used"] == 3
    assert len(af_model.predicted) == 3
    assert af_model.predicted[0] == "C" * 14
    assert [row["n"] for row in _csv_rows(tmp_path)] == ["0", "1", "2"]
    assert (tmp_path / "all_pdb" / "design0_n0.pdb").is_file()
    assert (tmp_path / "all_pdb" / "design0_n2.pdb").is_file()
    assert not (tmp_path / "all_pdb" / "design0_n3.pdb").exists()
    assert (tmp_path / "best.pdb").is_file()
    assert len(factory.seeds) == 3
    progress = json.loads((tmp_path / "progress.json").read_text(encoding="utf-8"))
    assert progress["phase"] == "alphafold"
    assert progress["accepted"] == 3


def test_loop_stops_at_the_cap(tmp_path):
    sequences = [_seq("C" * 14)] * 6
    stats, af_model, _factory = _run(tmp_path, sequences, num_seqs=4, max_attempts=5)
    assert stats["accepted"] == 1
    assert stats["attempts_used"] == 5
    assert stats["capped"] is True
    assert stats["skipped_batch_duplicates"] == 4
    assert len(af_model.predicted) == 1
    assert [row["n"] for row in _csv_rows(tmp_path)] == ["0"]


def test_db_and_batch_duplicates_are_counted_apart(tmp_path):
    known = "C" * 14
    sequences = [_seq(known), _seq("D" * 14), _seq(known), _seq("D" * 14), _seq("E" * 14)]
    stats, af_model, _factory = _run(tmp_path, sequences, num_seqs=2, max_attempts=10, known={known})
    assert stats["skipped_db_duplicates"] == 2
    assert stats["skipped_batch_duplicates"] == 1
    assert stats["accepted"] == 2
    assert stats["capped"] is False
    assert af_model.predicted == ["D" * 14, "E" * 14]
    assert [row["n"] for row in _csv_rows(tmp_path)] == ["0", "1"]


def test_zero_accepted_still_finishes(tmp_path):
    known = "C" * 14
    stats, af_model, _factory = _run(
        tmp_path, [_seq(known)] * 3, num_seqs=2, max_attempts=3, known={known}
    )
    assert stats["accepted"] == 0
    assert stats["capped"] is True
    assert stats["skipped_db_duplicates"] == 3
    assert af_model.predicted == []
    assert not (tmp_path / "best.pdb").exists()
    assert _csv_rows(tmp_path) == []
    saved = json.loads((tmp_path / "run_stats.json").read_text(encoding="utf-8"))
    assert saved["accepted"] == 0


def test_run_stats_merge_into_param_details(tmp_path):
    (tmp_path / "run_stats.json").write_text(
        json.dumps({"attempts_used": 12, "capped": True, "accepted": 4}),
        encoding="utf-8",
    )
    merged = merge_run_stats({"num_seqs": 8}, str(tmp_path))
    assert merged["num_seqs"] == 8
    assert merged["attempts_used"] == 12
    assert merged["capped"] is True
    assert merge_run_stats({"num_seqs": 8}, str(tmp_path / "missing")) == {"num_seqs": 8}


def test_status_returns_progress_while_running(tmp_path, monkeypatch):
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    monkeypatch.setattr(api_mod, "OUTPUTS", outputs)
    monkeypatch.setattr(api_mod, "RUN_STATUS_DB", outputs / "run_status.db")
    api_mod.init_run_status_db()
    api_mod.run_status_insert("job_m0", api_mod.TASK_MPNN_RF_DIFFUSION, api_mod.STATUS_RUNNING)
    folder = outputs / "job_m0"
    folder.mkdir()
    progress = {"phase": "sampling", "attempts": 25, "accepted": 3, "target": 100}
    (folder / "progress.json").write_text(json.dumps(progress), encoding="utf-8")
    body = api_mod.mpnn_status("job_m0")
    assert body["progress"] == progress


def test_child_log_streams_and_keeps_the_tail(tmp_path):
    from common.logger import stream_subprocess

    log_path = tmp_path / "child.log"
    code = "import sys\nprint('line-0')\nprint('line-1')\nsys.exit(3)\n"
    result = stream_subprocess([sys.executable, "-c", code], str(log_path), cwd=str(tmp_path))
    assert result.returncode == 3
    text = log_path.read_text(encoding="utf-8")
    assert "line-0" in text and "line-1" in text
    assert "line-1" in result.stdout
