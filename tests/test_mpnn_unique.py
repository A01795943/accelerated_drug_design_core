"""Unique-sequence loop without ColabDesign."""
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "api"))
sys.path.insert(0, str(ROOT / "pipeline"))

from mpnn_seed import diverse_mpnn_seed  # noqa: E402
from mpnn_unique import merge_run_stats, run_unique_design  # noqa: E402
import api as api_mod  # noqa: E402

AF_TERMS = ["plddt", "ptm", "i_ptm", "pae", "i_pae", "rmsd"]


def _seq(binder: str) -> str:
    return "A" * 72 + "/" + binder


class _QueueModel:
    def __init__(self, sequences):
        self.sequences = list(sequences)
        self.cursor = 0
        self.seeds = []
        self.af_calls = 0
        self.batches = []

    def set_seed(self, seed):
        self.seeds.append(seed)

    def get_af_inputs(self, af_model):
        self.af_calls += 1
        assert af_model.predicted == []

    def sample(self, num, batch, temperature):
        assert num == 1
        chunk = self.sequences[self.cursor:self.cursor + batch]
        assert len(chunk) == batch
        self.cursor += batch
        self.batches.append(batch)
        return {"seq": chunk, "score": [0.5] * batch}


class _Factory:
    def __init__(self, sequences):
        self.model = _QueueModel(sequences)
        self.calls = []

    def __call__(self, seed=None):
        self.calls.append(seed)
        return self.model


class _Log:
    def __init__(self):
        self.infos = []
        self.warnings = []

    def info(self, message, *args):
        self.infos.append(message % args if args else message)

    def warning(self, message, *args):
        self.warnings.append(message % args if args else message)


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


def _run(tmp_path, sequences, num_seqs, max_attempts, known=None, round_size=None, logger=None):
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
        round_size=round_size,
        logger=logger,
    )
    return stats, af_model, factory


def _assert_one_model(factory):
    assert factory.calls == [diverse_mpnn_seed(527, 0)]
    assert factory.model.af_calls == 1
    assert factory.model.seeds == [diverse_mpnn_seed(527, index) for index in range(len(factory.model.seeds))]


def _csv_rows(tmp_path):
    with open(tmp_path / "mpnn_results.csv", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_one_round_fills_the_request(tmp_path):
    sequences = [_seq("C" * 14), _seq("D" * 14), _seq("E" * 14), _seq("F" * 14)]
    log = _Log()
    stats, af_model, factory = _run(tmp_path, sequences, num_seqs=3, max_attempts=10, logger=log)
    _assert_one_model(factory)
    assert factory.model.batches == [3]
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
    assert log.warnings == []
    assert len(log.infos) == 2
    progress = json.loads((tmp_path / "progress.json").read_text(encoding="utf-8"))
    assert progress["phase"] == "alphafold"
    assert progress["accepted"] == 3
    assert set(progress) == {"phase", "attempts", "accepted", "target"}


def test_several_rounds_use_one_model(tmp_path, monkeypatch):
    import mpnn_unique

    sequences = [_seq(letter * 14) for letter in "CDEF"]
    recorded = []
    original = mpnn_unique.write_progress

    def capture(loc, phase, attempts, accepted, target):
        recorded.append((phase, attempts, accepted, target))
        original(loc, phase, attempts, accepted, target)

    monkeypatch.setattr(mpnn_unique, "write_progress", capture)
    stats, af_model, factory = _run(
        tmp_path, sequences, num_seqs=4, max_attempts=10, round_size=2
    )
    _assert_one_model(factory)
    assert factory.model.batches == [2, 2]
    assert factory.model.seeds == [diverse_mpnn_seed(527, 0), diverse_mpnn_seed(527, 1)]
    assert stats["accepted"] == 4
    assert stats["capped"] is False
    assert stats["attempts_used"] == 4
    assert af_model.predicted == ["C" * 14, "D" * 14, "E" * 14, "F" * 14]
    sampling = [row for row in recorded if row[0] == "sampling"]
    assert ("sampling", 2, 2, 4) in sampling
    assert ("sampling", 4, 4, 4) in sampling


def test_budget_exhaustion_caps_without_raising(tmp_path):
    log = _Log()
    stats, af_model, factory = _run(
        tmp_path,
        [_seq("C" * 14)] * 6,
        num_seqs=4,
        max_attempts=5,
        round_size=4,
        logger=log,
    )
    _assert_one_model(factory)
    assert factory.model.batches == [4, 1]
    assert stats["accepted"] == 1
    assert stats["attempts_used"] == 5
    assert stats["capped"] is True
    assert stats["skipped_batch_duplicates"] == 4
    assert len(af_model.predicted) == 1
    assert [row["n"] for row in _csv_rows(tmp_path)] == ["0"]
    assert len(log.warnings) == 1
    assert "missing=3" in log.warnings[0]


def test_duplicates_inside_one_batch_keep_output_order(tmp_path):
    sequences = [_seq("C" * 14), _seq("D" * 14), _seq("C" * 14), _seq("E" * 14)]
    stats, af_model, factory = _run(
        tmp_path, sequences, num_seqs=3, max_attempts=10, round_size=4
    )
    _assert_one_model(factory)
    assert factory.model.batches == [4]
    assert stats["accepted"] == 3
    assert stats["skipped_batch_duplicates"] == 1
    assert stats["skipped_db_duplicates"] == 0
    assert stats["attempts_used"] == 4
    assert stats["capped"] is False
    assert af_model.predicted == ["C" * 14, "D" * 14, "E" * 14]


def test_exclusion_file_rejects_before_batch_duplicates(tmp_path):
    known = "C" * 14
    sequences = [_seq(known), _seq("D" * 14), _seq(known), _seq("E" * 14)]
    stats, af_model, factory = _run(
        tmp_path, sequences, num_seqs=2, max_attempts=10, known={known}, round_size=4
    )
    _assert_one_model(factory)
    assert stats["skipped_db_duplicates"] == 2
    assert stats["skipped_batch_duplicates"] == 0
    assert stats["accepted"] == 2
    assert stats["capped"] is False
    assert af_model.predicted == ["D" * 14, "E" * 14]
    assert [row["n"] for row in _csv_rows(tmp_path)] == ["0", "1"]


def test_zero_accepted_still_finishes(tmp_path):
    known = "C" * 14
    log = _Log()
    stats, af_model, factory = _run(
        tmp_path,
        [_seq(known)] * 3,
        num_seqs=2,
        max_attempts=3,
        known={known},
        round_size=3,
        logger=log,
    )
    _assert_one_model(factory)
    assert stats["accepted"] == 0
    assert stats["capped"] is True
    assert stats["skipped_db_duplicates"] == 3
    assert stats["attempts_used"] == 3
    assert af_model.predicted == []
    assert not (tmp_path / "best.pdb").exists()
    assert _csv_rows(tmp_path) == []
    saved = json.loads((tmp_path / "run_stats.json").read_text(encoding="utf-8"))
    assert saved["accepted"] == 0
    assert len(log.warnings) == 1
    assert "missing=2" in log.warnings[0]


def test_round_size_changes_which_seed_draws_each_batch(tmp_path):
    sequences = [_seq(letter * 14) for letter in "CDEFGH"]
    _, _, small = _run(tmp_path / "small", sequences, num_seqs=3, max_attempts=10, round_size=1)
    _, _, wide = _run(tmp_path / "wide", sequences, num_seqs=3, max_attempts=10, round_size=2)
    _assert_one_model(small)
    _assert_one_model(wide)
    assert small.model.batches == [1, 1, 1]
    assert wide.model.batches == [2, 2]
    assert small.model.seeds != wide.model.seeds


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
