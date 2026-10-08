"""Shape of GET /metrics/system with NVML mocked."""
import sqlite3
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "api"))

import system_metrics  # noqa: E402


class _Util:
    def __init__(self, gpu: int):
        self.gpu = gpu


class _Mem:
    def __init__(self, used: int, total: int):
        self.used = used
        self.total = total


def _install_fake_nvml(monkeypatch):
    fake = types.ModuleType("pynvml")
    fake.NVML_TEMPERATURE_GPU = 0
    fake.calls = {"init": 0, "shutdown": 0}

    def nvmlInit():
        fake.calls["init"] += 1

    def nvmlShutdown():
        fake.calls["shutdown"] += 1

    def nvmlDeviceGetCount():
        return 1

    def nvmlDeviceGetHandleByIndex(index):
        return index

    def nvmlDeviceGetName(handle):
        return "Fake GPU"

    def nvmlDeviceGetUtilizationRates(handle):
        return _Util(42)

    def nvmlDeviceGetMemoryInfo(handle):
        return _Mem(used=512 * 1024 * 1024, total=1024 * 1024 * 1024)

    def nvmlDeviceGetTemperature(handle, sensor):
        return 61

    fake.nvmlInit = nvmlInit
    fake.nvmlShutdown = nvmlShutdown
    fake.nvmlDeviceGetCount = nvmlDeviceGetCount
    fake.nvmlDeviceGetHandleByIndex = nvmlDeviceGetHandleByIndex
    fake.nvmlDeviceGetName = nvmlDeviceGetName
    fake.nvmlDeviceGetUtilizationRates = nvmlDeviceGetUtilizationRates
    fake.nvmlDeviceGetMemoryInfo = nvmlDeviceGetMemoryInfo
    fake.nvmlDeviceGetTemperature = nvmlDeviceGetTemperature
    monkeypatch.setitem(sys.modules, "pynvml", fake)
    return fake


def _write_run_db(path: Path) -> None:
    with sqlite3.connect(str(path)) as conn:
        conn.execute(
            """
            CREATE TABLE run_status (
                run_id TEXT NOT NULL,
                task TEXT NOT NULL,
                status TEXT NOT NULL,
                PRIMARY KEY (run_id, task)
            )
            """
        )
        conn.executemany(
            "INSERT INTO run_status (run_id, task, status) VALUES (?, ?, ?)",
            [
                ("job_m0", "MPNN+RF_DIFFUSION", "RUNNING"),
                ("job_m0", "RD_DIFFUSION", "COMPLETED"),
                ("job_m1", "MPNN+RF_DIFFUSION", "RUNNING"),
                ("done", "MPNN+RF_DIFFUSION", "COMPLETED"),
            ],
        )


def test_metrics_shape_with_mocked_nvml(tmp_path, monkeypatch):
    fake = _install_fake_nvml(monkeypatch)
    system_metrics.reset_network_sample()
    db = tmp_path / "run_status.db"
    _write_run_db(db)
    disk = tmp_path / "outputs"
    disk.mkdir()

    payload = system_metrics.collect_system_metrics(db_path=db, disk_path=disk)

    assert set(payload) == {
        "timestamp",
        "hostname",
        "cpu",
        "memory",
        "gpus",
        "disk",
        "network",
        "runs",
    }
    assert payload["timestamp"].endswith("+00:00") or payload["timestamp"].endswith("Z")
    assert isinstance(payload["hostname"], str) and payload["hostname"]
    assert set(payload["cpu"]) == {"percent", "count", "load_avg_1m"}
    assert isinstance(payload["cpu"]["percent"], float)
    assert isinstance(payload["cpu"]["count"], int)
    assert set(payload["memory"]) == {"total_mb", "used_mb", "percent"}
    assert payload["gpus"] == [
        {
            "index": 0,
            "name": "Fake GPU",
            "util_percent": 42.0,
            "mem_used_mb": 512.0,
            "mem_total_mb": 1024.0,
            "mem_percent": 50.0,
            "temperature_c": 61,
        }
    ]
    assert set(payload["disk"]) == {"total_gb", "used_gb", "percent"}
    assert set(payload["network"]) == {
        "bytes_sent",
        "bytes_recv",
        "sent_per_sec",
        "recv_per_sec",
    }
    assert payload["network"]["sent_per_sec"] == 0.0
    assert payload["network"]["recv_per_sec"] == 0.0
    assert payload["runs"]["running_count"] == 2
    assert payload["runs"]["running_run_ids"] == ["job_m0", "job_m1"]
    assert fake.calls["shutdown"] == 1


def test_nvml_unavailable_returns_empty_gpus(tmp_path, monkeypatch):
    fake = types.ModuleType("pynvml")

    def nvmlInit():
        raise RuntimeError("no driver")

    fake.nvmlInit = nvmlInit
    monkeypatch.setitem(sys.modules, "pynvml", fake)
    system_metrics.reset_network_sample()
    disk = tmp_path / "outputs"
    disk.mkdir()

    payload = system_metrics.collect_system_metrics(db_path=tmp_path / "missing.db", disk_path=disk)
    assert payload["gpus"] == []
    assert payload["runs"] == {"running_count": 0, "running_run_ids": []}


def test_endpoint_returns_same_shape(tmp_path, monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    _install_fake_nvml(monkeypatch)
    import api as api_mod

    outputs = tmp_path / "outputs"
    outputs.mkdir()
    db = outputs / "run_status.db"
    _write_run_db(db)
    monkeypatch.setattr(api_mod, "OUTPUTS", outputs)
    monkeypatch.setattr(api_mod, "get_run_status_db_path", lambda: db)
    system_metrics.reset_network_sample()

    client = TestClient(api_mod.app)
    response = client.get("/metrics/system")
    assert response.status_code == 200
    body = response.json()
    assert body["gpus"][0]["name"] == "Fake GPU"
    assert body["runs"]["running_count"] == 2
    assert body["network"]["sent_per_sec"] == 0.0
