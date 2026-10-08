"""Lightweight host metrics for GET /metrics/system.

Sampling is non-blocking: cpu_percent(interval=None) returns the last
computed value (or 0.0 on the first call) and does not sleep.
"""
from __future__ import annotations

import os
import socket
import sqlite3
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import psutil

DISK_PATH = Path(os.environ.get("METRICS_DISK_PATH", "/workspace/outputs"))

_net_lock = threading.Lock()
_net_prev: tuple[float, int, int] | None = None


def reset_network_sample() -> None:
    """Forget the previous NIC sample (tests and process restart)."""
    global _net_prev
    with _net_lock:
        _net_prev = None


def _cpu() -> dict:
    percent = float(psutil.cpu_percent(interval=None))
    count = psutil.cpu_count(logical=True) or 0
    load_avg_1m = 0.0
    try:
        load_avg_1m = float(os.getloadavg()[0])
    except (AttributeError, OSError):
        getloadavg = getattr(psutil, "getloadavg", None)
        if getloadavg is not None:
            try:
                load_avg_1m = float(getloadavg()[0])
            except (OSError, AttributeError):
                load_avg_1m = 0.0
    return {"percent": percent, "count": int(count), "load_avg_1m": load_avg_1m}


def _memory() -> dict:
    vm = psutil.virtual_memory()
    return {
        "total_mb": round(vm.total / (1024 * 1024), 2),
        "used_mb": round(vm.used / (1024 * 1024), 2),
        "percent": float(vm.percent),
    }


def _gpus() -> list[dict]:
    try:
        import pynvml
    except ImportError:
        return []
    try:
        pynvml.nvmlInit()
    except Exception:
        return []
    gpus: list[dict] = []
    try:
        count = int(pynvml.nvmlDeviceGetCount())
        for index in range(count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(index)
            name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode("utf-8", errors="replace")
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            try:
                temperature = int(
                    pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                )
            except Exception:
                temperature = None
            mem_total = int(mem.total)
            mem_used = int(mem.used)
            mem_percent = (mem_used / mem_total * 100.0) if mem_total else 0.0
            gpus.append(
                {
                    "index": index,
                    "name": str(name),
                    "util_percent": float(util.gpu),
                    "mem_used_mb": round(mem_used / (1024 * 1024), 2),
                    "mem_total_mb": round(mem_total / (1024 * 1024), 2),
                    "mem_percent": round(mem_percent, 2),
                    "temperature_c": temperature,
                }
            )
    except Exception:
        return []
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass
    return gpus


def _disk(path: Path) -> dict:
    usage = psutil.disk_usage(str(path))
    return {
        "total_gb": round(usage.total / (1024 ** 3), 2),
        "used_gb": round(usage.used / (1024 ** 3), 2),
        "percent": float(usage.percent),
    }


def _network() -> dict:
    global _net_prev
    counters = psutil.net_io_counters()
    now = time.monotonic()
    sent = int(counters.bytes_sent)
    recv = int(counters.bytes_recv)
    with _net_lock:
        if _net_prev is None:
            sent_per_sec = 0.0
            recv_per_sec = 0.0
        else:
            prev_t, prev_sent, prev_recv = _net_prev
            elapsed = now - prev_t
            if elapsed <= 0:
                sent_per_sec = 0.0
                recv_per_sec = 0.0
            else:
                sent_per_sec = max(0.0, (sent - prev_sent) / elapsed)
                recv_per_sec = max(0.0, (recv - prev_recv) / elapsed)
        _net_prev = (now, sent, recv)
    return {
        "bytes_sent": sent,
        "bytes_recv": recv,
        "sent_per_sec": sent_per_sec,
        "recv_per_sec": recv_per_sec,
    }


def _running_runs(db_path: Path) -> dict:
    running_run_ids: list[str] = []
    if db_path.is_file():
        try:
            with sqlite3.connect(str(db_path)) as conn:
                rows = conn.execute(
                    "SELECT DISTINCT run_id FROM run_status WHERE status = ? ORDER BY run_id",
                    ("RUNNING",),
                ).fetchall()
            running_run_ids = [row[0] for row in rows]
        except sqlite3.Error:
            running_run_ids = []
    return {"running_count": len(running_run_ids), "running_run_ids": running_run_ids}


def collect_system_metrics(
    db_path: Path | None = None,
    disk_path: Path | None = None,
) -> dict:
    """Snapshot of CPU, memory, GPUs, disk, network and in-flight runs."""
    database = db_path if db_path is not None else Path("/workspace/outputs/run_status.db")
    disk = disk_path if disk_path is not None else DISK_PATH
    try:
        disk_info = _disk(disk)
    except (FileNotFoundError, OSError):
        disk_info = {"total_gb": 0.0, "used_gb": 0.0, "percent": 0.0}
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "cpu": _cpu(),
        "memory": _memory(),
        "gpus": _gpus(),
        "disk": disk_info,
        "network": _network(),
        "runs": _running_runs(database),
    }
