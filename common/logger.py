"""
Centralized per-run logging for the drug design pipeline.

Each run writes to logs/<run_id>.log. Propagate run_id via RUN_ID / LOG_DIR env
or --run_id / --run-id CLI flags in child scripts.
"""
from __future__ import annotations

import logging
import os
import subprocess
import sys
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional

_lock = threading.RLock()
_loggers: dict[str, logging.Logger] = {}


class RunLogFormatter(logging.Formatter):
    """2026-06-04 10:15:22 | run_id=... | process=... | module=... | INFO | message"""

    def format(self, record: logging.LogRecord) -> str:
        ts = datetime.fromtimestamp(record.created).strftime("%Y-%m-%d %H:%M:%S")
        run_id = getattr(record, "run_id", None) or os.environ.get("RUN_ID", "unknown")
        process_name = getattr(record, "process_name", None) or "unknown"
        module = record.module
        level = record.levelname
        msg = record.getMessage()
        if record.exc_info and not record.exc_text:
            record.exc_text = self.formatException(record.exc_info)
        if record.exc_text:
            msg = f"{msg}\n{record.exc_text}"
        return f"{ts} | run_id={run_id} | process={process_name} | module={module} | {level} | {msg}"


class ProcessSafeFileHandler(logging.Handler):
    """Thread-safe handler; uses flock on POSIX for multi-process append."""

    def __init__(self, log_path: Path) -> None:
        super().__init__()
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._io_lock = threading.RLock()

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record) + "\n"
            with self._io_lock:
                try:
                    import fcntl

                    with open(self.log_path, "a", encoding="utf-8") as f:
                        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                        try:
                            f.write(msg)
                            f.flush()
                        finally:
                            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
                except (ImportError, AttributeError, OSError):
                    with open(self.log_path, "a", encoding="utf-8") as f:
                        f.write(msg)
                        f.flush()
        except Exception:
            self.handleError(record)


def get_log_dir() -> Path:
    """Resolve logs directory (LOG_DIR env, /workspace/logs in container, else ./logs)."""
    env_dir = os.environ.get("LOG_DIR")
    if env_dir:
        return Path(env_dir)
    if Path("/workspace").is_dir():
        return Path("/workspace/logs")
    return Path.cwd() / "logs"


def get_log_path(run_id: str) -> Path:
    return get_log_dir() / f"{run_id}.log"


def init_run_logging(run_id: str, process_name: str) -> logging.Logger:
    """Create or return the shared file logger for this run_id (one file per run)."""
    os.environ["RUN_ID"] = run_id
    os.environ["LOG_DIR"] = str(get_log_dir())
    log_path = get_log_path(run_id)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    cache_key = run_id
    with _lock:
        if cache_key in _loggers:
            return _loggers[cache_key]

        base_logger = logging.getLogger(f"run.{run_id}")
        base_logger.setLevel(logging.DEBUG)
        base_logger.propagate = False

        if not base_logger.handlers:
            handler = ProcessSafeFileHandler(log_path)
            handler.setFormatter(RunLogFormatter())
            base_logger.addHandler(handler)
            if not log_path.exists():
                log_path.touch()

        _loggers[cache_key] = base_logger
        return base_logger


def get_logger(
    run_id: Optional[str] = None,
    process_name: Optional[str] = None,
) -> logging.LoggerAdapter:
    """
    Return a LoggerAdapter that injects run_id and process_name into every record.
    run_id: explicit id or RUN_ID env (required).
    process_name: explicit name or basename of sys.argv[0].
    """
    rid = (run_id or os.environ.get("RUN_ID") or "").strip()
    if not rid:
        raise ValueError("run_id is required (argument or RUN_ID environment variable)")
    pname = process_name or Path(sys.argv[0]).name or "unknown"
    base = init_run_logging(rid, pname)
    return logging.LoggerAdapter(
        base,
        {"run_id": rid, "process_name": pname},
    )


def run_env_for_child(run_id: str) -> dict[str, str]:
    """Environment dict for subprocess children (propagates RUN_ID and LOG_DIR)."""
    env = os.environ.copy()
    env["RUN_ID"] = run_id
    env["LOG_DIR"] = str(get_log_dir())
    return env


def log_subprocess_result(
    logger: logging.LoggerAdapter,
    result: subprocess.CompletedProcess,
    cmd: list[str],
    *,
    label: str = "subprocess",
    elapsed_sec: Optional[float] = None,
) -> None:
    """Log command, return code, stdout and stderr lines into the run log file."""
    logger.info("Command [%s]: %s", label, " ".join(cmd))
    if elapsed_sec is not None:
        logger.info("Elapsed [%s]: %.2f seconds", label, elapsed_sec)
    logger.info("Return code [%s]: %s", label, result.returncode)
    stdout = result.stdout or ""
    stderr = result.stderr or ""
    if stdout.strip():
        for line in stdout.splitlines():
            logger.info("[%s stdout] %s", label, line)
    if stderr.strip():
        for line in stderr.splitlines():
            logger.error("[%s stderr] %s", label, line)


def resolve_run_id(explicit: Optional[str] = None) -> str:
    """Use explicit run_id, RUN_ID env, or generate a new UUID hex."""
    import uuid

    rid = (explicit or os.environ.get("RUN_ID") or "").strip()
    if rid:
        return rid
    return uuid.uuid4().hex


def read_log_content(
    run_id: str,
    *,
    tail: Optional[int] = None,
    offset: int = 0,
) -> dict:
    """
    Read centralized log file for run_id.
    Returns dict with content, metadata; raises FileNotFoundError if missing.
    """
    log_path = get_log_path(run_id)
    if not log_path.is_file():
        raise FileNotFoundError(f"Log file not found for run_id '{run_id}': {log_path}")

    text = log_path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    total_lines = len(lines)

    if offset > 0:
        lines = lines[offset:]
    if tail is not None and tail > 0:
        lines = lines[-tail:]

    content = "\n".join(lines)
    if lines and text.endswith("\n"):
        content += "\n"

    stat = log_path.stat()
    return {
        "run_id": run_id,
        "log_file": str(log_path),
        "content": content,
        "total_lines": total_lines,
        "returned_lines": len(lines),
        "offset": offset,
        "tail": tail,
        "size_bytes": stat.st_size,
        "modified_at": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
    }


def list_run_logs() -> list[dict]:
    """List available run log files in the logs directory."""
    log_dir = get_log_dir()
    if not log_dir.is_dir():
        return []
    entries = []
    for path in sorted(log_dir.glob("*.log"), key=lambda p: p.stat().st_mtime, reverse=True):
        stat = path.stat()
        entries.append({
            "run_id": path.stem,
            "log_file": str(path),
            "size_bytes": stat.st_size,
            "modified_at": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
        })
    return entries
