#!/usr/bin/env python3
"""Resolve Ascend device log files with deterministic precedence."""

from __future__ import annotations

import glob
import os
import re
import stat
from collections.abc import Iterable, Iterator
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple


def get_log_root() -> Path:
    """Return log root: ASCEND_WORK_PATH first, then ~/ascend fallback."""
    ascend_work_path = os.environ.get("ASCEND_WORK_PATH")
    if ascend_work_path:
        env_root = Path(ascend_work_path).expanduser() / "log" / "debug"
        if env_root.is_dir():
            return env_root
    return Path.home() / "ascend" / "log" / "debug"


def infer_device_id_from_log_path(log_path: Path) -> Optional[str]:
    """Infer device id from any path segment like device-0."""
    for part in log_path.parts:
        match = re.fullmatch(r"device-(\d+)", part)
        if match:
            return match.group(1)
    return None


def _log_candidates(paths: Iterable[Path]) -> Iterator[tuple[float, Path]]:
    for path in paths:
        try:
            metadata = path.stat()
        except FileNotFoundError:
            continue
        if stat.S_ISREG(metadata.st_mode):
            yield metadata.st_mtime, path


def _latest_log(paths: Iterable[Path]) -> Optional[Path]:
    _, path = max(_log_candidates(paths), key=lambda item: (item[0], str(item[1])), default=(0, None))
    return path


def _extract_perf_timestamp(perf_path: Optional[Path]) -> Optional[datetime]:
    if perf_path is None:
        return None

    filename_match = re.search(r"perf_swimlane_(\d{8}_\d{6})", perf_path.name)
    if filename_match:
        try:
            return datetime.strptime(filename_match.group(1), "%Y%m%d_%H%M%S")
        except ValueError:
            pass

    if perf_path.exists():
        return datetime.fromtimestamp(perf_path.stat().st_mtime)
    return None


def _resolve_explicit_device_log(device_log: str) -> Tuple[Optional[Path], str]:
    path = Path(device_log).expanduser()
    if path.is_file():
        return path, "explicit --device-log file"

    if path.is_dir():
        best = _latest_log(path.glob("*.log"))
        if best is None:
            return None, f"explicit --device-log directory has no .log files: {path}"
        return best, f"explicit --device-log directory: {path}"

    if glob.has_magic(device_log):
        best = _latest_log(Path(match) for match in glob.iglob(str(path)))
        if best is None:
            return None, f"explicit --device-log glob had no matches: {device_log}"
        return best, f"explicit --device-log glob: {device_log}"

    return None, f"explicit --device-log path not found: {path}"


def _resolve_nearest_log(root: Path, perf_path: Optional[Path]) -> Tuple[Optional[Path], str]:
    device_dirs = sorted([p for p in root.glob("device-*") if p.is_dir()])
    if not device_dirs:
        return None, f"no device-* directories found under {root}"

    paths = (path for directory in device_dirs for path in directory.glob("*.log"))
    perf_dt = _extract_perf_timestamp(perf_path)
    if perf_dt is None:
        best = _latest_log(paths)
        reason = "auto-scan device-* (newest log)"
    else:
        perf_ts = perf_dt.timestamp()
        _, best = min(
            _log_candidates(paths),
            key=lambda item: (abs(item[0] - perf_ts), -item[0], str(item[1])),
            default=(0, None),
        )
        reason = f"auto-scan device-* (closest log to perf timestamp {perf_dt:%Y-%m-%d %H:%M:%S})"
    if best is None:
        return None, f"no .log files found under {root}/device-*"
    return best, reason


def resolve_device_log_path(
    device_id: Optional[str] = None,
    device_log: Optional[str] = None,
    perf_path: Optional[Path] = None,
) -> Tuple[Optional[Path], str]:
    """Resolve device log path with deterministic precedence.

    Priority:
      1) --device-log explicit path/dir/glob
      2) --device-id -> <log_root>/device-<id>/ newest .log
      3) auto-scan all device-* and choose nearest to perf timestamp
    """
    if device_log:
        return _resolve_explicit_device_log(device_log)

    root = get_log_root()

    if device_id is not None:
        device_dir = root / f"device-{device_id}"
        best = _latest_log(device_dir.glob("*.log"))
        if best is None:
            return None, f"device-id selection failed: no .log files in {device_dir}"
        return best, f"device-id selection: device-{device_id} under {root}"

    return _resolve_nearest_log(root, perf_path)
