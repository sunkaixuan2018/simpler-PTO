#!/usr/bin/env python3
"""
统计 swimlane 日志中的 CommBarrier 事件耗时。

用法示例:
  python tools/barrier_log_stats.py --log-dir output
  python tools/barrier_log_stats.py --log-dir output --timestamp 20260323_155712
  python tools/barrier_log_stats.py --log-dir output --device-start 0 --device-count 6
  python tools/barrier_log_stats.py --log-dir output --view aicore
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Iterable


@dataclass
class BarrierEvent:
    file_name: str
    device_id: int
    view: str
    ts_us: float
    dur_us: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="统计 CommBarrier 时长（支持 prefXXX_dX.log）")
    parser.add_argument(
        "--log-dir",
        type=Path,
        required=True,
        help="日志目录，例如 output",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default=r"(pref.*_d\d+\.log|merged_swimlane_\d+_\d+_d\d+\.json)",
        help=r"文件名正则，默认同时支持 prefXXX_dX.log 和 merged_swimlane_YYYYMMDD_HHMMSS_dX.json",
    )
    parser.add_argument(
        "--timestamp",
        type=str,
        default=None,
        help="只统计包含该时间串的文件，例如 20260323_155712",
    )
    parser.add_argument(
        "--device-start",
        type=int,
        default=None,
        help="设备起始编号（和 --device-count 配合），例如 0",
    )
    parser.add_argument(
        "--device-count",
        type=int,
        default=None,
        help="设备数量（和 --device-start 配合），例如 6，表示统计 d0..d5",
    )
    parser.add_argument(
        "--view",
        choices=("aicore", "aicpu", "both"),
        default="both",
        help="统计视图：aicore(pid=1) / aicpu(pid=2) / both",
    )
    parser.add_argument(
        "--event-name",
        type=str,
        default="CommBarrier",
        help="要统计的事件名前缀，默认 CommBarrier",
    )
    return parser.parse_args()


def _extract_device_id(file_name: str) -> int:
    match = re.search(r"_d(\d+)\.(?:log|json)$", file_name)
    return int(match.group(1)) if match else -1


def _iter_target_files(
    log_dir: Path,
    pattern: str,
    timestamp: str | None,
    device_start: int | None,
    device_count: int | None,
) -> Iterable[Path]:
    regex = re.compile(pattern)
    files = []
    for p in log_dir.iterdir():
        if not p.is_file():
            continue
        if not regex.fullmatch(p.name):
            continue
        if timestamp is not None and timestamp not in p.name:
            continue

        dev = _extract_device_id(p.name)
        if device_start is not None or device_count is not None:
            if device_start is None or device_count is None:
                continue
            if dev < device_start or dev >= device_start + device_count:
                continue
        files.append(p)
    files.sort(key=lambda p: p.name)
    return files


def _pid_to_view(pid: int) -> str:
    if pid == 1:
        return "aicore"
    if pid == 2:
        return "aicpu"
    return "unknown"


def parse_barrier_events(path: Path, event_prefix: str) -> list[BarrierEvent]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path.name} 不是合法 JSON: {exc}") from exc

    events = payload.get("traceEvents", [])
    result: list[BarrierEvent] = []
    dev_id = _extract_device_id(path.name)

    for ev in events:
        if ev.get("ph") != "X":
            continue
        name = str(ev.get("name", ""))
        if not name.startswith(event_prefix):
            continue
        pid = int(ev.get("pid", -1))
        view = _pid_to_view(pid)
        if view == "unknown":
            continue

        ts_us = float(ev.get("ts", 0.0))
        dur_us = float(ev.get("dur", 0.0))
        result.append(
            BarrierEvent(
                file_name=path.name,
                device_id=dev_id,
                view=view,
                ts_us=ts_us,
                dur_us=dur_us,
            )
        )

    return result


def summarize(events: list[BarrierEvent], view: str) -> None:
    filtered = [e for e in events if view == "both" or e.view == view]
    if not filtered:
        print("没有匹配到 CommBarrier 事件。")
        return

    filtered.sort(key=lambda e: (e.view, e.device_id, e.file_name, e.ts_us))

    print("=== 明细（按视图/设备/时间）===")
    for e in filtered:
        print(
            f"[{e.view:6}] dev={e.device_id:>2}  dur={e.dur_us / 1000.0:9.3f} ms"
            f"  ts={e.ts_us / 1000.0:9.3f} ms  file={e.file_name}"
        )

    print("\n=== 汇总（按视图）===")
    for cur_view in ("aicore", "aicpu"):
        cur = [e for e in filtered if e.view == cur_view]
        if not cur:
            continue
        durs_ms = [e.dur_us / 1000.0 for e in cur]
        print(
            f"{cur_view:6}: count={len(cur):>2}, min={min(durs_ms):.3f} ms, "
            f"max={max(durs_ms):.3f} ms, avg={mean(durs_ms):.3f} ms"
        )

    print("\n=== 每设备（同视图下可能有多条，取最大值）===")
    by_key: dict[tuple[str, int], float] = {}
    for e in filtered:
        key = (e.view, e.device_id)
        by_key[key] = max(by_key.get(key, 0.0), e.dur_us / 1000.0)
    for (cur_view, dev), dur_ms in sorted(by_key.items(), key=lambda x: (x[0][0], x[0][1])):
        print(f"[{cur_view:6}] dev={dev:>2}  max_barrier={dur_ms:.3f} ms")


def main() -> int:
    args = parse_args()
    log_dir: Path = args.log_dir
    if not log_dir.exists() or not log_dir.is_dir():
        print(f"log-dir 不存在或不是目录: {log_dir}")
        return 2

    files = list(
        _iter_target_files(
            log_dir=log_dir,
            pattern=args.pattern,
            timestamp=args.timestamp,
            device_start=args.device_start,
            device_count=args.device_count,
        )
    )
    if not files:
        print(
            "目录中没有匹配文件，"
            f"pattern={args.pattern}, timestamp={args.timestamp}, "
            f"device_start={args.device_start}, device_count={args.device_count}"
        )
        return 3

    all_events: list[BarrierEvent] = []
    bad_files: list[str] = []

    for fp in files:
        try:
            all_events.extend(parse_barrier_events(fp, args.event_name))
        except ValueError as exc:
            bad_files.append(str(exc))

    if bad_files:
        print("以下文件解析失败：")
        for msg in bad_files:
            print(f"  - {msg}")
        print("")

    summarize(all_events, args.view)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
