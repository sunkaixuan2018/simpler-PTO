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
    end_us: float
    pre_event_name: str | None = None
    pre_event_ts_us: float | None = None
    pre_event_dur_us: float | None = None


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
    parser.add_argument(
        "--pre-event",
        type=str,
        default="WindowMemCopyIn",
        help="用于定位 barrier 前慢点的前序事件名前缀，默认 WindowMemCopyIn；传空字符串可关闭",
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

    x_events = [ev for ev in events if ev.get("ph") == "X"]

    for ev in x_events:
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
        end_us = ts_us + dur_us
        result.append(
            BarrierEvent(
                file_name=path.name,
                device_id=dev_id,
                view=view,
                ts_us=ts_us,
                dur_us=dur_us,
                end_us=end_us,
            )
        )

    return result


def attach_pre_event(events: list[BarrierEvent], log_dir: Path, pre_event_prefix: str) -> None:
    if not pre_event_prefix:
        return

    by_file: dict[str, list[BarrierEvent]] = {}
    for e in events:
        by_file.setdefault(e.file_name, []).append(e)

    for file_name, barrier_list in by_file.items():
        path = log_dir / file_name
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        x_events = [ev for ev in payload.get("traceEvents", []) if ev.get("ph") == "X"]

        # 按 (pid, tid) 分组，匹配同一个 lane 上 barrier 之前最后一个 pre_event
        lane_events: dict[tuple[int, int], list[dict]] = {}
        for ev in x_events:
            pid = int(ev.get("pid", -1))
            tid = int(ev.get("tid", -1))
            lane_events.setdefault((pid, tid), []).append(ev)
        for lane in lane_events:
            lane_events[lane].sort(key=lambda x: float(x.get("ts", 0.0)))

        for b in barrier_list:
            target_pid = 1 if b.view == "aicore" else 2
            best_match: dict | None = None
            best_end = -1.0
            for (pid, _tid), arr in lane_events.items():
                if pid != target_pid:
                    continue
                for ev in arr:
                    name = str(ev.get("name", ""))
                    if not name.startswith(pre_event_prefix):
                        continue
                    ev_ts = float(ev.get("ts", 0.0))
                    ev_dur = float(ev.get("dur", 0.0))
                    ev_end = ev_ts + ev_dur
                    if ev_end <= b.ts_us and ev_end > best_end:
                        best_end = ev_end
                        best_match = ev
            if best_match is not None:
                b.pre_event_name = str(best_match.get("name", ""))
                b.pre_event_ts_us = float(best_match.get("ts", 0.0))
                b.pre_event_dur_us = float(best_match.get("dur", 0.0))


def summarize(events: list[BarrierEvent], view: str, pre_event_prefix: str) -> None:
    filtered = [e for e in events if view == "both" or e.view == view]
    if not filtered:
        print("没有匹配到 CommBarrier 事件。")
        return

    filtered.sort(key=lambda e: (e.view, e.device_id, e.file_name, e.ts_us))
    max_end_by_view: dict[str, float] = {}
    min_start_by_view: dict[str, float] = {}
    for cur_view in ("aicore", "aicpu"):
        cur = [e for e in filtered if e.view == cur_view]
        if not cur:
            continue
        max_end_by_view[cur_view] = max(e.end_us for e in cur)
        min_start_by_view[cur_view] = min(e.ts_us for e in cur)

    print("=== 明细（按视图/设备/时间）===")
    for e in filtered:
        latest_end = max_end_by_view.get(e.view, e.end_us)
        wait_to_latest_ms = (latest_end - e.ts_us) / 1000.0
        relative_end_gap_ms = (latest_end - e.end_us) / 1000.0
        print(
            f"[{e.view:6}] dev={e.device_id:>2}  dur={e.dur_us / 1000.0:9.3f} ms"
            f"  ts={e.ts_us / 1000.0:9.3f} ms  end={e.end_us / 1000.0:9.3f} ms"
            f"  wait_to_latest_end={wait_to_latest_ms:8.3f} ms"
            f"  end_gap={relative_end_gap_ms:8.3f} ms  file={e.file_name}"
        )

    print("\n=== 汇总（按视图）===")
    for cur_view in ("aicore", "aicpu"):
        cur = [e for e in filtered if e.view == cur_view]
        if not cur:
            continue
        durs_ms = [e.dur_us / 1000.0 for e in cur]
        window_ms = (max_end_by_view[cur_view] - min_start_by_view[cur_view]) / 1000.0
        print(
            f"{cur_view:6}: count={len(cur):>2}, min={min(durs_ms):.3f} ms, "
            f"max={max(durs_ms):.3f} ms, avg={mean(durs_ms):.3f} ms, "
            f"barrier_window={window_ms:.3f} ms"
        )

    print("\n=== 每设备（同视图下可能有多条，取最大值）===")
    by_key: dict[tuple[str, int], float] = {}
    for e in filtered:
        key = (e.view, e.device_id)
        by_key[key] = max(by_key.get(key, 0.0), e.dur_us / 1000.0)
    for (cur_view, dev), dur_ms in sorted(by_key.items(), key=lambda x: (x[0][0], x[0][1])):
        print(f"[{cur_view:6}] dev={dev:>2}  max_barrier={dur_ms:.3f} ms")

    print("\n=== 问题定位（按视图）===")
    for cur_view in ("aicore", "aicpu"):
        cur = [e for e in filtered if e.view == cur_view]
        if not cur:
            continue
        earliest = min(cur, key=lambda x: x.ts_us)
        latest = max(cur, key=lambda x: x.end_us)
        shortest = min(cur, key=lambda x: x.dur_us)
        longest = max(cur, key=lambda x: x.dur_us)
        print(f"[{cur_view}]")
        print(
            f"  - 最早进 barrier: dev={earliest.device_id}, ts={earliest.ts_us/1000.0:.3f} ms"
        )
        print(
            f"  - 最晚出 barrier: dev={latest.device_id}, end={latest.end_us/1000.0:.3f} ms"
        )
        print(
            f"  - 等待最长: dev={longest.device_id}, dur={longest.dur_us/1000.0:.3f} ms"
        )
        print(
            f"  - 等待最短(通常最后到): dev={shortest.device_id}, dur={shortest.dur_us/1000.0:.3f} ms"
        )

    if pre_event_prefix:
        print(f"\n=== 前序事件关联（{pre_event_prefix} -> CommBarrier）===")
        for cur_view in ("aicore", "aicpu"):
            cur = [e for e in filtered if e.view == cur_view]
            if not cur:
                continue
            print(f"[{cur_view}]")
            for e in sorted(cur, key=lambda x: x.device_id):
                if e.pre_event_name is None:
                    print(f"  - dev={e.device_id}: 未找到前序事件")
                    continue
                pre_end_us = (e.pre_event_ts_us or 0.0) + (e.pre_event_dur_us or 0.0)
                gap_ms = (e.ts_us - pre_end_us) / 1000.0
                print(
                    f"  - dev={e.device_id}: {e.pre_event_name} dur={(e.pre_event_dur_us or 0.0)/1000.0:.3f} ms, "
                    f"pre_end={pre_end_us/1000.0:.3f} ms, barrier_start={e.ts_us/1000.0:.3f} ms, gap={gap_ms:.3f} ms"
                )


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

    attach_pre_event(all_events, log_dir=log_dir, pre_event_prefix=args.pre_event.strip())
    summarize(all_events, args.view, pre_event_prefix=args.pre_event.strip())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
