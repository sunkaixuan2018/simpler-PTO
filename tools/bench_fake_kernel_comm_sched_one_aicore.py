#!/usr/bin/env python3
"""
Run and analyze fake_kernel_comm_sched_one_aicore.

The script mirrors the old bench_gather_comm_sched.py flow at a smaller scope:
run the distributed example for size/strategy combinations, verify correctness,
enable runtime profiling, then summarize foreground gather latency and debug
wait status.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import struct
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CASE_ROOT = PROJECT_ROOT / "examples" / "a2a3" / "tensormap_and_ringbuffer" / "fake_kernel_comm_sched_one_aicore"
KERNELS_DIR = CASE_ROOT / "kernels"
GOLDEN = CASE_ROOT / "golden.py"
RUN_EXAMPLE = PROJECT_ROOT / "examples" / "scripts" / "run_example.py"
SWIMLANE_CONVERTER = PROJECT_ROOT / "tools" / "swimlane_converter.py"
ARTIFACT_DIR = PROJECT_ROOT / "build" / "distributed" / "artifacts"
KERNEL_CONFIG = KERNELS_DIR / "kernel_config.py"

STRATEGIES = {
    "hybrid": 0,
    "mte": 1,
    "sync": 1,
    "sdma": 2,
    "async": 2,
}

FUNC_GATHER_SYNC = 1
FUNC_GATHER_ASYNC = 2
FUNC_BARRIER = 4
FUNC_DUMMY_SYNC = 5
FUNC_DUMMY_ASYNC = 6


def parse_size(value: str) -> int:
    text = value.strip().lower()
    scale = 1
    if text.endswith("kb"):
        scale = 1024
        text = text[:-2]
    elif text.endswith("k"):
        scale = 1024
        text = text[:-1]
    elif text.endswith("mb"):
        scale = 1024 * 1024
        text = text[:-2]
    elif text.endswith("m"):
        scale = 1024 * 1024
        text = text[:-1]
    elif text.endswith("gb"):
        scale = 1024 * 1024 * 1024
        text = text[:-2]
    elif text.endswith("g"):
        scale = 1024 * 1024 * 1024
        text = text[:-1]
    return int(float(text) * scale)


def human_bytes(value: int) -> str:
    for suffix, scale in (("G", 1024**3), ("M", 1024**2), ("K", 1024)):
        if value >= scale and value % scale == 0:
            return f"{value // scale}{suffix}"
    return str(value)


def parse_csv_list(value: str) -> list[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def parse_devices(value: str) -> list[int]:
    devices: list[int] = []
    for item in parse_csv_list(value):
        if "-" in item:
            start, end = item.split("-", 1)
            devices.extend(range(int(start), int(end) + 1))
        else:
            devices.append(int(item))
    return devices


def percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    idx = int(round((pct / 100.0) * (len(ordered) - 1)))
    return ordered[idx]


def trimmed(values: list[float], ratio: float) -> list[float]:
    if not values:
        return []
    ordered = sorted(values)
    cut = int(len(ordered) * ratio)
    if cut == 0 or cut * 2 >= len(ordered):
        return ordered
    return ordered[cut:-cut]


def summarize(values: list[float], trim_ratio: float) -> dict[str, float | int | None]:
    kept = trimmed(values, trim_ratio)
    if not kept:
        return {"count": 0, "avg": None, "p50": None, "p90": None, "min": None, "max": None}
    return {
        "count": len(kept),
        "avg": statistics.fmean(kept),
        "p50": percentile(kept, 50),
        "p90": percentile(kept, 90),
        "min": min(kept),
        "max": max(kept),
    }


def perf_files_since(since: float) -> list[Path]:
    candidates = list(ARTIFACT_DIR.glob("rank_*/perf_swimlane_*.json"))
    candidates += list((PROJECT_ROOT / "outputs").glob("perf_swimlane_*.json"))
    return sorted(p for p in candidates if p.stat().st_mtime >= since)


def rank_from_perf_path(path: Path) -> int | None:
    parent = path.parent.name
    if not parent.startswith("rank_"):
        return None
    try:
        return int(parent.split("_", 1)[1])
    except ValueError:
        return None


def merged_swimlane_path(perf_path: Path, strategy: str, size_bytes: int) -> Path:
    stem = perf_path.stem
    timestamp = stem[len("perf_swimlane_"):] if stem.startswith("perf_swimlane_") else str(int(time.time()))
    rank = perf_path.parent.name if perf_path.parent.name.startswith("rank_") else "rank_unknown"
    return (
        PROJECT_ROOT
        / "outputs"
        / f"merged_swimlane_one_aicore_{strategy}_{size_bytes}B_{rank}_{timestamp}.json"
    )


def convert_perf_files(perf_files: list[Path], args, strategy: str, size_bytes: int) -> dict:
    if not perf_files:
        return {"merged_swimlane_files": [], "merged_swimlane_failures": []}

    devices = parse_devices(args.devices)
    merged_files: list[str] = []
    failures: list[dict[str, str]] = []

    for perf_path in perf_files:
        output_path = merged_swimlane_path(perf_path, strategy, size_bytes)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable,
            str(SWIMLANE_CONVERTER),
            str(perf_path),
            "-o",
            str(output_path),
            "-k",
            str(KERNEL_CONFIG),
        ]

        rank = rank_from_perf_path(perf_path)
        if rank is not None and 0 <= rank < len(devices):
            cmd += ["-d", str(devices[rank])]

        proc = subprocess.run(
            cmd,
            cwd=PROJECT_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        if proc.returncode == 0 and output_path.exists():
            merged_files.append(str(output_path))
            print(f"  merged swimlane: {output_path}")
            continue

        message = (proc.stderr or proc.stdout or "").strip()
        failures.append({
            "perf_file": str(perf_path),
            "output_file": str(output_path),
            "returncode": str(proc.returncode),
            "message": message[-2000:],
        })
        print(f"  warning: swimlane_converter failed for {perf_path} (rc={proc.returncode})")
        if message:
            print(message[-1000:])

    return {
        "merged_swimlane_files": merged_files,
        "merged_swimlane_failures": failures,
    }


def load_perf_tasks(perf_files: list[Path], func_ids: set[int], warmup: int) -> tuple[list[dict], Counter]:
    measured: list[dict] = []
    func_counter: Counter = Counter()
    for path in perf_files:
        data = json.loads(path.read_text())
        tasks = [t for t in data.get("tasks", []) if int(t.get("func_id", -1)) in func_ids]
        tasks.sort(key=lambda t: (t.get("dispatch_time_us", t.get("start_time_us", 0.0)), t.get("task_id", 0)))
        if len(tasks) > warmup:
            tasks = tasks[warmup:]
        for task in tasks:
            func_counter[int(task.get("func_id", -1))] += 1
        measured.extend(tasks)
    return measured, func_counter


def analyze_perf(perf_files: list[Path], warmup: int, trim_ratio: float) -> dict:
    gather_tasks, gather_counter = load_perf_tasks(
        perf_files, {FUNC_GATHER_SYNC, FUNC_GATHER_ASYNC}, warmup)
    barrier_tasks, _ = load_perf_tasks(perf_files, {FUNC_BARRIER}, warmup)
    dummy_tasks, dummy_counter = load_perf_tasks(
        perf_files, {FUNC_DUMMY_SYNC, FUNC_DUMMY_ASYNC}, warmup)

    def exec_values(tasks: list[dict]) -> list[float]:
        return [float(t.get("duration_us", 0.0)) for t in tasks]

    def latency_values(tasks: list[dict]) -> list[float]:
        values = []
        for task in tasks:
            dispatch = task.get("dispatch_time_us")
            finish = task.get("finish_time_us")
            if dispatch is not None and finish is not None:
                values.append(float(finish) - float(dispatch))
        return values

    return {
        "perf_files": [str(p) for p in perf_files],
        "gather_func_id": gather_counter.most_common(1)[0][0] if gather_counter else None,
        "gather_exec": summarize(exec_values(gather_tasks), trim_ratio),
        "gather_latency": summarize(latency_values(gather_tasks), trim_ratio),
        "barrier_exec": summarize(exec_values(barrier_tasks), trim_ratio),
        "dummy_func_id": dummy_counter.most_common(1)[0][0] if dummy_counter else None,
        "dummy_exec": summarize(exec_values(dummy_tasks), trim_ratio),
    }


def analyze_debug_counts() -> dict:
    values: list[int] = []
    for path in ARTIFACT_DIR.glob("rank_*/debug_poll_counts.bin"):
        raw = path.read_bytes()
        count = len(raw) // 4
        values.extend(struct.unpack(f"<{count}i", raw))
    if not values:
        return {"debug_count": 0, "debug_success_ratio": None}
    ok = sum(1 for v in values if v == 1)
    return {"debug_count": len(values), "debug_success_ratio": ok / len(values)}


def run_case(args, strategy: str, size_bytes: int) -> dict:
    gather_count = max(1, (size_bytes + 3) // 4)
    env = os.environ.copy()
    env["PTO_PLATFORM"] = "a2a3"
    env["PTO_NRANKS"] = str(len(parse_devices(args.devices)))
    env["GATHER_COUNT"] = str(gather_count)
    env["GATHER_STRATEGY"] = strategy
    env["N_ITER"] = str(args.n_iter)
    env["DUMMY_COMM_BYTES"] = str(parse_size(args.dummy_comm_bytes))
    env["EXTREME_SERIALIZE_DUMMY"] = "1" if args.serialize_dummy else "0"

    if args.pto_isa_root:
        env["PTO_ISA_ROOT"] = str(Path(args.pto_isa_root).expanduser().resolve())

    cmd = [
        sys.executable,
        str(RUN_EXAMPLE),
        "-k",
        str(KERNELS_DIR),
        "-g",
        str(GOLDEN),
        "-p",
        "a2a3",
        "--devices",
        args.devices,
        "--clone-protocol",
        args.clone_protocol,
        "--enable-profiling",
    ]
    if args.pto_isa_commit:
        cmd += ["--pto-isa-commit", args.pto_isa_commit]

    since = time.time() - 1.0
    print(f"\n=== strategy={strategy} size={human_bytes(size_bytes)} count={gather_count} ===")
    proc = subprocess.run(cmd, cwd=PROJECT_ROOT, env=env, check=False)
    perf_files = perf_files_since(since)

    result = {
        "strategy": strategy,
        "strategy_code": STRATEGIES[strategy],
        "size_bytes": size_bytes,
        "gather_count": gather_count,
        "n_iter": args.n_iter,
        "warmup": args.warmup,
        "returncode": proc.returncode,
        "run_ok": proc.returncode == 0,
    }
    result.update(analyze_perf(perf_files, args.warmup, args.trim_ratio) if perf_files else {})
    result.update(convert_perf_files(perf_files, args, strategy, size_bytes))
    result.update(analyze_debug_counts())
    return result


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "strategy",
        "size_bytes",
        "gather_count",
        "run_ok",
        "gather_func_id",
        "gather_exec_avg_us",
        "gather_latency_avg_us",
        "barrier_exec_avg_us",
        "dummy_func_id",
        "dummy_exec_avg_us",
        "debug_success_ratio",
        "perf_files",
        "merged_swimlane_files",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "strategy": row.get("strategy"),
                "size_bytes": row.get("size_bytes"),
                "gather_count": row.get("gather_count"),
                "run_ok": row.get("run_ok"),
                "gather_func_id": row.get("gather_func_id"),
                "gather_exec_avg_us": (row.get("gather_exec") or {}).get("avg"),
                "gather_latency_avg_us": (row.get("gather_latency") or {}).get("avg"),
                "barrier_exec_avg_us": (row.get("barrier_exec") or {}).get("avg"),
                "dummy_func_id": row.get("dummy_func_id"),
                "dummy_exec_avg_us": (row.get("dummy_exec") or {}).get("avg"),
                "debug_success_ratio": row.get("debug_success_ratio"),
                "perf_files": ";".join(row.get("perf_files", [])),
                "merged_swimlane_files": ";".join(row.get("merged_swimlane_files", [])),
            })


def print_summary(rows: list[dict]) -> None:
    def fmt_us(value) -> str:
        return f"{value:.2f}" if value is not None else "N/A"

    def fmt_ratio(value) -> str:
        return f"{value:.3f}" if value is not None else "N/A"

    print("\nSummary")
    print(f"{'strategy':<8} {'size':>8} {'ok':>3} {'func':>4} {'exec_us':>10} {'lat_us':>10} {'debug':>8}")
    for row in rows:
        exec_avg = (row.get("gather_exec") or {}).get("avg")
        lat_avg = (row.get("gather_latency") or {}).get("avg")
        debug_ratio = row.get("debug_success_ratio")
        print(
            f"{row['strategy']:<8} {human_bytes(row['size_bytes']):>8} "
            f"{str(row['run_ok']):>3} {str(row.get('gather_func_id')):>4} "
            f"{fmt_us(exec_avg):>10} "
            f"{fmt_us(lat_avg):>10} "
            f"{fmt_ratio(debug_ratio):>8}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark fake_kernel_comm_sched_one_aicore")
    parser.add_argument("--devices", default="0,1,2,3", help="Comma-separated device ids")
    parser.add_argument("--strategies", default="mte,sdma,hybrid", help="mte,sdma,hybrid")
    parser.add_argument("--sizes", default="1K,4K,16K,64K,256K,1M", help="Transfer sizes in bytes")
    parser.add_argument("--n-iter", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--trim-ratio", type=float, default=0.10)
    parser.add_argument("--dummy-comm-bytes", default="16M")
    parser.add_argument("--serialize-dummy", action="store_true")
    parser.add_argument("--pto-isa-root", default=None)
    parser.add_argument("--pto-isa-commit", default=None)
    parser.add_argument("--clone-protocol", default="https", choices=["ssh", "https"])
    parser.add_argument("--keep-going", action="store_true")
    parser.add_argument("--output-csv", default=str(PROJECT_ROOT / "outputs" / "bench_one_aicore.csv"))
    parser.add_argument("--output-json", default=str(PROJECT_ROOT / "outputs" / "bench_one_aicore.json"))
    args = parser.parse_args()

    strategies = parse_csv_list(args.strategies)
    for strategy in strategies:
        if strategy not in STRATEGIES:
            raise ValueError(f"Unsupported strategy {strategy!r}; choose from {sorted(STRATEGIES)}")
    sizes = [parse_size(x) for x in parse_csv_list(args.sizes)]

    rows = []
    for strategy in strategies:
        for size_bytes in sizes:
            row = run_case(args, strategy, size_bytes)
            rows.append(row)
            if not row["run_ok"] and not args.keep_going:
                break
        if rows and not rows[-1]["run_ok"] and not args.keep_going:
            break

    write_csv(Path(args.output_csv), rows)
    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(rows, indent=2))
    print_summary(rows)
    print(f"\nCSV: {args.output_csv}")
    print(f"JSON: {args.output_json}")
    return 0 if all(r["run_ok"] for r in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
