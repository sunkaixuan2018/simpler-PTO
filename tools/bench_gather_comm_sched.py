#!/usr/bin/env python3
"""
Batch benchmark for fake_kernel_comm_sched: gather operator performance.

Tests MTE (sync TGATHER) and SDMA (async TGET_ASYNC) strategies across
multiple total communication sizes on 4 cards.

Each invocation runs N_ITER=200 gather pipelines back-to-back. The first
N_WARMUP=100 are discarded; the remaining measured window is averaged after
trimming both tails by a configurable ratio (default 10%).

Usage:
    python tools/bench_gather_comm_sched.py --platform a2a3 --first-device 4
    python tools/bench_gather_comm_sched.py --platform a2a3sim
    python tools/bench_gather_comm_sched.py --strategies mte sdma --sizes 4K 16K 64K
"""

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Benchmark configuration
# ---------------------------------------------------------------------------

STRATEGY_NAMES = ["mte", "sdma", "hybrid"]

# Total communication volumes in bytes (all ranks combined, float32 gather)
SIZE_LABELS = [
    "1K", "2K", "4K", "8K", "16K", "32K", "64K",
    "128K", "256K", "512K", "1M", "2M", "4M", "8M",
]
SIZE_BYTES = {
    "1K":    1 * 1024,
    "2K":    2 * 1024,
    "4K":    4 * 1024,
    "8K":    8 * 1024,
    "16K":  16 * 1024,
    "32K":  32 * 1024,
    "64K":  64 * 1024,
    "128K": 128 * 1024,
    "256K": 256 * 1024,
    "512K": 512 * 1024,
    "1M":   1 * 1024 ** 2,
    "2M":   2 * 1024 ** 2,
    "4M":   4 * 1024 ** 2,
    "8M":   8 * 1024 ** 2,
}

# Gather func_ids in the profiling data
FUNC_ID_GATHER_SYNC  = 1  # GatherSync  (MTE / TGATHER)
FUNC_ID_GATHER_ASYNC = 2  # GatherAsync (SDMA / TGET_ASYNC)
FUNC_ID_COMM_BARRIER = 4  # CommBarrier

N_ITER   = 200  # total gather iterations per invocation (must match C++ N_ITER)
N_WARMUP = 100  # warm-up iterations to discard
TRIM_RATIO_DEFAULT = 0.10  # trim this ratio from both tails before averaging

# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

_SCRIPT_DIR   = Path(__file__).parent.resolve()
_PROJECT_ROOT = _SCRIPT_DIR.parent
_OUTPUTS_DIR  = _PROJECT_ROOT / "outputs"

_BASE_EXAMPLE_DIR = _PROJECT_ROOT / "examples" / "tensormap_and_ringbuffer"
_NORMAL_KERNELS_DIR = _BASE_EXAMPLE_DIR / "fake_kernel_comm_sched" / "kernels"
_NORMAL_GOLDEN_PATH = _BASE_EXAMPLE_DIR / "fake_kernel_comm_sched" / "golden.py"
_EXTREME_KERNELS_DIR = _BASE_EXAMPLE_DIR / "fake_kernel_comm_sched_extreme" / "kernels"
_EXTREME_GOLDEN_PATH = _BASE_EXAMPLE_DIR / "fake_kernel_comm_sched_extreme" / "golden.py"
_RUNNER       = _PROJECT_ROOT / "examples" / "scripts" / "multi_card_run_example.py"


def _gather_count_for(total_bytes: int, n_ranks: int) -> int:
    """Elements per rank for a given total comm volume (float32)."""
    return total_bytes // (n_ranks * 4)


# ---------------------------------------------------------------------------
# Diagnostic helpers
# ---------------------------------------------------------------------------

def _compute_distribution_stats(values: list[float]) -> dict:
    """Compute p50/p90/p99/max/min/avg and outlier percentage for a list of values."""
    if not values:
        return {"count": 0, "avg": 0, "min": 0, "p50": 0, "p90": 0, "p99": 0, "max": 0, "pct_gt_2x_p50": 0}
    s = sorted(values)
    n = len(s)
    p50 = s[n // 2]
    p90 = s[int(n * 0.90)]
    p99 = s[int(n * 0.99)]
    gt_2x = sum(1 for v in s if v > 2 * p50)
    return {
        "count": n,
        "avg": sum(s) / n,
        "min": s[0],
        "p50": p50,
        "p90": p90,
        "p99": p99,
        "max": s[-1],
        "pct_gt_2x_p50": gt_2x / n * 100,
    }


def _parse_gather_detail(perf_file: Path) -> tuple[list[dict], list[dict]]:
    """
    Parse per-iteration timing breakdown from perf JSON.

    Returns (gather_rows, barrier_rows) where each row is a dict with:
        iter, dispatch_time_us, start_time_us, end_time_us, finish_time_us,
        head_oh_us, exec_us, tail_oh_us, latency_us
    """
    with open(perf_file, encoding="utf-8") as f:
        data = json.load(f)

    tasks = data.get("tasks", [])

    gather_tasks = [
        t for t in tasks
        if t.get("func_id") in (FUNC_ID_GATHER_SYNC, FUNC_ID_GATHER_ASYNC)
    ]
    barrier_tasks = [
        t for t in tasks
        if t.get("func_id") == FUNC_ID_COMM_BARRIER
    ]

    def _to_rows(task_list):
        task_list.sort(key=lambda t: t.get("dispatch_time_us", t.get("start_time_us", 0)))
        rows = []
        for i, t in enumerate(task_list):
            dispatch = t.get("dispatch_time_us", 0)
            start = t.get("start_time_us", 0)
            end = t.get("end_time_us", 0)
            finish = t.get("finish_time_us", 0)
            rows.append({
                "iter": i,
                "func_id": t.get("func_id"),
                "dispatch_time_us": dispatch,
                "start_time_us": start,
                "end_time_us": end,
                "finish_time_us": finish,
                "head_oh_us": start - dispatch,
                "exec_us": t.get("duration_us", end - start),
                "tail_oh_us": finish - end,
                "latency_us": finish - dispatch,
            })
        return rows

    return _to_rows(gather_tasks), _to_rows(barrier_tasks)


def _print_diagnostic_report(
    strategy: str,
    size_label: str,
    gather_rows: list[dict],
    barrier_rows: list[dict],
) -> None:
    """Print per-segment distribution stats and anomaly correlation."""
    if not gather_rows:
        print("    [diagnostic] No gather data found.")
        return

    measured = gather_rows[N_WARMUP:]
    if not measured:
        measured = gather_rows

    # --- Per-segment distribution ---
    segments = ["head_oh_us", "exec_us", "tail_oh_us", "latency_us"]
    stats = {seg: _compute_distribution_stats([r[seg] for r in measured]) for seg in segments}

    print(f"\n    === Diagnostic: {strategy.upper()} {size_label} (n={len(measured)} measured) ===")
    print(f"    {'Segment':<14s} {'avg':>10s} {'p50':>10s} {'p90':>10s} {'p99':>10s} {'max':>10s} {'>2x_p50':>8s}")
    print(f"    {'-'*66}")
    for seg in segments:
        s = stats[seg]
        label = seg.replace("_us", "")
        print(f"    {label:<14s} {s['avg']:>10.1f} {s['p50']:>10.1f} {s['p90']:>10.1f} "
              f"{s['p99']:>10.1f} {s['max']:>10.1f} {s['pct_gt_2x_p50']:>7.1f}%")

    # --- Identify dominant anomaly segment ---
    max_range = 0
    dominant_seg = ""
    for seg in ["head_oh_us", "exec_us", "tail_oh_us"]:
        rng = stats[seg]["max"] - stats[seg]["p50"]
        if rng > max_range:
            max_range = rng
            dominant_seg = seg
    print(f"    >> Dominant anomaly segment: {dominant_seg.replace('_us', '')} "
          f"(max-p50 = {max_range:.1f} us)")

    # --- Anomaly correlation with barrier ---
    p50_lat = stats["latency_us"]["p50"]
    anomaly_iters = [r for r in measured if r["latency_us"] > 2 * p50_lat]
    if anomaly_iters and barrier_rows:
        # Build barrier lookup by approximate iteration index
        # Barrier tasks: 1 startup + N_ITER per-iteration = N_ITER+1 total
        # Per-iteration barriers start at index 1, gather iterations at index 0
        barrier_measured = barrier_rows[N_WARMUP + 1:] if len(barrier_rows) > N_WARMUP + 1 else barrier_rows[1:]
        print(f"\n    Anomaly iterations (latency > 2x p50 = {2*p50_lat:.1f} us): {len(anomaly_iters)}/{len(measured)}")
        print(f"    {'iter':>6s} {'gather_exec':>12s} {'gather_lat':>12s} {'barrier_exec':>13s} {'barrier_tail':>13s}")
        shown = 0
        for r in anomaly_iters:
            idx = r["iter"] - N_WARMUP
            b_exec = ""
            b_tail = ""
            if 0 <= idx < len(barrier_measured):
                b = barrier_measured[idx]
                b_exec = f"{b['exec_us']:.1f}"
                b_tail = f"{b['tail_oh_us']:.1f}"
            print(f"    {r['iter']:>6d} {r['exec_us']:>12.1f} {r['latency_us']:>12.1f} {b_exec:>13s} {b_tail:>13s}")
            shown += 1
            if shown >= 15:
                print(f"    ... ({len(anomaly_iters) - shown} more)")
                break
    print()


def _find_newest_perf_file(root_device: int, after_mtime: float) -> Path | None:
    """Return the newest perf_swimlane_*_d{root_device}.json created after after_mtime."""
    if not _OUTPUTS_DIR.exists():
        return None
    pattern = f"perf_swimlane_*_d{root_device}.json"
    candidates = [
        p for p in _OUTPUTS_DIR.glob(pattern)
        if p.stat().st_mtime > after_mtime
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _trimmed_mean(values: list[float], trim_ratio: float) -> tuple[float, int]:
    """Return (trimmed_mean, effective_sample_count)."""
    if not values:
        return 0.0, 0
    s = sorted(values)
    n = len(s)
    k = int(n * trim_ratio)
    if 2 * k >= n:
        return sum(s) / n, n
    core = s[k:n - k]
    return sum(core) / len(core), len(core)


def _parse_gather_stats(perf_file: Path, trim_ratio: float) -> dict | None:
    """
    Parse gather task stats from the raw perf JSON.
    Keep the last (N_ITER - N_WARMUP) gather samples by dispatch order, then
    compute trimmed means by removing trim_ratio at both tails.

    Returns a dict with keys:
        func_id, func_name, total_count, measured_count, trimmed_count,
        trim_ratio, avg_exec_us, avg_latency_us
    or None if no gather tasks were found.
    """
    with open(perf_file, encoding="utf-8") as f:
        data = json.load(f)

    tasks = data.get("tasks", [])

    gather_tasks = [
        t for t in tasks
        if t.get("func_id") in (FUNC_ID_GATHER_SYNC, FUNC_ID_GATHER_ASYNC)
    ]
    if not gather_tasks:
        return None

    # Sort chronologically by dispatch time (fall back to start time)
    gather_tasks.sort(key=lambda t: t.get("dispatch_time_us", t.get("start_time_us", 0)))

    total_count = len(gather_tasks)
    measured = gather_tasks[N_WARMUP:]
    if not measured:
        measured = gather_tasks  # fewer than N_WARMUP samples; use all

    exec_vals    = [t["duration_us"] for t in measured]
    latency_vals = [
        t["finish_time_us"] - t["dispatch_time_us"]
        for t in measured
        if "finish_time_us" in t and "dispatch_time_us" in t
    ]

    avg_exec, trimmed_exec_count = _trimmed_mean(exec_vals, trim_ratio)
    if latency_vals:
        avg_latency, trimmed_lat_count = _trimmed_mean(latency_vals, trim_ratio)
    else:
        avg_latency, trimmed_lat_count = avg_exec, trimmed_exec_count
    trimmed_count = min(trimmed_exec_count, trimmed_lat_count)

    # Determine which gather kernel was used (most frequent in measured window)
    func_ids = [t["func_id"] for t in measured]
    func_id  = max(set(func_ids), key=func_ids.count)
    func_name = "GatherSync" if func_id == FUNC_ID_GATHER_SYNC else "GatherAsync"

    return {
        "func_id":         func_id,
        "func_name":       func_name,
        "total_count":     total_count,
        "measured_count":  len(measured),
        "trimmed_count":   trimmed_count,
        "trim_ratio":      trim_ratio,
        "avg_exec_us":     avg_exec,
        "avg_latency_us":  avg_latency,
    }


def run_case(
    strategy: str,
    size_label: str,
    n_ranks: int,
    first_device: int,
    platform: str,
    verbose: bool,
    trim_ratio: float,
    kernels_dir: Path,
    golden_path: Path,
    extreme_mode: bool,
    diagnostic: bool = False,
) -> tuple[dict, list[dict]]:
    """Run one (strategy, size) case and return result dict."""
    total_bytes   = SIZE_BYTES[size_label]
    gather_count  = _gather_count_for(total_bytes, n_ranks)
    root_device   = first_device  # rank 0 = first_device

    print(f"  [{strategy:6s}] {size_label:5s}  gather_count={gather_count:6d} ...", end="", flush=True)

    env = os.environ.copy()
    env["GATHER_COUNT"]    = str(gather_count)
    env["GATHER_STRATEGY"] = strategy
    env["N_DEVICES"]       = str(n_ranks)
    env["FIRST_DEVICE"]    = str(first_device)
    env["N_ITER"]          = str(N_ITER)
    if extreme_mode:
        env["GATHER_CASE"] = "extreme"

    cmd = [
        sys.executable, str(_RUNNER),
        "-k", str(kernels_dir),
        "-g", str(golden_path),
        "--n-devices", str(n_ranks),
        "--first-device", str(first_device),
        "-p", platform,
        "--enable-profiling",
        "--silent",
    ]

    before_mtime = time.time()
    proc = subprocess.run(cmd, env=env, capture_output=not verbose, text=True)

    result = {
        "strategy":        strategy,
        "size":            size_label,
        "total_bytes":     total_bytes,
        "gather_count":    gather_count,
        "n_ranks":         n_ranks,
        "n_iter":          N_ITER,
        "warmup_samples":  N_WARMUP,
        "measured_samples": N_ITER - N_WARMUP,
        "trim_ratio":      trim_ratio,
        "extreme_mode":    extreme_mode,
        "run_ok":          proc.returncode == 0,
        "func_id":         None,
        "func_name":       None,
        "total_count":     None,
        "trimmed_count":   None,
        "avg_exec_us":     None,
        "avg_latency_us":  None,
    }

    if not result["run_ok"]:
        print(f"  FAILED (exit {proc.returncode})")
        if verbose and proc.stderr:
            print(proc.stderr[-800:])
        return result, []

    # Find profiling file
    perf_file = None
    for _ in range(10):          # wait up to ~5s for async profiling write
        perf_file = _find_newest_perf_file(root_device, before_mtime)
        if perf_file:
            break
        time.sleep(0.5)

    if not perf_file:
        print("  WARN: no perf file found")
        return result, []

    stats = _parse_gather_stats(perf_file, trim_ratio=trim_ratio)

    if stats:
        result.update({
            "func_id":         stats["func_id"],
            "func_name":       stats["func_name"],
            "total_count":     stats["total_count"],
            "trimmed_count":   stats["trimmed_count"],
            "avg_exec_us":     stats["avg_exec_us"],
            "avg_latency_us":  stats["avg_latency_us"],
        })
        kernel_tag = "GatherSync→MTE" if stats["func_id"] == FUNC_ID_GATHER_SYNC else "GatherAsync→SDMA"
        measured   = stats["measured_count"]
        trimmed    = stats["trimmed_count"]
        print(f"  exec={stats['avg_exec_us']:.1f}us  lat={stats['avg_latency_us']:.1f}us"
              f"  [n={measured}, trimmed={trimmed}]  [{kernel_tag}]")
    else:
        print("  WARN: gather stats not found in perf JSON")
        if verbose:
            print(f"  (perf file: {perf_file})")

    # Diagnostic: per-iteration breakdown
    detail_rows = []
    if diagnostic:
        gather_rows, barrier_rows = _parse_gather_detail(perf_file)
        _print_diagnostic_report(strategy, size_label, gather_rows, barrier_rows)
        for r in gather_rows:
            # Find paired barrier row
            b_idx = r["iter"]  # barrier index offset: +1 for startup barrier
            b_exec = None
            b_tail = None
            if b_idx + 1 < len(barrier_rows):
                b = barrier_rows[b_idx + 1]
                b_exec = b["exec_us"]
                b_tail = b["tail_oh_us"]
            detail_rows.append({
                "strategy": strategy,
                "size": size_label,
                **r,
                "barrier_exec_us": b_exec,
                "barrier_tail_oh_us": b_tail,
            })

    return result, detail_rows


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def print_summary(results: list[dict]) -> None:
    header = (
        f"{'Strategy':8s}  {'Size':5s}  {'GATHER_COUNT':12s}  "
        f"{'Avg_Exec(us)':>13s}  {'Avg_Lat(us)':>12s}  {'Actual_Kernel'}"
    )
    sep = "-" * len(header)
    print()
    print("=" * len(header))
    if results:
        n_iter = results[0].get("n_iter", N_ITER)
        warmup = results[0].get("warmup_samples", N_WARMUP)
        trim_ratio = float(results[0].get("trim_ratio", TRIM_RATIO_DEFAULT))
    else:
        n_iter = N_ITER
        warmup = N_WARMUP
        trim_ratio = TRIM_RATIO_DEFAULT
    measured = n_iter - warmup
    print(
        "Gather Performance Summary  "
        f"(avg of last {measured} / {n_iter} iterations, trim={trim_ratio * 100:.1f}% each tail)"
    )
    print("=" * len(header))
    print(header)
    print(sep)

    for r in results:
        exec_str = f"{r['avg_exec_us']:.2f}" if r["avg_exec_us"] is not None else "N/A"
        lat_str  = f"{r['avg_latency_us']:.2f}" if r["avg_latency_us"] is not None else "N/A"
        if r["func_name"]:
            kernel_tag = (
                f"{r['func_name']} (→MTE)"  if r["func_id"] == FUNC_ID_GATHER_SYNC
                else f"{r['func_name']} (→SDMA)"
            )
        else:
            kernel_tag = "FAILED" if not r["run_ok"] else "N/A"

        print(
            f"{r['strategy']:8s}  {r['size']:5s}  {r['gather_count']:12d}  "
            f"{exec_str:>13s}  {lat_str:>12s}  {kernel_tag}"
        )

    print("=" * len(header))


def print_wait_status_stats(results: list[dict]) -> None:
    """Print SDMA wait status statistics read from per-case JSON files written by golden.py."""
    sdma_results = [r for r in results if r.get("func_id") == FUNC_ID_GATHER_ASYNC and r.get("run_ok")]
    if not sdma_results:
        return

    header = f"  {'Size':5s}  {'n_ranks':7s}  {'success':>10s}  {'fail':>10s}  {'samples':>10s}"
    sep = "-" * (len(header))
    print()
    print("=" * len(header))
    if results:
        n_iter = results[0].get("n_iter", N_ITER)
        warmup = results[0].get("warmup_samples", N_WARMUP)
    else:
        n_iter = N_ITER
        warmup = N_WARMUP
    measured = n_iter - warmup
    print(f"SDMA Wait Status Statistics  (ranks combined, last {measured} of {n_iter} iterations)")
    print("=" * len(header))
    print(header)
    print(sep)

    for r in sdma_results:
        prefix = "poll_counts_extreme" if r.get("extreme_mode") else "poll_counts"
        fname = _OUTPUTS_DIR / f"{prefix}_sdma_gc{r['gather_count']}_r{r['n_ranks']}.json"
        if not fname.exists():
            print(f"  {r['size']:5s}  (no poll data)")
            continue

        with open(fname) as f:
            data = json.load(f)

        poll = data["poll_counts"]   # list[list[int]]: [N_ITER][n_ranks]
        measured = poll[N_WARMUP:]   # discard warm-up
        all_vals = [v for row in measured for v in row]

        if not all_vals:
            continue

        total = len(all_vals)
        success = sum(1 for v in all_vals if int(v) != 0)
        fail = total - success
        success_pct = success / total * 100
        fail_pct = fail / total * 100

        print(
            f"  {r['size']:5s}  {r['n_ranks']:7d}  "
            f"{success_pct:>9.1f}%  {fail_pct:>9.1f}%  {total:>10d}"
        )

    print("=" * len(header))


def save_csv(results: list[dict], out_path: Path) -> None:
    fields = [
        "strategy", "size", "total_bytes", "gather_count", "n_ranks",
        "n_iter", "warmup_samples", "measured_samples", "trim_ratio", "extreme_mode",
        "run_ok", "func_id", "func_name", "total_count", "trimmed_count", "avg_exec_us", "avg_latency_us",
    ]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    print(f"\nCSV saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Batch benchmark: fake_kernel_comm_sched gather performance",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python tools/bench_gather_comm_sched.py --platform a2a3 --first-device 4
  python tools/bench_gather_comm_sched.py --platform a2a3sim
  python tools/bench_gather_comm_sched.py --strategies mte sdma --sizes 4K 64K
        """,
    )
    parser.add_argument("-p", "--platform", default="a2a3",
                        choices=["a2a3", "a2a3sim"],
                        help="Platform (default: a2a3)")
    parser.add_argument("--first-device", type=int, default=0,
                        help="First device ID (default: 0)")
    parser.add_argument("--n-devices", type=int, default=4,
                        help="Number of devices/ranks (default: 4)")
    parser.add_argument("--strategies", nargs="+", default=["mte", "sdma"],
                        choices=STRATEGY_NAMES,
                        help="Strategies to test (default: mte sdma)")
    parser.add_argument("--sizes", nargs="+", default=SIZE_LABELS,
                        choices=list(SIZE_BYTES.keys()),
                        help="Total comm sizes to test (default: all)")
    parser.add_argument(
        "--extreme-case",
        action="store_true",
        help="Run fake_kernel_comm_sched_extreme (single AICPU + same-AICore dual AIV stress case)",
    )
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Show subprocess output")
    parser.add_argument("--diagnostic", action="store_true",
                        help="Enable per-iteration timing breakdown and anomaly analysis")
    parser.add_argument(
        "--show-sdma-wait-stats",
        action="store_true",
        help="Show SDMA wait status statistics from debug JSON (default: off)",
    )
    parser.add_argument(
        "--trim-ratio",
        type=float,
        default=TRIM_RATIO_DEFAULT,
        help="Trim ratio per tail for mean stats after warm-up (default: 0.10)",
    )
    parser.add_argument("--output-dir", type=Path, default=_OUTPUTS_DIR,
                        help="Output directory for CSV (default: outputs/)")

    args = parser.parse_args()

    if args.trim_ratio < 0 or args.trim_ratio >= 0.5:
        parser.error("--trim-ratio must be in [0.0, 0.5)")

    case_name = "fake_kernel_comm_sched_extreme" if args.extreme_case else "fake_kernel_comm_sched"
    kernels_dir = _EXTREME_KERNELS_DIR if args.extreme_case else _NORMAL_KERNELS_DIR
    golden_path = _EXTREME_GOLDEN_PATH if args.extreme_case else _NORMAL_GOLDEN_PATH
    print(f"Benchmark: {case_name} gather")
    print(f"  platform={args.platform}  first_device={args.first_device}  n_devices={args.n_devices}")
    print(f"  strategies={args.strategies}")
    print(f"  sizes={args.sizes}")
    print(
        f"  iterations per case: {N_ITER} total, {N_WARMUP} warm-up, "
        f"{N_ITER - N_WARMUP} measured, trim={args.trim_ratio * 100:.1f}% per tail"
    )
    print()

    results = []
    all_detail_rows = []
    for strategy in args.strategies:
        print(f"--- Strategy: {strategy.upper()} ---")
        for size_label in args.sizes:
            r, detail = run_case(
                strategy=strategy,
                size_label=size_label,
                n_ranks=args.n_devices,
                first_device=args.first_device,
                platform=args.platform,
                verbose=args.verbose,
                trim_ratio=args.trim_ratio,
                kernels_dir=kernels_dir,
                golden_path=golden_path,
                extreme_mode=args.extreme_case,
                diagnostic=args.diagnostic,
            )
            results.append(r)
            all_detail_rows.extend(detail)
        print()

    print_summary(results)
    if args.show_sdma_wait_stats:
        print_wait_status_stats(results)

    # Save CSV
    args.output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = args.output_dir / f"bench_gather_{ts}.csv"
    save_csv(results, csv_path)

    # Save per-iteration detail CSV when diagnostic is enabled
    if args.diagnostic and all_detail_rows:
        detail_fields = [
            "strategy", "size", "iter", "func_id",
            "dispatch_time_us", "start_time_us", "end_time_us", "finish_time_us",
            "head_oh_us", "exec_us", "tail_oh_us", "latency_us",
            "barrier_exec_us", "barrier_tail_oh_us",
        ]
        detail_path = args.output_dir / f"bench_gather_detail_{ts}.csv"
        with open(detail_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=detail_fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(all_detail_rows)
        print(f"Detail CSV saved: {detail_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
