#!/usr/bin/env python3
"""
Preset batch benchmark for fake_kernel_comm_sched.

This script reuses the single-case execution/statistics logic in
tools/bench_gather_comm_sched.py, but fixes the benchmark target to the
normal fake_kernel_comm_sched case and provides defaults that match the
common scheduler comparison workflow:

- strategies/modes: mte, sdma, hybrid
- sizes: 1K .. 4M
- case: fake_kernel_comm_sched
- hybrid threshold: fixed at 512KB in scheduler

It can optionally repeat the whole sweep for multiple rounds so the user can
observe cross-round stability while keeping one consolidated CSV.

Examples:
    python tools/bench_fake_kernel_comm_sched.py --platform a2a3 --first-device 4
    python tools/bench_fake_kernel_comm_sched.py --platform a2a3sim --rounds 3
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from datetime import datetime
from pathlib import Path

import bench_gather_comm_sched as base


DEFAULT_STRATEGIES = ["mte", "sdma", "hybrid"]
DEFAULT_SIZES = [
    "1K", "2K", "4K", "8K", "16K", "32K", "64K",
    "128K", "256K", "512K", "1M", "2M", "4M",
]


def _validate_rounds(value: str) -> int:
    rounds = int(value)
    if rounds < 1:
        raise argparse.ArgumentTypeError("--rounds must be >= 1")
    return rounds


def _round_header(round_idx: int, rounds: int) -> None:
    banner = f" Round {round_idx}/{rounds} "
    print()
    print("=" * len(banner))
    print(banner)
    print("=" * len(banner))


def _print_overall_summary(
    results: list[dict],
    rounds: int,
    strategies: list[str],
    sizes: list[str],
) -> None:
    if rounds <= 1 or not results:
        return

    print()
    print("=" * 93)
    print("Overall Round Summary (mean of per-round metrics)")
    print("=" * 93)
    print(
        f"{'Strategy':8s}  {'Size':5s}  {'Success':>7s}  "
        f"{'Mean_Exec(us)':>13s}  {'Mean_Lat(us)':>12s}"
    )
    print("-" * 93)

    for strategy in strategies:
        for size in sizes:
            rows = [
                r for r in results
                if r.get("strategy") == strategy and r.get("size") == size and r.get("run_ok")
            ]
            if not rows:
                continue

            exec_vals = [r["avg_exec_us"] for r in rows if r.get("avg_exec_us") is not None]
            lat_vals = [r["avg_latency_us"] for r in rows if r.get("avg_latency_us") is not None]
            mean_exec = sum(exec_vals) / len(exec_vals) if exec_vals else None
            mean_lat = sum(lat_vals) / len(lat_vals) if lat_vals else None
            exec_str = f"{mean_exec:.2f}" if mean_exec is not None else "N/A"
            lat_str = f"{mean_lat:.2f}" if mean_lat is not None else "N/A"
            print(f"{strategy:8s}  {size:5s}  {len(rows):7d}  {exec_str:>13s}  {lat_str:>12s}")

    print("=" * 93)


def _save_csv(results: list[dict], out_path: Path) -> None:
    fields = [
        "round",
        "strategy", "size", "total_bytes", "gather_count", "n_ranks",
        "n_iter", "warmup_samples", "measured_samples", "trim_ratio", "case_mode",
        "dummy_comm_bytes", "serialize_dummy", "profile_root_only",
        "run_ok", "attempts", "last_error",
        "func_id", "func_name", "total_count", "trimmed_count", "avg_exec_us", "avg_latency_us",
        "perf_file", "covered_ok", "covered_file",
    ]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    print(f"\nCSV saved: {out_path}")


def _save_detail_csv(detail_rows: list[dict], out_path: Path) -> None:
    fields = [
        "round", "strategy", "size", "iter", "func_id",
        "dispatch_time_us", "start_time_us", "end_time_us", "finish_time_us",
        "head_oh_us", "exec_us", "tail_oh_us", "latency_us",
        "barrier_exec_us", "barrier_tail_oh_us",
    ]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(detail_rows)
    print(f"Detail CSV saved: {out_path}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Preset benchmark sweep for fake_kernel_comm_sched",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python tools/bench_fake_kernel_comm_sched.py --platform a2a3 --first-device 4
  python tools/bench_fake_kernel_comm_sched.py --platform a2a3sim --rounds 3
  python tools/bench_fake_kernel_comm_sched.py --sizes 64K 256K 1M --modes hybrid
        """,
    )
    parser.add_argument(
        "-p", "--platform",
        default="a2a3",
        choices=["a2a3", "a2a3sim"],
        help="Platform (default: a2a3)",
    )
    parser.add_argument(
        "--first-device",
        type=int,
        default=0,
        help="First device ID (default: 0)",
    )
    parser.add_argument(
        "--n-devices",
        type=int,
        default=4,
        help="Number of devices/ranks (default: 4)",
    )
    parser.add_argument(
        "--strategies", "--modes",
        nargs="+",
        default=DEFAULT_STRATEGIES,
        choices=base.STRATEGY_NAMES,
        help="Strategies/modes to test (default: mte sdma hybrid)",
    )
    parser.add_argument(
        "--sizes",
        nargs="+",
        default=DEFAULT_SIZES,
        choices=list(base.SIZE_BYTES.keys()),
        help="Total comm sizes to test (default: 1K..4M)",
    )
    parser.add_argument(
        "--rounds",
        type=_validate_rounds,
        default=1,
        help="Repeat the whole strategy/size sweep this many times (default: 1)",
    )
    parser.add_argument(
        "--round-interval-s",
        type=float,
        default=0.0,
        help="Sleep this many seconds between rounds (default: 0)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Show subprocess output",
    )
    parser.add_argument(
        "--diagnostic",
        action="store_true",
        help="Enable per-iteration timing breakdown and anomaly analysis",
    )
    parser.add_argument(
        "--show-sdma-wait-stats",
        action="store_true",
        help="Show SDMA wait status statistics from debug JSON (default: off)",
    )
    parser.add_argument(
        "--trim-ratio",
        type=float,
        default=base.TRIM_RATIO_DEFAULT,
        help="Trim ratio per tail for mean stats after warm-up (default: 0.10)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=base._OUTPUTS_DIR,
        help="Output directory for CSV (default: outputs/)",
    )

    args = parser.parse_args()

    if args.trim_ratio < 0 or args.trim_ratio >= 0.5:
        parser.error("--trim-ratio must be in [0.0, 0.5)")
    if args.round_interval_s < 0:
        parser.error("--round-interval-s must be >= 0")

    base._cleanup_output_files(args.verbose)

    print("Benchmark: fake_kernel_comm_sched")
    print(f"  platform={args.platform}  first_device={args.first_device}  n_devices={args.n_devices}")
    print(f"  strategies={args.strategies}")
    print(f"  sizes={args.sizes}")
    print("  hybrid threshold=512KB (fixed in scheduler)")
    print(f"  rounds={args.rounds}  round_interval_s={args.round_interval_s}")
    print(
        f"  iterations per case: {base.N_ITER} total, {base.N_WARMUP} warm-up, "
        f"{base.N_ITER - base.N_WARMUP} measured, trim={args.trim_ratio * 100:.1f}% per tail"
    )

    all_results: list[dict] = []
    all_detail_rows: list[dict] = []

    for round_idx in range(1, args.rounds + 1):
        _round_header(round_idx, args.rounds)

        round_results: list[dict] = []
        for strategy in args.strategies:
            print(f"--- Strategy: {strategy.upper()} ---")
            for size_label in args.sizes:
                result, detail_rows = base.run_case(
                    strategy=strategy,
                    size_label=size_label,
                    n_ranks=args.n_devices,
                    first_device=args.first_device,
                    platform=args.platform,
                    verbose=args.verbose,
                    trim_ratio=args.trim_ratio,
                    kernels_dir=base._NORMAL_KERNELS_DIR,
                    golden_path=base._NORMAL_GOLDEN_PATH,
                    case_mode="normal",
                    dummy_comm_bytes=0,
                    serialize_dummy=0,
                    profile_root_only=1,
                    diagnostic=args.diagnostic,
                )
                result["round"] = round_idx
                round_results.append(result)
                all_results.append(result)

                for row in detail_rows:
                    row["round"] = round_idx
                    all_detail_rows.append(row)
            print()

        base.print_summary(round_results)
        if args.show_sdma_wait_stats:
            base.print_wait_status_stats(round_results)

        if round_idx < args.rounds and args.round_interval_s > 0:
            print(f"\nSleeping {args.round_interval_s:.1f}s before next round...")
            time.sleep(args.round_interval_s)

    _print_overall_summary(all_results, args.rounds, args.strategies, args.sizes)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = args.output_dir / f"bench_fake_kernel_comm_sched_{ts}.csv"
    _save_csv(all_results, csv_path)

    if args.diagnostic and all_detail_rows:
        detail_path = args.output_dir / f"bench_fake_kernel_comm_sched_detail_{ts}.csv"
        _save_detail_csv(all_detail_rows, detail_path)

    base._apply_output_permissions(args.verbose)
    return 0


if __name__ == "__main__":
    sys.exit(main())
