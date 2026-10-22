#!/usr/bin/env python3

import argparse
import csv
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize OpenVINO profiling CSV files")
    parser.add_argument("profiles", nargs="+", type=Path)
    parser.add_argument("--limit", type=int, default=15, help="rows shown per table")
    parser.add_argument("--gaps", type=int, default=0, metavar="N", help="show the N largest device timeline gaps")
    return parser.parse_args()


def print_group(title: str, values: dict[str, list[int]], total_us: int, limit: int) -> None:
    print(f"\n{title}")
    print(f"{'device ms':>10}  {'count':>7}  {'share':>7}  name")
    for name, (time_us, count) in sorted(values.items(), key=lambda item: item[1][0], reverse=True)[:limit]:
        share = 100.0 * time_us / total_us if total_us else 0.0
        print(f"{time_us / 1000:10.3f}  {count:7d}  {share:6.1f}%  {name}")


def print_gaps(rows: list[dict[str, str]], limit: int) -> None:
    timeline = sorted(
        (row for row in rows if int(row["start_time_us"]) > 0), key=lambda row: int(row["start_time_us"])
    )
    if not timeline:
        return

    gaps: list[tuple[int, dict[str, str]]] = []
    completed_until = int(timeline[0]["start_time_us"]) + int(timeline[0]["real_time_us"])
    for row in timeline[1:]:
        start_us = int(row["start_time_us"])
        if start_us > completed_until:
            gaps.append((start_us - completed_until, row))
        completed_until = max(completed_until, start_us + int(row["real_time_us"]))

    print("\nLargest device timeline gaps before node")
    print("Profiling adds synchronization; a gap marks a boundary and is not time caused by the following node.")
    for gap_us, row in sorted(gaps, key=lambda item: item[0], reverse=True)[:limit]:
        print(f'{gap_us:8d} us  {row["node_type"]:24} {row["node_name"]} | {row["exec_type"]}')


def analyze(path: Path, limit: int, gap_limit: int) -> None:
    by_type: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    by_impl: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    by_node: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    executed_rows: list[dict[str, str]] = []
    total_real_us = 0
    total_cpu_us = 0
    executed = 0

    with path.open(newline="") as profile_file:
        for row in csv.DictReader(profile_file):
            if row["status"] != "EXECUTED":
                continue
            executed_rows.append(row)
            real_us = int(row["real_time_us"])
            total_real_us += real_us
            total_cpu_us += int(row["cpu_time_us"])
            executed += 1
            for values, key in (
                (by_type, row["node_type"]),
                (by_impl, row["exec_type"]),
                (by_node, f'{row["node_type"]}: {row["node_name"]}'),
            ):
                values[key][0] += real_us
                values[key][1] += 1

    print(f"\n== {path} ==")
    print(f"executed nodes: {executed}")
    print(f"summed device time: {total_real_us / 1000:.3f} ms")
    print(f"summed CPU time: {total_cpu_us / 1000:.3f} ms")
    print_group("Node types", by_type, total_real_us, limit)
    print_group("Implementations", by_impl, total_real_us, limit)
    print_group("Individual nodes", by_node, total_real_us, limit)
    if gap_limit > 0:
        print_gaps(executed_rows, gap_limit)


def main() -> None:
    args = parse_args()
    for profile in args.profiles:
        analyze(profile, args.limit, args.gaps)


if __name__ == "__main__":
    main()