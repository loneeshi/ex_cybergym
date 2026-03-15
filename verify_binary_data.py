#!/usr/bin/env python3
"""
Verify CyberGym binary-only server data completeness.

Checks that the extracted cybergym-server-data directory has all required
files for each task in the dataset. Also checks which runner images are
needed (some tasks specify non-default runners).

Usage:
    python3 verify_binary_data.py /path/to/cybergym-server-data [--dataset datasets.json]
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path


def check_arvo_task(data_dir: Path, task_num: str) -> dict:
    """Check if an arvo task has all required binary files."""
    issues = []
    runners = set()

    for mode in ("vul", "fix"):
        mode_dir = data_dir / "arvo" / task_num / mode
        if not mode_dir.exists():
            issues.append(f"missing {mode}/ directory")
            continue

        arvo_bin = mode_dir / "arvo"
        if not arvo_bin.exists():
            issues.append(f"missing {mode}/arvo binary")

        libs_dir = mode_dir / "libs"
        if not libs_dir.exists():
            issues.append(f"missing {mode}/libs/")

        out_dir = mode_dir / "out"
        if not out_dir.exists():
            issues.append(f"missing {mode}/out/")

        runner_file = mode_dir / "runner"
        if runner_file.exists():
            runner = runner_file.read_text().strip()
            runners.add(runner)

    return {"issues": issues, "runners": runners}


def check_ossfuzz_task(data_dir: Path, task_num: str) -> dict:
    """Check if an oss-fuzz task has all required binary files."""
    issues = []

    for mode in ("vul", "fix"):
        mode_dir = data_dir / "oss-fuzz" / task_num / mode
        if not mode_dir.exists():
            issues.append(f"missing {mode}/ directory")
            continue

        meta_file = mode_dir / "metadata.json"
        if not meta_file.exists():
            issues.append(f"missing {mode}/metadata.json")

        out_dir = mode_dir / "out"
        if not out_dir.exists():
            issues.append(f"missing {mode}/out/")

    return {"issues": issues, "runners": set()}


def main():
    parser = argparse.ArgumentParser(description="Verify CyberGym binary data")
    parser.add_argument("data_dir", help="Path to cybergym-server-data directory")
    parser.add_argument(
        "--dataset",
        "-d",
        default=None,
        help="Path to datasets.json (to cross-check task coverage)",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"ERROR: {data_dir} does not exist")
        sys.exit(1)

    print(f"Checking binary data in: {data_dir}")
    print()

    # Scan what's in the data directory
    arvo_dir = data_dir / "arvo"
    ossfuzz_dir = data_dir / "oss-fuzz"

    arvo_tasks = (
        sorted(arvo_dir.iterdir(), key=lambda p: int(p.name))
        if arvo_dir.exists()
        else []
    )
    ossfuzz_tasks = (
        sorted(ossfuzz_dir.iterdir(), key=lambda p: int(p.name))
        if ossfuzz_dir.exists()
        else []
    )

    print(f"Found {len(arvo_tasks)} arvo tasks, {len(ossfuzz_tasks)} oss-fuzz tasks")
    print()

    # Check each task
    all_runners = Counter()
    broken_tasks = []
    ok_count = 0

    for task_dir in arvo_tasks:
        result = check_arvo_task(data_dir, task_dir.name)
        for r in result["runners"]:
            all_runners[r] += 1
        if result["issues"]:
            broken_tasks.append((f"arvo:{task_dir.name}", result["issues"]))
        else:
            ok_count += 1

    for task_dir in ossfuzz_tasks:
        result = check_ossfuzz_task(data_dir, task_dir.name)
        if result["issues"]:
            broken_tasks.append((f"oss-fuzz:{task_dir.name}", result["issues"]))
        else:
            ok_count += 1

    total = len(arvo_tasks) + len(ossfuzz_tasks)
    print(f"Results: {ok_count}/{total} tasks OK, {len(broken_tasks)} with issues")

    if broken_tasks:
        print(f"\nBroken tasks (showing first 20):")
        for tid, issues in broken_tasks[:20]:
            print(f"  {tid}: {', '.join(issues)}")
        if len(broken_tasks) > 20:
            print(f"  ... and {len(broken_tasks) - 20} more")

    # Runner images needed
    print(f"\nRunner images needed:")
    print(f"  cybergym/oss-fuzz-base-runner:latest (default)")
    if all_runners:
        for runner, count in all_runners.most_common():
            print(f"  {runner} ({count} tasks)")
    else:
        print(f"  (no tasks specify custom runners)")

    # Cross-check with dataset
    if args.dataset:
        ds_path = Path(args.dataset)
        if ds_path.exists():
            dataset = json.loads(ds_path.read_text())
            ds_task_nums = set()
            for tid in dataset.keys():
                parts = tid.split(":")
                ds_task_nums.add((parts[0], parts[1]))

            data_task_nums = set()
            for t in arvo_tasks:
                data_task_nums.add(("arvo", t.name))
            for t in ossfuzz_tasks:
                data_task_nums.add(("oss-fuzz", t.name))

            missing = ds_task_nums - data_task_nums
            extra = data_task_nums - ds_task_nums

            print(f"\nDataset cross-check ({len(dataset)} tasks in dataset):")
            print(
                f"  Covered: {len(ds_task_nums & data_task_nums)}/{len(ds_task_nums)}"
            )
            if missing:
                print(f"  Missing from binary data: {len(missing)}")
                for subset, num in sorted(missing)[:10]:
                    print(f"    {subset}:{num}")
                if len(missing) > 10:
                    print(f"    ... and {len(missing) - 10} more")
            if extra:
                print(f"  Extra (in data but not dataset): {len(extra)}")
        else:
            print(f"\nWARNING: dataset file not found: {ds_path}")

    print()


if __name__ == "__main__":
    main()
