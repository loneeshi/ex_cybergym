import json
import random
from collections import defaultdict
import argparse
from pathlib import Path


def create_subset(
    dataset_path: str,
    successful_tasks_path: str,
    out_path: str,
    target_size: int = 300,
    seed: int = 42,
):
    random.seed(seed)

    # 1. Load full dataset
    with open(dataset_path, "r") as f:
        full_data = json.load(f)

    print(f"Loaded {len(full_data)} total tasks from {dataset_path}")

    # 2. Load successful tasks (these MUST be included)
    with open(successful_tasks_path, "r") as f:
        successful_tasks = set(json.load(f))

    # Verify successful tasks exist in full data
    successful_tasks = {t for t in successful_tasks if t in full_data}
    print(f"Including {len(successful_tasks)} previously successful tasks.")

    # 3. Categorize remaining tasks for stratified sampling
    # Stratify by (language, crash_type)
    remaining_tasks = []
    strata = defaultdict(list)

    for task_id, info in full_data.items():
        if task_id in successful_tasks:
            continue

        lang = info.get("project_language", "unknown")
        # Simplify crash type for grouping
        raw_crash = info.get("crash_type", "unknown")

        # Basic grouping to avoid too many small strata
        if "Heap-buffer-overflow" in raw_crash:
            c_type = "heap_overflow"
        elif "Stack-buffer-overflow" in raw_crash:
            c_type = "stack_overflow"
        elif "Null-dereference" in raw_crash:
            c_type = "null_deref"
        elif "Use-after-free" in raw_crash:
            c_type = "uaf"
        elif "Assertion-failure" in raw_crash or "assert" in raw_crash.lower():
            c_type = "assert"
        elif "Timeout" in raw_crash:
            c_type = "timeout"
        elif "OOM" in raw_crash or "Out-of-memory" in raw_crash:
            c_type = "oom"
        elif "Leak" in raw_crash:
            c_type = "leak"
        else:
            c_type = "other"

        stratum_key = f"{lang}::{c_type}"
        strata[stratum_key].append(task_id)
        remaining_tasks.append(task_id)

    # 4. Calculate how many we need from each stratum
    needed = target_size - len(successful_tasks)
    print(
        f"Need to sample {needed} more tasks from {len(remaining_tasks)} remaining tasks."
    )

    selected_remaining = []
    for stratum_key, tasks in strata.items():
        # Proportional allocation
        proportion = len(tasks) / len(remaining_tasks)
        num_to_sample = int(round(proportion * needed))

        if num_to_sample > 0:
            sampled = random.sample(tasks, min(num_to_sample, len(tasks)))
            selected_remaining.extend(sampled)

    # Handle rounding errors (might be slightly over or under needed)
    diff = needed - len(selected_remaining)
    if diff > 0:
        # Need a few more, pick randomly from unselected
        unselected = list(set(remaining_tasks) - set(selected_remaining))
        selected_remaining.extend(random.sample(unselected, diff))
    elif diff < 0:
        # Have a few too many, remove randomly
        to_remove = random.sample(selected_remaining, abs(diff))
        for t in to_remove:
            selected_remaining.remove(t)

    # 5. Combine and save
    final_subset = sorted(list(successful_tasks) + selected_remaining)

    print(f"\nFinal subset size: {len(final_subset)}")
    print("Stratification breakdown of the NEWly sampled tasks:")
    breakdown = defaultdict(int)
    for t in selected_remaining:
        lang = full_data[t].get("project_language", "unknown")
        breakdown[lang] += 1

    for lang, count in sorted(breakdown.items(), key=lambda x: -x[1]):
        print(f"  {lang}: {count} tasks")

    with open(out_path, "w") as f:
        json.dump(final_subset, f, indent=2)

    print(f"\nSubset saved to {out_path}")
    print(
        "You can run evolution on this subset by adding: --subset-file cybergym_subset_300.json"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="cybergym_dataset.json")
    parser.add_argument("--success-file", default="successful_tasks_r1.json")
    parser.add_argument("--out", default="cybergym_subset_300.json")
    parser.add_argument("--size", type=int, default=300)
    args = parser.parse_args()

    create_subset(args.dataset, args.success_file, args.out, args.size)
