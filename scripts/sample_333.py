#!/usr/bin/env python3
"""Sample 333 tasks from CyberGym dataset, stratified by project_name."""

import json
import random
import math
from collections import Counter
from pathlib import Path

SEED = 42
TARGET = 333
DATASET_PATH = Path(
    "/inspire/hdd/project/multi-agent/niexiaohang-25130061/holos_synergy_experiments/ex_cybergym/cybergym_dataset.json"
)
OUTPUT_PATH = Path(
    "/inspire/hdd/project/multi-agent/niexiaohang-25130061/holos_synergy_experiments/ex_cybergym/configs/sampled_333.txt"
)

random.seed(SEED)

# 1. Load and group by project_name
with open(DATASET_PATH) as f:
    dataset = json.load(f)

total = len(dataset)
print(f"Total tasks: {total}")

groups: dict[str, list[str]] = {}
for task_id, info in dataset.items():
    proj = info["project_name"]
    groups.setdefault(proj, []).append(task_id)

print(f"Total projects: {len(groups)}")

# Shuffle each group for reproducibility
for task_ids in groups.values():
    random.shuffle(task_ids)

# 2. Compute proportional allocation: max(1, round(count * 333/1507))
ratio = TARGET / total
allocations: dict[str, int] = {}
for proj, task_ids in groups.items():
    allocations[proj] = max(1, round(len(task_ids) * ratio))

current_total = sum(allocations.values())
print(f"Initial allocation total: {current_total}")

# 3. Adjust to hit exactly TARGET
if current_total > TARGET:
    # Trim from largest groups (those with allocation > 1), one at a time
    overshoot = current_total - TARGET
    # Sort projects by allocation descending, break ties by name for determinism
    sorted_projs = sorted(allocations.keys(), key=lambda p: (-allocations[p], p))
    idx = 0
    while overshoot > 0:
        proj = sorted_projs[idx % len(sorted_projs)]
        if allocations[proj] > 1:
            allocations[proj] -= 1
            overshoot -= 1
        idx += 1
        # Safety: if we cycled through all and can't reduce, break
        if idx > len(sorted_projs) * TARGET:
            break
elif current_total < TARGET:
    # Add extras randomly from projects that have room
    undershoot = TARGET - current_total
    eligible = [p for p in groups if allocations[p] < len(groups[p])]
    random.shuffle(eligible)
    idx = 0
    while undershoot > 0 and eligible:
        proj = eligible[idx % len(eligible)]
        if allocations[proj] < len(groups[proj]):
            allocations[proj] += 1
            undershoot -= 1
        else:
            eligible.remove(proj)
            if not eligible:
                break
            continue
        idx += 1

final_total = sum(allocations.values())
assert final_total == TARGET, f"Allocation mismatch: {final_total} != {TARGET}"

# 4. Sample from each group
sampled: list[str] = []
for proj, task_ids in groups.items():
    n = allocations[proj]
    sampled.extend(task_ids[:n])  # Already shuffled

assert len(sampled) == TARGET, f"Sampled {len(sampled)} != {TARGET}"

# Sort for deterministic output
sampled.sort()

# 5. Write output
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(OUTPUT_PATH, "w") as f:
    for task_id in sampled:
        f.write(task_id + "\n")

print(f"\nWrote {len(sampled)} task IDs to {OUTPUT_PATH}")

# 6. Summary
prefix_counts = Counter(tid.split(":")[0] for tid in sampled)
print(f"\nBreakdown by prefix:")
for prefix, count in sorted(prefix_counts.items()):
    print(f"  {prefix}: {count}")

project_counts = Counter(dataset[tid]["project_name"] for tid in sampled)
print(f"\nTop 10 projects in sample:")
for proj, count in project_counts.most_common(10):
    orig = len(groups[proj])
    print(f"  {proj}: {count} sampled (of {orig} total)")
