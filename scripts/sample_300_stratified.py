#!/usr/bin/env python3
"""
Sample 300 tasks from CyberGym dataset with stratified sampling.

Stratification dimensions:
  1. project_language (C / C++ / other)
  2. crash_type (inferred from vulnerability_description)
  3. project_name (within each stratum, proportional to project size)

This ensures the sample covers diverse vulnerability types and languages,
not just the most common ones.
"""

import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

SEED = 2026
TARGET = 300
DATASET_PATH = Path(
    "/inspire/hdd/project/multi-agent/niexiaohang-25130061/"
    "holos_synergy_experiments/ex_cybergym/cybergym_dataset.json"
)
OUTPUT_PATH = Path(
    "/inspire/hdd/project/multi-agent/niexiaohang-25130061/"
    "holos_synergy_experiments/ex_cybergym/configs/sampled_300.txt"
)

random.seed(SEED)

# ── 1. Load dataset ──────────────────────────────────────────────────────

with open(DATASET_PATH) as f:
    dataset = json.load(f)

total = len(dataset)
print(f"Total tasks in dataset: {total}")

# ── 2. Infer crash_type from vulnerability_description ────────────────────

# Ordered by specificity (more specific patterns first)
CRASH_TYPE_PATTERNS = [
    ("heap-buffer-overflow", [r"heap-buffer-overflow", r"heap buffer overflow"]),
    ("stack-buffer-overflow", [r"stack-buffer-overflow", r"stack buffer overflow"]),
    ("buffer-overflow", [r"buffer-overflow", r"buffer overflow", r"out-of-bound"]),
    ("heap-use-after-free", [r"heap-use-after-free"]),
    ("use-after-free", [r"use-after-free", r"use after free"]),
    (
        "null-dereference",
        [
            r"null\s*(pointer)?\s*(dereference|deref|access)",
            r"segv.*null",
            r"null.*segv",
            r"nullptr",
        ],
    ),
    ("segv", [r"\bsegv\b", r"segmentation fault", r"sigsegv"]),
    ("stack-overflow", [r"stack-overflow", r"stack overflow"]),
    ("integer-overflow", [r"integer-overflow", r"integer overflow", r"signed integer"]),
    ("out-of-memory", [r"out-of-memory", r"oom\b", r"alloc.*fail"]),
    ("undefined-behavior", [r"ubsan", r"undefined behavior", r"undefined-behavior"]),
    ("timeout", [r"\btimeout\b", r"hang\b"]),
    ("assertion-failure", [r"assert.*fail", r"abort\b"]),
    ("memory-leak", [r"memory.?leak", r"leak"]),
]


def infer_crash_type(vuln_desc: str) -> str:
    """Extract crash type from vulnerability description using pattern matching."""
    desc_lower = vuln_desc.lower()
    for crash_type, patterns in CRASH_TYPE_PATTERNS:
        for pattern in patterns:
            if re.search(pattern, desc_lower):
                return crash_type
    return "other"


# Annotate each task
for task_id, info in dataset.items():
    desc = info.get("vulnerability_description", "")
    info["_crash_type"] = infer_crash_type(desc)
    lang = info.get("project_language", "unknown").lower()
    if lang in ("c", "c++"):
        info["_lang_group"] = lang
    else:
        info["_lang_group"] = "other"

# ── 3. Build strata: (lang_group, crash_type) ────────────────────────────

strata: dict[tuple[str, str], list[str]] = defaultdict(list)
for task_id, info in dataset.items():
    key = (info["_lang_group"], info["_crash_type"])
    strata[key].append(task_id)

# Shuffle within each stratum
for task_ids in strata.values():
    random.shuffle(task_ids)

print(f"\nStrata (lang_group, crash_type) — {len(strata)} groups:")
for key in sorted(strata.keys()):
    print(f"  {key}: {len(strata[key])} tasks")

# ── 4. Proportional allocation across strata ──────────────────────────────

ratio = TARGET / total
allocations: dict[tuple[str, str], int] = {}
for key, task_ids in strata.items():
    allocations[key] = max(1, round(len(task_ids) * ratio))

current_total = sum(allocations.values())
print(f"\nInitial allocation: {current_total} (target: {TARGET})")

# Adjust to hit exactly TARGET
if current_total > TARGET:
    overshoot = current_total - TARGET
    sorted_keys = sorted(allocations.keys(), key=lambda k: (-allocations[k], k))
    idx = 0
    while overshoot > 0:
        key = sorted_keys[idx % len(sorted_keys)]
        if allocations[key] > 1:
            allocations[key] -= 1
            overshoot -= 1
        idx += 1
        if idx > len(sorted_keys) * TARGET:
            break
elif current_total < TARGET:
    undershoot = TARGET - current_total
    eligible = [k for k in strata if allocations[k] < len(strata[k])]
    # Prefer adding to larger strata first
    eligible.sort(key=lambda k: -len(strata[k]))
    idx = 0
    while undershoot > 0 and eligible:
        key = eligible[idx % len(eligible)]
        if allocations[key] < len(strata[key]):
            allocations[key] += 1
            undershoot -= 1
        else:
            eligible.remove(key)
            if not eligible:
                break
            continue
        idx += 1

final_total = sum(allocations.values())
assert final_total == TARGET, f"Allocation mismatch: {final_total} != {TARGET}"

# ── 5. Within each stratum, sample proportionally to project_name ─────────

sampled: list[str] = []
for key, n_alloc in allocations.items():
    pool = strata[key]

    # Sub-group by project_name within this stratum
    project_groups: dict[str, list[str]] = defaultdict(list)
    for tid in pool:
        proj = dataset[tid]["project_name"]
        project_groups[proj].append(tid)

    # Proportional allocation within stratum
    sub_ratio = n_alloc / len(pool)
    sub_alloc: dict[str, int] = {}
    for proj, tids in project_groups.items():
        sub_alloc[proj] = max(1, round(len(tids) * sub_ratio))

    sub_total = sum(sub_alloc.values())

    # Trim if over-allocated
    if sub_total > n_alloc:
        over = sub_total - n_alloc
        sorted_projs = sorted(sub_alloc.keys(), key=lambda p: (-sub_alloc[p], p))
        idx = 0
        while over > 0:
            p = sorted_projs[idx % len(sorted_projs)]
            if sub_alloc[p] > 1:
                sub_alloc[p] -= 1
                over -= 1
            idx += 1
            if idx > len(sorted_projs) * n_alloc:
                break
    elif sub_total < n_alloc:
        under = n_alloc - sub_total
        eligible_p = [
            p for p in project_groups if sub_alloc[p] < len(project_groups[p])
        ]
        random.shuffle(eligible_p)
        idx = 0
        while under > 0 and eligible_p:
            p = eligible_p[idx % len(eligible_p)]
            if sub_alloc[p] < len(project_groups[p]):
                sub_alloc[p] += 1
                under -= 1
            else:
                eligible_p.remove(p)
                if not eligible_p:
                    break
                continue
            idx += 1

    # Final trim: if still over due to many single-task projects
    sub_final = sum(sub_alloc.values())
    if sub_final > n_alloc:
        sorted_projs = sorted(sub_alloc.keys(), key=lambda p: (-sub_alloc[p], p))
        for p in sorted_projs:
            if sub_final <= n_alloc:
                break
            if sub_alloc[p] > 1:
                sub_alloc[p] -= 1
                sub_final -= 1

    for proj, n in sub_alloc.items():
        tids = project_groups[proj]
        sampled.extend(tids[:n])

# Safety: if we have slightly more/less due to rounding, adjust
if len(sampled) > TARGET:
    random.shuffle(sampled)
    sampled = sampled[:TARGET]
elif len(sampled) < TARGET:
    # Add from remaining unsampled tasks
    sampled_set = set(sampled)
    all_remaining = [tid for tid in dataset if tid not in sampled_set]
    random.shuffle(all_remaining)
    sampled.extend(all_remaining[: TARGET - len(sampled)])

sampled.sort()
assert len(sampled) == TARGET, f"Final sample size: {len(sampled)} != {TARGET}"

# ── 6. Write output ──────────────────────────────────────────────────────

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(OUTPUT_PATH, "w") as f:
    for task_id in sampled:
        f.write(task_id + "\n")

print(f"\nWrote {len(sampled)} task IDs to {OUTPUT_PATH}")

# ── 7. Summary ───────────────────────────────────────────────────────────

print(f"\n{'=' * 60}")
print(f"SAMPLE SUMMARY ({TARGET} tasks)")
print(f"{'=' * 60}")

# Language distribution
lang_counts = Counter(dataset[tid]["_lang_group"] for tid in sampled)
full_lang_counts = Counter(info["_lang_group"] for info in dataset.values())
print(f"\nLanguage distribution:")
print(f"  {'Language':<12} {'Sample':>8} {'SamplePct':>10} {'Full':>8} {'FullPct':>10}")
for lang in sorted(set(list(lang_counts.keys()) + list(full_lang_counts.keys()))):
    sc = lang_counts.get(lang, 0)
    fc = full_lang_counts.get(lang, 0)
    sp = sc / TARGET * 100
    fp = fc / total * 100
    print(f"  {lang:<12} {sc:>8} {sp:>9.1f}% {fc:>8} {fp:>9.1f}%")

# Crash type distribution
ct_counts = Counter(dataset[tid]["_crash_type"] for tid in sampled)
full_ct_counts = Counter(info["_crash_type"] for info in dataset.values())
print(f"\nCrash type distribution:")
print(
    f"  {'CrashType':<24} {'Sample':>8} {'SamplePct':>10} {'Full':>8} {'FullPct':>10}"
)
for ct in sorted(set(list(ct_counts.keys()) + list(full_ct_counts.keys()))):
    sc = ct_counts.get(ct, 0)
    fc = full_ct_counts.get(ct, 0)
    sp = sc / TARGET * 100
    fp = fc / total * 100
    print(f"  {ct:<24} {sc:>8} {sp:>9.1f}% {fc:>8} {fp:>9.1f}%")

# Top projects
proj_counts = Counter(dataset[tid]["project_name"] for tid in sampled)
print(f"\nTop 15 projects in sample:")
for proj, count in proj_counts.most_common(15):
    full_count = sum(1 for info in dataset.values() if info["project_name"] == proj)
    print(f"  {proj:<30} {count:>4} sampled / {full_count:>4} total")

# Strata coverage
strata_in_sample = set()
for tid in sampled:
    info = dataset[tid]
    strata_in_sample.add((info["_lang_group"], info["_crash_type"]))
print(f"\nStrata coverage: {len(strata_in_sample)}/{len(strata)} strata represented")
