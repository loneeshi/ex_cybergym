#!/usr/bin/env python3
"""
CyberGym MEMRL evolution runner — 20-round iterative benchmark.

Each round runs the full benchmark with 16 concurrency, collects PoCs by task_id,
then feeds completed trajectories into MEMRL to build memory. The next round loads
the checkpoint from the previous round, enabling experience-driven evolution.

Architecture:
  Round 1: No memory → run full benchmark → build MEMRL memories → save checkpoint
  Round 2: Load Round 1 checkpoint → run → build → save
  ...
  Round N: Load Round N-1 checkpoint → run → build → save

After all rounds, generates a cross-round evolution report showing PoC rate
improvement, per-task coverage gain, and memory growth over time.

Usage:
    # Full 20-round evolution, 16 concurrent
    python3.13 run_evolution.py -s http://10.245.198.154:8000 -c 16 --rounds 20

    # Quick test: 3 rounds, 4 tasks, 4 concurrent
    python3.13 run_evolution.py -s http://10.245.198.154:8000 -c 4 -n 4 --rounds 3

    # Resume from round 8 (rounds 1-7 already completed)
    python3.13 run_evolution.py -s http://10.245.198.154:8000 --rounds 20 --resume-from 8

    # Custom output directory
    python3.13 run_evolution.py -s http://10.245.198.154:8000 --rounds 20 \
        -o results/evo_experiment_01
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import shutil
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Import from run_batch ───────────────────────────────────────────────────

from run_batch import (
    DEFAULT_CONCURRENCY,
    DEFAULT_LEVEL,
    DEFAULT_MODEL,
    DEFAULT_SERVER,
    DEFAULT_STEP_LIMIT,
    DEFAULT_TIMEOUT,
    SYSTEM_PROMPT,
    USER_PROMPT_TEMPLATE,
    MemRLHelper,
    _extract_session_trajectory,
    build_user_prompt,
    check_server_health,
    load_dataset_instances,
    load_task_ids_from_file,
    print_summary,
    run_batch,
)


# ── Intra-round resume helpers ────────────────────────────────────────────


def _load_completed_round_tasks(
    round_dir: Path,
    valid_task_ids: set[str] | None = None,
) -> tuple[list[dict[str, Any]], set[str]]:
    """Scan round_dir/tasks/ for completed task results from a previous run.

    Args:
        round_dir: Directory containing tasks/ subdirectory with result JSONs.
        valid_task_ids: If provided, only load results for tasks in this set.
            This prevents stale results from a previous larger run from leaking
            in when the output directory is reused with fewer tasks.

    Returns:
        (completed_results, completed_task_ids)
        Only tasks with status == "completed" are considered done.
        error / timeout tasks will be retried.
    """
    tasks_dir = round_dir / "tasks"
    if not tasks_dir.exists():
        return [], set()

    completed: list[dict[str, Any]] = []
    completed_ids: set[str] = set()
    status_counts: dict[str, int] = {}
    n_skipped_invalid = 0

    for f in tasks_dir.glob("*.json"):
        try:
            r = json.loads(f.read_text())
            status = r.get("status", "unknown")
            status_counts[status] = status_counts.get(status, 0) + 1
            if status == "completed":
                tid = r.get("task_id", "")
                base_tid = tid.split("/")[0] if "/" in tid else tid
                if valid_task_ids is not None and base_tid not in valid_task_ids:
                    n_skipped_invalid += 1
                    continue
                completed.append(r)
                completed_ids.add(base_tid)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Failed to read task file %s: %s", f, e)

    total_files = sum(status_counts.values())
    status_str = ", ".join(f"{s}={c}" for s, c in sorted(status_counts.items()))
    logger.info(
        "Loaded %d task files from %s: %s",
        total_files,
        tasks_dir.name,
        status_str,
    )
    if n_skipped_invalid:
        logger.info(
            "  Skipped %d tasks not in current valid_task_ids", n_skipped_invalid
        )

    # Count PoC stats for completed tasks
    n_poc = sum(1 for r in completed if r.get("poc_found"))
    n_validated = sum(1 for r in completed if r.get("validation_passed") is not None)
    n_val_passed = sum(1 for r in completed if r.get("validation_passed"))
    n_val_srv_err = sum(1 for r in completed if r.get("validation_server_error"))
    logger.info(
        "  Completed: %d (poc_found=%d, validated=%d, val_passed=%d, val_srv_err=%d)",
        len(completed),
        n_poc,
        n_validated,
        n_val_passed,
        n_val_srv_err,
    )

    return completed, completed_ids


def _replay_memrl_for_completed_tasks(
    completed_results: list[dict[str, Any]],
    sessions_dir: Path,
    instances: dict[str, dict[str, Any]],
    memrl: MemRLHelper,
    *,
    max_workers: int = 16,
) -> int:
    """Replay MEMRL build for previously completed tasks during intra-round resume.

    When resuming a partially completed round, tasks that finished before the
    crash already have results on disk, but their MEMRL memories were lost with
    the process.  This rebuilds those memories so the round-end checkpoint is
    complete.

    Uses ThreadPoolExecutor for concurrent replay.  Thread-safety is handled
    inside MemRLHelper.build() which holds ``_state_lock`` around the SQLite /
    Qdrant writes, while the expensive embedding pre-computation runs outside
    the lock so threads overlap on network I/O.

    NOTE: Q-value updates for *retrieved* memories from the original run cannot
    be replayed (retrieved IDs are not persisted).  Only the newly built memory's
    self-update is replayed.  This causes minor Q-value drift — acceptable given
    the alternative of losing all memories for skipped tasks.

    Returns:
        Number of memories successfully built.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    if not completed_results:
        return 0

    def _replay_one(r: dict[str, Any]) -> int:
        task_id = r.get("task_id", "")
        base_tid = task_id.split("/")[0] if "/" in task_id else task_id
        instance = instances.get(base_tid, {})

        safe_name = task_id.replace("/", "__").replace(":", "_")
        session_file = sessions_dir / f"{safe_name}.json"
        session_trajectory = _extract_session_trajectory(session_file)

        poc_found = r.get("poc_found", False)
        real_success = r.get("validation_passed", poc_found)

        trajectory_summary = (
            f"## Task: {task_id}\n"
            f"Project: {instance.get('project_name', '?')} "
            f"({instance.get('project_language', '?')})\n"
            f"Crash type: {instance.get('crash_type', '?')}\n"
            f"Status: {r.get('status')} | PoC found: {poc_found} | "
            f"Validation passed: {real_success}\n"
            f"Vul exit code: {r.get('vul_exit_code', 'N/A')} | "
            f"Fix exit code: {r.get('fix_exit_code', 'N/A')}\n"
            f"Steps: {r.get('metrics', {}).get('step_count', 0)}\n"
        )
        if session_trajectory:
            trajectory_summary += (
                f"\n## Agent Problem-Solving Trajectory\n{session_trajectory}\n"
            )

        mem_id = memrl.build(
            task_description=instance.get("vulnerability_description", ""),
            trajectory=trajectory_summary,
            metadata={
                "source": "cybergym",
                "task_id": task_id,
                "project": instance.get("project_name", ""),
                "project_language": instance.get("project_language", ""),
                "crash_type": instance.get("crash_type", ""),
                "success": real_success,
                "validated": r.get("validation_passed") is not None,
                "level": task_id.split("/")[-1] if "/" in task_id else "level1",
                "replayed": True,
                "poc_found": poc_found,
                "status": r.get("status", ""),
            },
        )
        if mem_id:
            # NOTE: Do NOT call update_values() here.  build_memory() already
            # initialises q_value to q_init_pos/q_init_neg based on the
            # ``success`` metadata flag.  If a checkpoint exists, the stored
            # Q-values already reflect the original run's updates.  Calling
            # update_values() again would double-count the reward, inflating
            # Q-values for successful tasks and deflating them for failures.
            return 1
        return 0

    n_total = len(completed_results)
    n_built = 0
    n_done = 0

    logger.info(
        "MEMRL replay: %d tasks with %d workers...",
        n_total,
        max_workers,
    )
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_replay_one, r): r for r in completed_results}
        for future in as_completed(futures):
            try:
                n_built += future.result()
            except Exception as e:
                r = futures[future]
                logger.warning(
                    "MEMRL replay failed for %s: %s",
                    r.get("task_id", "?"),
                    e,
                )
            n_done += 1
            if n_done % 100 == 0 or n_done == n_total:
                logger.info(
                    "MEMRL replay progress: %d/%d (built: %d)",
                    n_done,
                    n_total,
                    n_built,
                )

    return n_built


# ── Evolution orchestrator ──────────────────────────────────────────────────


def collect_poc_coverage(results: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Collect PoC results indexed by task_id.

    Returns:
        {task_id: {"poc_found": bool, "poc_size": int, "poc_base64": str, ...}}
    """
    coverage: dict[str, dict[str, Any]] = {}
    for r in results:
        tid = r.get("task_id", "")
        if not tid:
            continue
        coverage[tid] = {
            "poc_found": r.get("poc_found", False),
            "poc_size": r.get("poc_size", 0),
            "poc_base64": r.get("poc_base64", ""),
            "status": r.get("status", "unknown"),
            "elapsed": r.get("elapsed", 0),
            "steps": r.get("metrics", {}).get("step_count", 0),
            "validation_passed": r.get("validation_passed"),
            "vul_exit_code": r.get("vul_exit_code"),
            "fix_exit_code": r.get("fix_exit_code"),
        }
    return coverage


def merge_poc_bank(
    poc_bank: dict[str, dict[str, Any]],
    new_results: dict[str, dict[str, Any]],
    round_num: int,
) -> tuple[int, int]:
    """Merge newly found PoCs into the cumulative bank.

    Only adds PoCs that weren't previously found (first-hit wins).

    Returns:
        (new_pocs_added, total_pocs_in_bank)
    """
    added = 0
    for tid, info in new_results.items():
        if info.get("poc_found"):
            if tid not in poc_bank or not poc_bank[tid].get("poc_found"):
                poc_bank[tid] = {**info, "first_found_round": round_num}
                added += 1
    total = sum(1 for v in poc_bank.values() if v.get("poc_found"))
    return added, total


def build_round_summary(
    round_num: int,
    results: list[dict[str, Any]],
    poc_bank: dict[str, dict[str, Any]],
    elapsed: float,
    new_pocs: int,
    total_pocs: int,
    total_tasks: int,
) -> dict[str, Any]:
    """Build summary dict for a single round."""
    n = len(results)
    n_completed = sum(1 for r in results if r.get("status") == "completed")
    n_timeout = sum(1 for r in results if r.get("status") == "timeout")
    n_error = sum(1 for r in results if r.get("status") == "error")
    n_poc_this_round = sum(1 for r in results if r.get("poc_found"))
    n_with_memory = sum(1 for r in results if r.get("had_memory"))

    total_steps = sum(r.get("metrics", {}).get("step_count", 0) for r in results)
    total_tok_in = sum(
        r.get("metrics", {}).get("tokens", {}).get("input", 0) for r in results
    )
    total_tok_out = sum(
        r.get("metrics", {}).get("tokens", {}).get("output", 0) for r in results
    )

    n_validated = sum(1 for r in results if r.get("validation_passed") is not None)
    n_val_passed = sum(1 for r in results if r.get("validation_passed"))

    return {
        "round": round_num,
        "tasks_run": n,
        "completed": n_completed,
        "timeout": n_timeout,
        "error": n_error,
        "poc_found_this_round": n_poc_this_round,
        "poc_rate_this_round": round(n_poc_this_round / max(n, 1) * 100, 2),
        "validated": n_validated,
        "validation_passed": n_val_passed,
        "validation_pass_rate": round(n_val_passed / max(n_validated, 1) * 100, 2)
        if n_validated
        else 0,
        "new_pocs_added": new_pocs,
        "cumulative_pocs": total_pocs,
        "cumulative_poc_rate": round(total_pocs / max(total_tasks, 1) * 100, 2),
        "with_memory": n_with_memory,
        "total_steps": total_steps,
        "total_tokens_in": total_tok_in,
        "total_tokens_out": total_tok_out,
        "wall_time": round(elapsed, 1),
        "avg_elapsed": round(sum(r.get("elapsed", 0) for r in results) / max(n, 1), 1),
        "timestamp": datetime.now().isoformat(),
    }


def print_round_header(
    round_num: int, total_rounds: int, checkpoint: str | None
) -> None:
    banner = f"ROUND {round_num}/{total_rounds}"
    print(f"\n{'#' * 72}")
    print(f"##  {banner:^64}  ##")
    print(f"{'#' * 72}")
    if checkpoint:
        logger.info("Loading MEMRL checkpoint: %s", checkpoint)
    else:
        logger.info("No checkpoint — starting fresh (Round 1)")


def print_evolution_report(
    round_summaries: list[dict[str, Any]],
    poc_bank: dict[str, dict[str, Any]],
    total_tasks: int,
    total_elapsed: float,
) -> None:
    """Print the final cross-round evolution report."""
    print(f"\n{'=' * 72}")
    print(f"  MEMRL EVOLUTION REPORT — {len(round_summaries)} Rounds")
    print(f"{'=' * 72}")

    # Round-by-round table
    print(
        f"\n  {'Round':>5} {'PoC/Rnd':>8} {'Rate':>7} {'Valid':>6} {'VRate':>7} "
        f"{'New':>5} {'Cumul':>6} {'CumRate':>8} {'Mem':>5} {'Time':>8}"
    )
    print(
        f"  {'-' * 5} {'-' * 8} {'-' * 7} {'-' * 6} {'-' * 7} "
        f"{'-' * 5} {'-' * 6} {'-' * 8} {'-' * 5} {'-' * 8}"
    )
    for s in round_summaries:
        vp = s.get("validation_passed", 0)
        vr = s.get("validation_pass_rate", 0)
        print(
            f"  {s['round']:5d} {s['poc_found_this_round']:8d} "
            f"{s['poc_rate_this_round']:6.1f}% {vp:6d} {vr:6.1f}% "
            f"{s['new_pocs_added']:5d} {s['cumulative_pocs']:6d} "
            f"{s['cumulative_poc_rate']:7.1f}% "
            f"{s['with_memory']:5d} {s['wall_time']:7.0f}s"
        )

    # Overall stats
    total_rounds = len(round_summaries)
    first_rate = round_summaries[0]["poc_rate_this_round"] if round_summaries else 0
    last_rate = round_summaries[-1]["poc_rate_this_round"] if round_summaries else 0
    best_rate = (
        max(s["poc_rate_this_round"] for s in round_summaries) if round_summaries else 0
    )
    best_round = (
        next(
            s["round"] for s in round_summaries if s["poc_rate_this_round"] == best_rate
        )
        if round_summaries
        else 0
    )
    cum_pocs = round_summaries[-1]["cumulative_pocs"] if round_summaries else 0

    print(f"\n{'─' * 72}")
    print(f"  Summary")
    print(f"{'─' * 72}")
    print(f"  Total rounds:          {total_rounds}")
    print(f"  Total tasks/round:     {total_tasks}")
    print(f"  Round 1 PoC rate:      {first_rate:.1f}%")
    print(f"  Last round PoC rate:   {last_rate:.1f}%")
    print(f"  Best round PoC rate:   {best_rate:.1f}% (Round {best_round})")
    print(f"  Improvement:           {last_rate - first_rate:+.1f} pp")
    print(
        f"  Cumulative unique PoCs:{cum_pocs}/{total_tasks} "
        f"({cum_pocs / max(total_tasks, 1) * 100:.1f}%)"
    )
    print(
        f"  Total wall time:       {total_elapsed:.0f}s ({total_elapsed / 3600:.1f}h)"
    )

    total_tokens_in = sum(s["total_tokens_in"] for s in round_summaries)
    total_tokens_out = sum(s["total_tokens_out"] for s in round_summaries)
    print(f"  Total tokens:          {total_tokens_in:,} in / {total_tokens_out:,} out")

    # Per-task first-found distribution
    found_rounds: dict[int, int] = defaultdict(int)
    for v in poc_bank.values():
        if v.get("poc_found") and "first_found_round" in v:
            found_rounds[v["first_found_round"]] += 1

    if found_rounds:
        print(f"\n{'─' * 72}")
        print(f"  New PoCs discovered per round")
        print(f"{'─' * 72}")
        for r in sorted(found_rounds):
            bar = "█" * found_rounds[r]
            print(f"  Round {r:3d}: {found_rounds[r]:4d}  {bar}")

    print(f"\n{'=' * 72}\n")


async def run_single_round(
    round_num: int,
    server: str,
    task_ids: list[str],
    instances: dict[str, dict[str, Any]],
    model: str,
    level: str,
    concurrency: int,
    timeout: int,
    step_limit: int,
    output_dir: Path,
    memrl: Optional[MemRLHelper],
    cybergym_server: Optional[str] = None,
    memrl_build_only: bool = False,
) -> list[dict[str, Any]]:
    """Execute a single round — thin wrapper around run_batch."""
    return await run_batch(
        server=server,
        task_ids=task_ids,
        instances=instances,
        model=model,
        level=level,
        concurrency=concurrency,
        timeout=timeout,
        step_limit=step_limit,
        output_dir=output_dir,
        memrl=memrl,
        cybergym_server=cybergym_server,
        memrl_build_only=memrl_build_only,
    )


# ── Intra-round retry for error/timeout tasks ────────────────────────

MAX_RETRY_PASSES = 2
RETRY_TIMEOUT_BUMP = 1000  # extra seconds per retry pass for timeout tasks
RETRY_CONCURRENCY_CAP = 16  # lower concurrency for retries to protect server


def _retry_failed_tasks(
    round_num: int,
    all_results: list[dict[str, Any]],
    *,
    server: str,
    task_ids_full: list[str],
    instances: dict[str, dict[str, Any]],
    model: str,
    level: str,
    concurrency: int,
    base_timeout: int,
    step_limit: int,
    output_dir: Path,
    memrl: Optional[MemRLHelper],
    cybergym_server: Optional[str] = None,
    memrl_build_only: bool = False,
) -> list[dict[str, Any]]:
    """Retry error/timeout tasks with bumped timeout.

    - error tasks: retried with same timeout
    - timeout tasks: retried with timeout + RETRY_TIMEOUT_BUMP per pass
    - Concurrency is capped at RETRY_CONCURRENCY_CAP to protect the server
      (retried tasks are typically harder / longer-running)
    - Max RETRY_PASSES attempts; stops early if no failures remain

    Returns the updated full results list (failed entries replaced).
    """
    # Build lookup: base_task_id → result
    results_by_tid: dict[str, dict[str, Any]] = {}
    for r in all_results:
        tid = r.get("task_id", "")
        base = tid.split("/")[0] if "/" in tid else tid
        results_by_tid[base] = r

    for retry_pass in range(1, MAX_RETRY_PASSES + 1):
        failed_tids = [
            tid
            for tid, r in results_by_tid.items()
            if r.get("status") in ("error", "timeout")
        ]
        if not failed_tids:
            break

        retry_timeout = base_timeout + RETRY_TIMEOUT_BUMP * retry_pass
        retry_concurrency = max(1, min(RETRY_CONCURRENCY_CAP, len(failed_tids)))

        n_err = sum(
            1 for t in failed_tids if results_by_tid[t].get("status") == "error"
        )
        n_tout = sum(
            1 for t in failed_tids if results_by_tid[t].get("status") == "timeout"
        )
        logger.info(
            "Round %d retry pass %d/%d: %d tasks "
            "(error=%d, timeout=%d) — timeout=%ds, concurrency=%d",
            round_num,
            retry_pass,
            MAX_RETRY_PASSES,
            len(failed_tids),
            n_err,
            n_tout,
            retry_timeout,
            retry_concurrency,
        )

        retry_results = asyncio.run(
            run_single_round(
                round_num=round_num,
                server=server,
                task_ids=failed_tids,
                instances=instances,
                model=model,
                level=level,
                concurrency=retry_concurrency,
                timeout=retry_timeout,
                step_limit=step_limit,
                output_dir=output_dir,
                memrl=memrl,
                cybergym_server=cybergym_server,
                memrl_build_only=memrl_build_only,
            )
        )

        # Merge: replace old failed entries with retry results
        n_recovered = 0
        for r in retry_results:
            tid = r.get("task_id", "")
            base = tid.split("/")[0] if "/" in tid else tid
            old_status = results_by_tid.get(base, {}).get("status")
            results_by_tid[base] = r
            if r.get("status") == "completed" and old_status in ("error", "timeout"):
                n_recovered += 1

        n_still_failed = sum(
            1
            for r in results_by_tid.values()
            if r.get("status") in ("error", "timeout")
        )
        logger.info(
            "Round %d retry pass %d done: recovered=%d, still_failed=%d",
            round_num,
            retry_pass,
            n_recovered,
            n_still_failed,
        )

    return list(results_by_tid.values())


def run_evolution(
    server: str,
    model: str,
    level: str,
    concurrency: int,
    timeout: int,
    step_limit: int,
    num_rounds: int,
    memrl_config: str,
    task_ids: list[str],
    instances: dict[str, dict[str, Any]],
    base_output_dir: Path,
    resume_from: int = 1,
    cybergym_server: Optional[str] = None,
) -> None:
    """Main evolution loop: run N rounds with MEMRL checkpoint chaining.

    Round 1: Run without MEMRL (no memory yet). After all tasks complete,
             validate PoCs via cybergym_server, then build memories with
             real pass/fail reward. Save checkpoint.
    Round 2+: Load previous checkpoint. Run with MEMRL + inline validation
              (each task validated immediately, real reward used for memory).
              Save checkpoint.
    """

    total_tasks = len(task_ids)

    # ── Persistent state across rounds ──
    poc_bank: dict[str, dict[str, Any]] = {}  # task_id → best PoC info
    round_summaries: list[dict[str, Any]] = []

    # ── Load state from previously completed rounds (for resume) ──
    if resume_from > 1:
        logger.info("Resuming from round %d — loading previous state...", resume_from)
        for prev_round in range(1, resume_from):
            prev_dir = base_output_dir / f"round_{prev_round:03d}"
            prev_results_file = prev_dir / "all_results.json"
            prev_summary_file = prev_dir / "round_summary.json"

            if prev_results_file.exists():
                prev_results = json.loads(prev_results_file.read_text())
                prev_coverage = collect_poc_coverage(prev_results)
                new_pocs, total_pocs = merge_poc_bank(
                    poc_bank, prev_coverage, prev_round
                )
                logger.info(
                    "  Round %d: loaded %d results, %d new PoCs (cumulative: %d)",
                    prev_round,
                    len(prev_results),
                    new_pocs,
                    total_pocs,
                )

            if prev_summary_file.exists():
                round_summaries.append(json.loads(prev_summary_file.read_text()))

        # ── Warn if key parameters differ from the original run ──
        prev_config_file = base_output_dir / "evolution_config.json"
        if prev_config_file.exists():
            try:
                prev_cfg = json.loads(prev_config_file.read_text())
                mismatches: list[str] = []
                if prev_cfg.get("model") != model:
                    mismatches.append(f"model: {prev_cfg.get('model')} → {model}")
                if prev_cfg.get("level") != level:
                    mismatches.append(f"level: {prev_cfg.get('level')} → {level}")
                if prev_cfg.get("num_tasks") != total_tasks:
                    mismatches.append(
                        f"num_tasks: {prev_cfg.get('num_tasks')} → {total_tasks}"
                    )
                if prev_cfg.get("step_limit") != step_limit:
                    mismatches.append(
                        f"step_limit: {prev_cfg.get('step_limit')} → {step_limit}"
                    )
                if prev_cfg.get("timeout") != timeout:
                    mismatches.append(f"timeout: {prev_cfg.get('timeout')} → {timeout}")
                if mismatches:
                    logger.warning(
                        "Resume parameter mismatch with previous run:\n  %s",
                        "\n  ".join(mismatches),
                    )
            except (json.JSONDecodeError, OSError):
                pass

    # ── Save evolution config ──
    evo_config = {
        "server": server,
        "model": model,
        "level": level,
        "concurrency": concurrency,
        "timeout": timeout,
        "step_limit": step_limit,
        "num_rounds": num_rounds,
        "num_tasks": total_tasks,
        "memrl_config": memrl_config,
        "cybergym_server": cybergym_server,
        "resume_from": resume_from,
        "started_at": datetime.now().isoformat(),
    }
    base_output_dir.mkdir(parents=True, exist_ok=True)
    (base_output_dir / "evolution_config.json").write_text(
        json.dumps(evo_config, indent=2)
    )

    total_t0 = time.monotonic()

    # Initialize MEMRL once and reuse across all rounds.
    # Memories built inline during each round persist in memory;
    # checkpoints are saved for crash recovery only, not reloaded.
    memrl_helper: Optional[MemRLHelper] = None
    # Find the latest valid checkpoint — even for resume_from == 1, a
    # previous partial run may have saved a checkpoint that we can load
    # instead of replaying all completed tasks from scratch.
    checkpoint_path: str | None = None
    search_up_to = resume_from if resume_from > 1 else 2  # check round_001 too
    for prev_r in range(search_up_to - 1, 0, -1):
        prev_ckpt = base_output_dir / f"round_{prev_r:03d}" / "memrl_checkpoint"
        cube_check = prev_ckpt / "snapshot" / "cybergym" / "cube"
        if prev_ckpt.exists() and cube_check.exists():
            checkpoint_path = str(prev_ckpt)
            logger.info("Found valid checkpoint from round %d: %s", prev_r, prev_ckpt)
            break
    if not checkpoint_path:
        logger.info("No existing checkpoint found — MEMRL starts fresh")
    memrl_init_t0 = time.monotonic()
    try:
        memrl_helper = MemRLHelper(
            config_path=memrl_config,
            checkpoint_path=checkpoint_path,
        )
        logger.info(
            "MEMRL initialized in %.1fs (checkpoint=%s)",
            time.monotonic() - memrl_init_t0,
            "loaded" if checkpoint_path else "none",
        )
    except Exception as e:
        logger.error("MEMRL init failed: %s", e)
        logger.warning("Continuing WITHOUT memory")

    for round_num in range(resume_from, num_rounds + 1):
        round_dir = base_output_dir / f"round_{round_num:03d}"
        round_dir.mkdir(parents=True, exist_ok=True)

        print_round_header(round_num, num_rounds, None)

        # Save round config
        (round_dir / "config.json").write_text(
            json.dumps(
                {
                    "round": round_num,
                    "server": server,
                    "model": model,
                    "level": level,
                    "concurrency": concurrency,
                    "timeout": timeout,
                    "step_limit": step_limit,
                    "num_tasks": total_tasks,
                    "memrl_enabled": True,
                    "memrl_reused": round_num > 1,
                    "include_task_prompt": False,
                    "system_prompt": SYSTEM_PROMPT,
                    "timestamp": datetime.now().isoformat(),
                },
                indent=2,
            )
        )

        # Run the batch
        # Round 1: memrl build_only (build memories inline, no retrieval)
        # Round 2+: full memrl (retrieve + build)
        round_t0 = time.monotonic()
        if round_num == 1:
            # ── Round 1 intra-round resume: skip already-completed tasks ──
            prev_completed, prev_completed_ids = _load_completed_round_tasks(
                round_dir, valid_task_ids=set(task_ids)
            )
            remaining_task_ids = [
                tid for tid in task_ids if tid not in prev_completed_ids
            ]
            if prev_completed:
                logger.info(
                    "Round 1 resume: %d tasks already completed, %d remaining "
                    "(skipping %d error/timeout for retry)",
                    len(prev_completed),
                    len(remaining_task_ids),
                    len(task_ids) - len(prev_completed) - len(remaining_task_ids),
                )
                # Replay MEMRL build for completed tasks so round-end checkpoint
                # includes all memories (not just those from newly run tasks).
                # SKIP if caches were already rebuilt from checkpoint data
                # (dict_memory populated by _rebuild_caches_from_checkpoint).
                if memrl_helper:
                    dm = getattr(memrl_helper.service, "dict_memory", None)
                    if dm:
                        logger.info(
                            "MEMRL caches already populated from checkpoint "
                            "(%d unique tasks) — skipping replay",
                            len(dm),
                        )
                    else:
                        replay_t0 = time.monotonic()
                        logger.info(
                            "Replaying MEMRL build for %d previously completed tasks...",
                            len(prev_completed),
                        )
                        n_replayed = _replay_memrl_for_completed_tasks(
                            prev_completed,
                            round_dir / "sessions",
                            instances,
                            memrl_helper,
                        )
                        replay_elapsed = time.monotonic() - replay_t0
                        logger.info(
                            "MEMRL replay done: %d/%d memories rebuilt in %.1fs (%.1f tasks/s)",
                            n_replayed,
                            len(prev_completed),
                            replay_elapsed,
                            len(prev_completed) / max(replay_elapsed, 0.1),
                        )

            if remaining_task_ids:
                logger.info(
                    "Round 1: submitting %d tasks to benchmark server (concurrency=%d)...",
                    len(remaining_task_ids),
                    concurrency,
                )
                submit_t0 = time.monotonic()
                new_results = asyncio.run(
                    run_single_round(
                        round_num=round_num,
                        server=server,
                        task_ids=remaining_task_ids,
                        instances=instances,
                        model=model,
                        level=level,
                        concurrency=concurrency,
                        timeout=timeout,
                        step_limit=step_limit,
                        output_dir=round_dir,
                        memrl=memrl_helper,
                        cybergym_server=cybergym_server,
                        memrl_build_only=True,
                    )
                )
                submit_elapsed = time.monotonic() - submit_t0
                n_new_ok = sum(1 for r in new_results if r.get("status") == "completed")
                n_new_err = sum(1 for r in new_results if r.get("status") == "error")
                n_new_tout = sum(1 for r in new_results if r.get("status") == "timeout")
                n_new_poc = sum(1 for r in new_results if r.get("poc_found"))
                logger.info(
                    "Round 1 batch done: %d tasks in %.0fs — "
                    "completed=%d, error=%d, timeout=%d, poc_found=%d",
                    len(new_results),
                    submit_elapsed,
                    n_new_ok,
                    n_new_err,
                    n_new_tout,
                    n_new_poc,
                )
            else:
                new_results = []
                logger.info("Round 1: all tasks already completed — skipping batch")

            # Merge: previously completed + newly run
            results = prev_completed + new_results

            # ── Retry error/timeout tasks with bumped timeout ──
            n_failed = sum(
                1 for r in results if r.get("status") in ("error", "timeout")
            )
            if n_failed > 0:
                logger.info(
                    "Round 1: %d error/timeout tasks — starting retry passes", n_failed
                )
                results = _retry_failed_tasks(
                    round_num,
                    results,
                    server=server,
                    task_ids_full=task_ids,
                    instances=instances,
                    model=model,
                    level=level,
                    concurrency=concurrency,
                    base_timeout=timeout,
                    step_limit=step_limit,
                    output_dir=round_dir,
                    memrl=memrl_helper,
                    cybergym_server=cybergym_server,
                    memrl_build_only=True,
                )
        else:
            # Round 2+: memrl retrieves memories, inline validation for reward
            # ── Intra-round resume: skip already-completed tasks ──
            prev_completed, prev_completed_ids = _load_completed_round_tasks(
                round_dir, valid_task_ids=set(task_ids)
            )
            remaining_task_ids = [
                tid for tid in task_ids if tid not in prev_completed_ids
            ]
            if prev_completed:
                logger.info(
                    "Round %d resume: %d tasks already completed, %d remaining",
                    round_num,
                    len(prev_completed),
                    len(remaining_task_ids),
                )
                # Replay MEMRL build for completed tasks
                # SKIP if caches already populated from checkpoint
                if memrl_helper:
                    dm = getattr(memrl_helper.service, "dict_memory", None)
                    if dm and len(dm) >= len(prev_completed):
                        logger.info(
                            "MEMRL caches already populated (%d unique tasks) "
                            "— skipping replay",
                            len(dm),
                        )
                    else:
                        replay_t0 = time.monotonic()
                        logger.info(
                            "Replaying MEMRL build for %d previously completed tasks...",
                            len(prev_completed),
                        )
                        n_replayed = _replay_memrl_for_completed_tasks(
                            prev_completed,
                            round_dir / "sessions",
                            instances,
                            memrl_helper,
                        )
                        replay_elapsed = time.monotonic() - replay_t0
                        logger.info(
                            "MEMRL replay done: %d/%d memories rebuilt in %.1fs (%.1f tasks/s)",
                            n_replayed,
                            len(prev_completed),
                            replay_elapsed,
                            len(prev_completed) / max(replay_elapsed, 0.1),
                        )

            if remaining_task_ids:
                logger.info(
                    "Round %d: submitting %d tasks to benchmark server (concurrency=%d)...",
                    round_num,
                    len(remaining_task_ids),
                    concurrency,
                )
                submit_t0 = time.monotonic()
                new_results = asyncio.run(
                    run_single_round(
                        round_num=round_num,
                        server=server,
                        task_ids=remaining_task_ids,
                        instances=instances,
                        model=model,
                        level=level,
                        concurrency=concurrency,
                        timeout=timeout,
                        step_limit=step_limit,
                        output_dir=round_dir,
                        memrl=memrl_helper,
                        cybergym_server=cybergym_server,
                    )
                )
                submit_elapsed = time.monotonic() - submit_t0
                n_new_ok = sum(1 for r in new_results if r.get("status") == "completed")
                n_new_err = sum(1 for r in new_results if r.get("status") == "error")
                n_new_tout = sum(1 for r in new_results if r.get("status") == "timeout")
                n_new_poc = sum(1 for r in new_results if r.get("poc_found"))
                logger.info(
                    "Round %d batch done: %d tasks in %.0fs — "
                    "completed=%d, error=%d, timeout=%d, poc_found=%d",
                    round_num,
                    len(new_results),
                    submit_elapsed,
                    n_new_ok,
                    n_new_err,
                    n_new_tout,
                    n_new_poc,
                )
            else:
                new_results = []
                logger.info(
                    "Round %d: all tasks already completed — skipping batch",
                    round_num,
                )

            results = prev_completed + new_results

            # ── Retry error/timeout tasks with bumped timeout ──
            n_failed = sum(
                1 for r in results if r.get("status") in ("error", "timeout")
            )
            if n_failed > 0:
                logger.info(
                    "Round %d: %d error/timeout tasks — starting retry passes",
                    round_num,
                    n_failed,
                )
                results = _retry_failed_tasks(
                    round_num,
                    results,
                    server=server,
                    task_ids_full=task_ids,
                    instances=instances,
                    model=model,
                    level=level,
                    concurrency=concurrency,
                    base_timeout=timeout,
                    step_limit=step_limit,
                    output_dir=round_dir,
                    memrl=memrl_helper,
                    cybergym_server=cybergym_server,
                    memrl_build_only=False,
                )
        round_elapsed = time.monotonic() - round_t0

        # Print per-round summary (reuse existing function)
        print_summary(results, round_elapsed)

        # Save MEMRL checkpoint for next round
        if memrl_helper:
            ckpt_t0 = time.monotonic()
            ckpt_dir = str(round_dir / "memrl_checkpoint")
            memrl_helper.save_checkpoint(ckpt_dir)
            logger.info(
                "MEMRL checkpoint saved in %.1fs → %s",
                time.monotonic() - ckpt_t0,
                ckpt_dir,
            )

        # Collect PoCs and update bank
        round_coverage = collect_poc_coverage(results)
        new_pocs, total_pocs = merge_poc_bank(poc_bank, round_coverage, round_num)

        # Build round summary
        summary = build_round_summary(
            round_num=round_num,
            results=results,
            poc_bank=poc_bank,
            elapsed=round_elapsed,
            new_pocs=new_pocs,
            total_pocs=total_pocs,
            total_tasks=total_tasks,
        )
        round_summaries.append(summary)

        # Save round outputs
        (round_dir / "all_results.json").write_text(
            json.dumps(results, indent=2, ensure_ascii=False)
        )
        (round_dir / "round_summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False)
        )

        # Log progress
        n_srv_err = sum(1 for r in results if r.get("validation_server_error"))
        n_err = sum(1 for r in results if r.get("status") == "error")
        n_tout = sum(1 for r in results if r.get("status") == "timeout")
        n_completed = sum(1 for r in results if r.get("status") == "completed")
        logger.info(
            "Round %d/%d complete — %d tasks: completed=%d, error=%d, timeout=%d",
            round_num,
            num_rounds,
            len(results),
            n_completed,
            n_err,
            n_tout,
        )
        logger.info(
            "  PoC this round: %d/%d (%.1f%%) | New unique: %d | Cumulative: %d/%d (%.1f%%)",
            summary["poc_found_this_round"],
            len(results),
            summary["poc_rate_this_round"],
            new_pocs,
            total_pocs,
            total_tasks,
            summary["cumulative_poc_rate"],
        )
        if summary.get("validated", 0) > 0:
            logger.info(
                "  Validation: %d tested, %d passed (%.1f%%), %d server errors",
                summary["validated"],
                summary["validation_passed"],
                summary["validation_pass_rate"],
                n_srv_err,
            )
        logger.info(
            "  Wall time: %.0fs | Avg per task: %.1fs",
            round_elapsed,
            summary["avg_elapsed"],
        )

        # Save intermediate evolution state (for crash recovery)
        _save_evolution_state(base_output_dir, round_summaries, poc_bank, total_tasks)

    total_elapsed = time.monotonic() - total_t0

    # ── Final evolution report ──
    print_evolution_report(round_summaries, poc_bank, total_tasks, total_elapsed)

    # ── Save final outputs ──
    _save_evolution_state(base_output_dir, round_summaries, poc_bank, total_tasks)

    # Save the complete PoC bank (without base64 data, to keep it small)
    poc_bank_summary: dict[str, dict[str, Any]] = {}
    for tid, info in poc_bank.items():
        poc_bank_summary[tid] = {k: v for k, v in info.items() if k != "poc_base64"}
    (base_output_dir / "poc_bank.json").write_text(
        json.dumps(poc_bank_summary, indent=2, ensure_ascii=False)
    )

    # Save the full PoC bank (with base64) for validation
    (base_output_dir / "poc_bank_full.json").write_text(
        json.dumps(poc_bank, indent=2, ensure_ascii=False)
    )

    logger.info("Evolution complete. Results at %s", base_output_dir)


def _save_evolution_state(
    base_dir: Path,
    round_summaries: list[dict[str, Any]],
    poc_bank: dict[str, dict[str, Any]],
    total_tasks: int,
) -> None:
    """Save evolution progress for crash recovery and monitoring."""
    state = {
        "rounds_completed": len(round_summaries),
        "total_tasks": total_tasks,
        "round_summaries": round_summaries,
        "poc_coverage": {
            "total_unique_pocs": sum(
                1 for v in poc_bank.values() if v.get("poc_found")
            ),
            "task_ids_with_poc": [
                tid for tid, v in poc_bank.items() if v.get("poc_found")
            ],
        },
        "last_updated": datetime.now().isoformat(),
    }
    (base_dir / "evolution_state.json").write_text(
        json.dumps(state, indent=2, ensure_ascii=False)
    )


# ── CLI ─────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="CyberGym MEMRL evolution — multi-round benchmark with memory",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  %(prog)s -s http://10.245.198.154:8000 -c 16 --rounds 20\n"
            "  %(prog)s -s http://10.245.198.154:8000 -c 4 -n 4 --rounds 3\n"
            "  %(prog)s -s http://10.245.198.154:8000 --rounds 20 --resume-from 8\n"
        ),
    )

    # Server & model
    p.add_argument(
        "--server",
        "-s",
        default=DEFAULT_SERVER,
        help=f"Benchmark server URL (default: {DEFAULT_SERVER})",
    )
    p.add_argument(
        "--model",
        "-m",
        default=DEFAULT_MODEL,
        help=f"Model to use (default: {DEFAULT_MODEL})",
    )
    p.add_argument(
        "--level", default=DEFAULT_LEVEL, help=f"Task level (default: {DEFAULT_LEVEL})"
    )

    # Execution
    p.add_argument(
        "--concurrency",
        "-c",
        type=int,
        default=DEFAULT_CONCURRENCY,
        help=f"Concurrent tasks (default: {DEFAULT_CONCURRENCY})",
    )
    p.add_argument(
        "--timeout",
        "-t",
        type=int,
        default=DEFAULT_TIMEOUT,
        help=f"Per-task timeout in seconds (default: {DEFAULT_TIMEOUT})",
    )
    p.add_argument(
        "--step-limit",
        type=int,
        default=DEFAULT_STEP_LIMIT,
        help=f"Max agent steps per task (default: {DEFAULT_STEP_LIMIT})",
    )

    # Tasks
    p.add_argument(
        "--num-tasks",
        "-n",
        type=int,
        default=None,
        help="Number of tasks to run (default: all)",
    )
    p.add_argument(
        "--task-file", type=str, default=None, help="File with task IDs (one per line)"
    )

    # Evolution
    p.add_argument(
        "--rounds",
        type=int,
        default=20,
        help="Number of evolution rounds (default: 20)",
    )
    p.add_argument(
        "--resume-from",
        type=int,
        default=1,
        help="Resume from this round number (previous rounds must exist)",
    )

    # MEMRL
    p.add_argument(
        "--memrl-config",
        type=str,
        default="configs/cybergym_memrl.yaml",
        help="MEMRL config YAML path",
    )

    # Validation
    p.add_argument(
        "--cybergym-server",
        type=str,
        default=None,
        help="CyberGym validation server URL (required for real reward signals)",
    )

    # Output
    p.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default=None,
        help="Base output directory (default: results/evo_TIMESTAMP)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Show config and task list without running",
    )

    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Output directory
    if args.output_dir:
        base_output_dir = Path(args.output_dir)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_output_dir = Path(__file__).parent / "results" / f"evo_{ts}"

    # Load dataset
    instances = load_dataset_instances()

    # Select tasks
    if args.task_file:
        task_ids = load_task_ids_from_file(args.task_file)
    else:
        task_ids = list(instances.keys())

    if args.num_tasks:
        task_ids = task_ids[: args.num_tasks]

    total_tasks = len(task_ids)

    # Print config
    logger.info("=" * 72)
    logger.info("CyberGym MEMRL Evolution Runner")
    logger.info("  Server:       %s", args.server)
    logger.info("  Model:        %s", args.model)
    logger.info("  Level:        %s", args.level)
    logger.info("  Tasks:        %d", total_tasks)
    logger.info("  Concurrency:  %d", args.concurrency)
    logger.info("  Timeout:      %ds", args.timeout)
    logger.info("  Step limit:   %d", args.step_limit)
    logger.info("  Rounds:       %d", args.rounds)
    logger.info("  Resume from:  %d", args.resume_from)
    logger.info("  MEMRL config: %s", args.memrl_config)
    logger.info("  CyberGym:     %s", args.cybergym_server or "(none — no validation)")
    logger.info("  Output:       %s", base_output_dir)
    logger.info("=" * 72)

    if args.dry_run:
        for i, tid in enumerate(task_ids):
            inst = instances.get(tid, {})
            proj = inst.get("project_name", "?")
            lang = inst.get("project_language", "?")
            print(f"  [{i + 1:3d}] {tid}/{args.level}  ({proj}, {lang})")
        print(
            f"\n  Total: {total_tasks} tasks × {args.rounds} rounds = "
            f"{total_tasks * args.rounds} task runs (dry-run)"
        )
        return

    if not task_ids:
        logger.warning("No tasks to run!")
        return

    # Verify MEMRL config exists
    if not Path(args.memrl_config).exists():
        logger.error("MEMRL config not found: %s", args.memrl_config)
        sys.exit(1)

    # Check server health
    check_server_health(args.server)

    # Run evolution
    run_evolution(
        server=args.server,
        model=args.model,
        level=args.level,
        concurrency=args.concurrency,
        timeout=args.timeout,
        step_limit=args.step_limit,
        num_rounds=args.rounds,
        memrl_config=args.memrl_config,
        task_ids=task_ids,
        instances=instances,
        base_output_dir=base_output_dir,
        resume_from=args.resume_from,
        cybergym_server=args.cybergym_server,
    )


if __name__ == "__main__":
    main()
