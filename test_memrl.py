#!/usr/bin/env python3
"""
Test MEMRL pipeline using existing task results.

Reads completed task results, builds memories from their trajectories,
updates Q-values based on validation outcomes, and tests retrieval
to verify the full MEMRL loop works.

Usage:
    python3.13 test_memrl.py results/retest_failed_21
    python3.13 test_memrl.py results/retest_failed_21 --memrl-config configs/cybergym_memrl.yaml
    python3.13 test_memrl.py results/retest_failed_21 --save-checkpoint results/memrl_ckpt
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

from run_batch import (
    MemRLHelper,
    _extract_session_trajectory,
    load_dataset_instances,
)


def load_results(results_dir: Path) -> list[dict[str, Any]]:
    """Load all task result JSONs from results_dir/tasks/."""
    tasks_dir = results_dir / "tasks"
    if not tasks_dir.exists():
        logger.error("No tasks/ directory in %s", results_dir)
        sys.exit(1)

    results = []
    for f in sorted(tasks_dir.glob("*.json")):
        try:
            data = json.loads(f.read_text())
            if data.get("status") == "completed":
                results.append(data)
        except Exception:
            continue
    return results


def main():
    p = argparse.ArgumentParser(description="Test MEMRL with existing results")
    p.add_argument("results_dir", type=str, help="Path to results directory")
    p.add_argument(
        "--memrl-config",
        type=str,
        default="configs/cybergym_memrl.yaml",
        help="MEMRL config YAML",
    )
    p.add_argument(
        "--memrl-checkpoint",
        type=str,
        default=None,
        help="Load existing MEMRL checkpoint before building",
    )
    p.add_argument(
        "--save-checkpoint",
        type=str,
        default=None,
        help="Save MEMRL checkpoint after building (default: <results_dir>/memrl_checkpoint)",
    )
    p.add_argument(
        "--test-queries",
        type=int,
        default=5,
        help="Number of retrieval queries to test after building",
    )
    args = p.parse_args()

    results_dir = Path(args.results_dir)
    sessions_dir = results_dir / "sessions"

    # ── Load results ──
    results = load_results(results_dir)
    logger.info("Loaded %d completed results from %s", len(results), results_dir)

    if not results:
        logger.error("No completed results found!")
        sys.exit(1)

    # ── Load dataset for instance metadata ──
    instances = load_dataset_instances()

    # ── Initialize MEMRL ──
    logger.info("Initializing MEMRL from %s ...", args.memrl_config)
    memrl = MemRLHelper(
        config_path=args.memrl_config,
        checkpoint_path=args.memrl_checkpoint,
    )
    logger.info("MEMRL initialized")

    # ── Phase 1: Build memories from results (concurrent) ──
    print("\n" + "=" * 60)
    print("Phase 1: Building memories from task results")
    print("=" * 60)

    from concurrent.futures import ThreadPoolExecutor, as_completed

    built_mem_ids: list[str] = []
    built_successes: list[float] = []
    build_details: list[dict[str, Any]] = []
    n_built = 0
    n_build_failed = 0

    def _build_one(i: int, r: dict[str, Any]) -> dict[str, Any]:
        task_id = r.get("task_id", "")
        base_tid = task_id.split("/")[0] if "/" in task_id else task_id
        inst = instances.get(base_tid, {})

        poc_found = r.get("poc_found", False)
        real_success = r.get("validation_passed", poc_found)

        safe_name = task_id.replace("/", "__").replace(":", "_")
        session_file = sessions_dir / f"{safe_name}.json"
        session_trajectory = _extract_session_trajectory(session_file)

        trajectory_summary = (
            f"## Task: {task_id}\n"
            f"Project: {inst.get('project_name', '?')} "
            f"({inst.get('project_language', '?')})\n"
            f"Crash type: {inst.get('crash_type', '?')}\n"
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
        else:
            trajectory_summary += (
                "\n(No session data — task errored before agent started)\n"
            )

        t0 = time.monotonic()
        mem_id = memrl.build(
            task_description=inst.get("vulnerability_description", ""),
            trajectory=trajectory_summary,
            metadata={
                "source": "cybergym",
                "task_id": task_id,
                "project": inst.get("project_name", ""),
                "success": real_success,
                "validated": r.get("validation_passed") is not None,
                "level": task_id.split("/")[-1] if "/" in task_id else "level1",
            },
        )
        build_time = time.monotonic() - t0

        icon = "✓" if mem_id else "✗"
        success_icon = "pass" if real_success else "fail"
        has_session = "with trajectory" if session_trajectory else "no trajectory"
        logger.info(
            "[%d/%d] %s build  %s (%s, %s) — mem_id=%s, %.1fs",
            i + 1,
            len(results),
            icon,
            task_id,
            success_icon,
            has_session,
            mem_id or "FAILED",
            build_time,
        )

        return {
            "task_id": task_id,
            "mem_id": mem_id,
            "success": real_success,
            "has_trajectory": bool(session_trajectory),
            "build_time": round(build_time, 2),
        }

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(_build_one, i, r): r for i, r in enumerate(results)}
        for future in as_completed(futures):
            try:
                detail = future.result()
                build_details.append(detail)
                if detail["mem_id"]:
                    built_mem_ids.append(detail["mem_id"])
                    built_successes.append(1.0 if detail["success"] else 0.0)
                    n_built += 1
                else:
                    n_build_failed += 1
            except Exception as e:
                n_build_failed += 1
                task_id = futures[future].get("task_id", "?")
                logger.warning("Memory build failed for %s: %s", task_id, e)

    n_success = sum(1 for s in built_successes if s > 0)
    n_failure = n_built - n_success
    print(f"\n  Built: {n_built}/{len(results)} memories ({n_build_failed} failed)")
    print(f"  Success: {n_success}, Failure: {n_failure}")

    # ── Phase 2: Update Q-values ──
    if built_mem_ids:
        print("\n" + "=" * 60)
        print("Phase 2: Updating Q-values")
        print("=" * 60)

        memrl.update_values(
            built_successes,
            [[mid] for mid in built_mem_ids],
        )
        logger.info(
            "Q-value update done: %d memories "
            "(%d success → Q≈0.3, %d failure → Q≈-0.3)",
            n_built,
            n_success,
            n_failure,
        )

    # ── Phase 3: Test retrieval ──
    print("\n" + "=" * 60)
    print("Phase 3: Testing memory retrieval")
    print("=" * 60)

    test_tasks = results[: args.test_queries]
    for i, r in enumerate(test_tasks):
        task_id = r.get("task_id", "")
        base_tid = task_id.split("/")[0] if "/" in task_id else task_id
        inst = instances.get(base_tid, {})

        query = (
            f"{inst.get('vulnerability_description', '')} {inst.get('crash_type', '')}"
        )

        t0 = time.monotonic()
        context, retrieved_ids = memrl.retrieve(query)
        retrieve_time = time.monotonic() - t0

        print(f"\n  [{i + 1}] Query for: {task_id}")
        print(f"      Project: {inst.get('project_name', '?')}")
        print(f"      Retrieved {len(retrieved_ids)} memories in {retrieve_time:.2f}s")
        if retrieved_ids:
            for mid in retrieved_ids:
                print(f"        - {mid}")
        if context:
            preview = context[:200].replace("\n", " ")
            print(f"      Context preview: {preview}...")
        else:
            print(f"      Context: (empty)")

    # ── Phase 4: Save checkpoint ──
    ckpt_path = args.save_checkpoint or str(results_dir / "memrl_checkpoint")
    print("\n" + "=" * 60)
    print(f"Phase 4: Saving checkpoint to {ckpt_path}")
    print("=" * 60)
    memrl.save_checkpoint(ckpt_path)

    # Save build report
    report = {
        "results_dir": str(results_dir),
        "total_results": len(results),
        "memories_built": n_built,
        "success_memories": n_success,
        "failure_memories": n_failure,
        "checkpoint_path": ckpt_path,
        "build_details": build_details,
    }
    report_path = results_dir / "memrl_build_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    logger.info("Build report saved to %s", report_path)

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)
    print(f"  Memories built: {n_built}")
    print(f"  Checkpoint: {ckpt_path}")
    print(f"  Report: {report_path}")
    print(f"\n  To use this checkpoint in evolution:")
    print(f"    python3.13 run_evolution.py ... --memrl-checkpoint {ckpt_path}")
    print()


if __name__ == "__main__":
    main()
