#!/usr/bin/env python3
"""
Validate existing PoC results against the CyberGym dual-container server.

Reads task result JSONs from a results directory and sends each PoC
to the validation server without re-running the agent.

Usage:
    python3 validate_results.py results/retest_failed_21
    python3 validate_results.py results/retest_failed_21 --server http://10.1.2.168:3000
    python3 validate_results.py results/retest_failed_21 -c 4
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

from run_batch import validate_poc_inline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

DEFAULT_SERVER = "http://10.1.2.168:3000"


async def validate_all(
    results_dir: Path,
    server: str,
    concurrency: int,
) -> list[dict[str, Any]]:
    tasks_dir = results_dir / "tasks"
    if not tasks_dir.exists():
        logger.error("No tasks/ directory found in %s", results_dir)
        sys.exit(1)

    task_files = sorted(tasks_dir.glob("*.json"))
    if not task_files:
        logger.error("No task result files found in %s", tasks_dir)
        sys.exit(1)

    to_validate: list[dict[str, Any]] = []
    skipped = 0
    for f in task_files:
        try:
            data = json.loads(f.read_text())
        except Exception:
            continue
        poc_b64 = data.get("poc_base64", "")
        if not poc_b64 or not data.get("poc_found"):
            skipped += 1
            continue
        to_validate.append({"file": f, "data": data})

    total = len(to_validate)
    logger.info(
        "Found %d tasks with PoC to validate (%d skipped, no PoC)",
        total,
        skipped,
    )

    sem = asyncio.Semaphore(concurrency)
    results: list[dict[str, Any]] = []
    passed = 0
    failed = 0

    async def validate_one(item: dict, idx: int) -> dict[str, Any]:
        nonlocal passed, failed
        async with sem:
            data = item["data"]
            task_id = data.get("task_id", "?")
            poc_b64 = data["poc_base64"]

            vr = await asyncio.to_thread(
                validate_poc_inline,
                server,
                task_id,
                poc_b64,
            )

            icon = "✓" if vr.get("passed") else "✗"
            if vr.get("passed"):
                passed += 1
            else:
                failed += 1
            logger.info(
                "[%d/%d] %s %s — vul_exit=%s, fix_exit=%s, passed=%s",
                idx + 1,
                total,
                icon,
                task_id,
                vr.get("vul_exit_code"),
                vr.get("fix_exit_code"),
                vr.get("passed"),
            )

            data["validation"] = vr
            data["validation_passed"] = vr.get("passed", False)
            data["vul_exit_code"] = vr.get("vul_exit_code")
            data["fix_exit_code"] = vr.get("fix_exit_code")

            item["file"].write_text(json.dumps(data, indent=2, ensure_ascii=False))

            results.append(data)
            return data

    tasks = [validate_one(item, i) for i, item in enumerate(to_validate)]
    await asyncio.gather(*tasks)

    print("\n" + "=" * 60)
    print("Validation Results")
    print("=" * 60)
    print(f"  Total with PoC: {total}")
    print(f"  Passed:         {passed}/{total}")
    print(f"  Failed:         {failed}/{total}")
    if total:
        print(f"  Pass rate:      {passed / total * 100:.1f}%")
    print("=" * 60)

    for r in results:
        icon = "✓" if r.get("validation_passed") else "✗"
        print(
            f"  {icon} {r.get('task_id')}: "
            f"vul_exit={r.get('vul_exit_code')}, "
            f"fix_exit={r.get('fix_exit_code')}"
        )
    print()

    return results


def main():
    p = argparse.ArgumentParser(description="Validate existing PoC results")
    p.add_argument("results_dir", type=str, help="Path to results directory")
    p.add_argument(
        "--server", default=DEFAULT_SERVER, help="CyberGym validation server"
    )
    p.add_argument("-c", "--concurrency", type=int, default=4)
    args = p.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        logger.error("Results directory not found: %s", results_dir)
        sys.exit(1)

    t0 = time.monotonic()
    results = asyncio.run(validate_all(results_dir, args.server, args.concurrency))
    elapsed = time.monotonic() - t0
    logger.info("Validation completed in %.1fs", elapsed)

    summary_path = results_dir / "validation_summary.json"
    n_passed = sum(1 for r in results if r.get("validation_passed"))
    summary = {
        "total": len(results),
        "passed": n_passed,
        "failed": len(results) - n_passed,
        "pass_rate": round(n_passed / max(len(results), 1) * 100, 2),
        "elapsed": round(elapsed, 1),
        "server": args.server,
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    logger.info("Summary saved to %s", summary_path)


if __name__ == "__main__":
    main()
