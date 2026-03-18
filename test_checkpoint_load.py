#!/usr/bin/env python3
"""Quick smoke test: load Round 1 checkpoint with fixed code."""

import os
import sys
import logging
import time

# Force our logger to WARNING so memos library can't swallow it
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
    force=True,
)
logger = logging.getLogger("SMOKE_TEST")
logger.setLevel(logging.WARNING)


def log(msg):
    """Print + log to ensure visibility."""
    print(f"[SMOKE] {msg}", flush=True)
    logger.warning(msg)


# Ensure local memrl is used
memrl_root = os.path.join(os.path.dirname(__file__), "MemRL")
if memrl_root not in sys.path:
    sys.path.insert(0, memrl_root)

CHECKPOINT_PATH = "results/evo_20260315_232848/round_001/memrl_checkpoint"
MEMRL_CONFIG = "configs/cybergym_memrl.yaml"


def main():
    from run_batch import MemRLHelper

    ckpt = os.path.abspath(CHECKPOINT_PATH)
    cfg = os.path.abspath(MEMRL_CONFIG)

    if not os.path.isdir(ckpt):
        log(f"✗ Checkpoint not found: {ckpt}")
        sys.exit(1)
    if not os.path.isfile(cfg):
        log(f"✗ Config not found: {cfg}")
        sys.exit(1)

    # Step 1: Load checkpoint
    log("Step 1/3: Loading checkpoint...")
    t0 = time.monotonic()
    try:
        helper = MemRLHelper(config_path=cfg, checkpoint_path=ckpt)
    except Exception as e:
        log(f"✗ FAILED to init MemRLHelper: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
    elapsed = time.monotonic() - t0
    log(f"✓ Checkpoint loaded in {elapsed:.1f}s")

    # Check state
    svc = helper.service
    dm = getattr(svc, "dict_memory", None)
    qe = getattr(svc, "query_embeddings", None)
    qdrant_dir = getattr(svc, "_qdrant_dir", None)

    log(f"  dict_memory: {len(dm) if dm else 0} unique tasks")
    log(f"  query_embeddings: {len(qe) if qe else 0} entries")
    log(f"  _qdrant_dir: {qdrant_dir}")

    if qdrant_dir and os.path.isdir(qdrant_dir):
        files = os.listdir(qdrant_dir)
        sqlite_files = [f for f in files if f.endswith((".sqlite", ".wal", ".db"))]
        log(f"  qdrant files: {len(files)} total, {len(sqlite_files)} sqlite")

    # Step 2: Test write (build_memory calls LLM, ~10-30s)
    log("Step 2/3: Testing memory write (LLM call, may take 10-30s)...")
    t1 = time.monotonic()
    try:
        mem_id = svc.build_memory(
            task_description="SMOKE_TEST: test task for checkpoint validation",
            trajectory="Step 1: Read code\nStep 2: Found bug\nStep 3: Fixed it",
            metadata={"source_benchmark": "smoke_test", "success": True},
        )
        elapsed = time.monotonic() - t1
        log(f"✓ build_memory succeeded in {elapsed:.1f}s: mem_id={mem_id}")
    except Exception as e:
        elapsed = time.monotonic() - t1
        log(f"✗ build_memory FAILED after {elapsed:.1f}s: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

    # Step 3: Test save
    import tempfile, shutil

    test_ckpt_dir = tempfile.mkdtemp(prefix="smoke_test_ckpt_")
    log(f"Step 3/3: Testing checkpoint save...")
    t2 = time.monotonic()
    try:
        helper.save_checkpoint(test_ckpt_dir)
        elapsed = time.monotonic() - t2
        snap_qdrant = os.path.join(test_ckpt_dir, "snapshot", "cybergym", "qdrant")
        has_qdrant = os.path.isdir(snap_qdrant)
        qdrant_files = os.listdir(snap_qdrant) if has_qdrant else []
        log(
            f"✓ save_checkpoint succeeded in {elapsed:.1f}s (qdrant_saved={has_qdrant}, files={len(qdrant_files)})"
        )
    except Exception as e:
        elapsed = time.monotonic() - t2
        log(f"✗ save_checkpoint FAILED after {elapsed:.1f}s: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
    finally:
        shutil.rmtree(test_ckpt_dir, ignore_errors=True)

    total = time.monotonic() - t0
    log(f"=== ALL 3 SMOKE TESTS PASSED in {total:.1f}s ===")


if __name__ == "__main__":
    main()
