# CyberGym Evolution Hot Restart

## Overview

When `run_evolution.py` crashes or is killed mid-round, hot restart resumes from where it left off: completed tasks are skipped, MEMRL caches are rebuilt from checkpoint data, and only remaining tasks are submitted.

## Current State (Round 1)

- **Output dir**: `results/evo_20260315_232848/`
- **Total tasks**: 1507
- **Completed**: ~1382 (in `round_001/tasks/*.json` with `status=completed`)
- **Remaining**: ~125 (error/timeout tasks to retry + never-started tasks)
- **Checkpoint**: `round_001/memrl_checkpoint/` (~361MB, saved during first run)

## Hot Restart Flow

### Step 1: MEMRL Init + Checkpoint Load (`run_batch.py:MemRLHelper._init`)

```
MemRLHelper(config_path=..., checkpoint_path=round_001/memrl_checkpoint)
  → load_checkpoint_snapshot()  # restores Qdrant vectors + SQLite metadata
  → _rebuild_caches_from_checkpoint()  # NEW: reads textual_memory.json
```

`_rebuild_caches_from_checkpoint` reads `textual_memory.json` to populate:
- `dict_memory`: `task_description → [memory_id, ...]` (used by `retrieve_query()`)
- `query_embeddings`: `task_description → embedding_vector` (used by `retrieve_query()`)

These are Python dicts that `load_checkpoint_snapshot` does NOT restore — only Qdrant/SQLite data is restored.

### Step 2: Round 1 Resume (`run_evolution.py:run_evolution`, round_num==1 branch)

```python
prev_completed, prev_completed_ids = _load_completed_round_tasks(round_dir, valid_task_ids=...)
remaining_task_ids = [tid for tid in task_ids if tid not in prev_completed_ids]
```

- Scans `round_001/tasks/*.json` for `status=completed`
- Builds list of remaining task IDs (error/timeout/missing)

### Step 3: MEMRL Replay (SKIPPED if caches populated)

Previously: called `build_memory()` for each completed task (~1382 LLM API calls, ~40-60min).

Now: checks `dict_memory` — if already populated from checkpoint, **skips replay entirely**.

```python
dm = getattr(memrl_helper.service, "dict_memory", None)
if dm:
    logger.info("MEMRL caches already populated — skipping replay")
```

### Step 4: Run Remaining Tasks

```python
new_results = asyncio.run(run_single_round(
    task_ids=remaining_task_ids,  # only ~125 tasks
    memrl_build_only=True,        # Round 1: build memories, no retrieval
    ...
))
```

### Step 5: Round-End Checkpoint Save

```python
memrl_helper.save_checkpoint(round_dir / "memrl_checkpoint")
```

**Critical question**: Does this checkpoint include BOTH the loaded memories (1382) AND the new memories (~125)?

## Key Files

| File | Role |
|------|------|
| `run_evolution.py` | Evolution orchestrator, round loop, resume logic |
| `run_batch.py` | `MemRLHelper` class (build/retrieve/checkpoint), `run_single_round()` |
| `run_batch.py:MemRLHelper._init` | Lines ~317-461: init, checkpoint load, cache rebuild |
| `run_batch.py:MemRLHelper._rebuild_caches_from_checkpoint` | Lines ~463-530: reads textual_memory.json |
| `run_batch.py:MemRLHelper.build` | Lines ~548-605: builds memory (lock around build_memory + cache update) |
| `run_batch.py:MemRLHelper.save_checkpoint` | Lines ~607-618: saves checkpoint snapshot |
| `run_evolution.py:_load_completed_round_tasks` | Lines ~100-150: loads completed task results from disk |
| `run_evolution.py:_replay_memrl_for_completed_tasks` | Lines ~152-260: replay (skipped if caches populated) |
| `run_evolution.py:run_evolution` | Lines ~560+: main evolution loop |

## Checkpoint Structure

```
round_001/memrl_checkpoint/
  snapshot/cybergym/
    cube/
      config.json
      textual_memory.json   # ~200MB, contains all memory entries
    qdrant/
      meta.json             # Qdrant vector collection snapshot
    snapshot_meta.json
```

### textual_memory.json Format

Array of memory points:
```json
[
  {
    "id": "<qdrant-point-id>",
    "vector": [0.023, -0.006, ...],   // 4096-dim embedding
    "payload": {
      "id": "<memory-uuid>",          // memory_id used in dict_memory
      "memory": "<task_description>",  // vulnerability description (key for dict_memory)
      "metadata": {
        "full_content": "Task: ...\n\nSCRIPT:...\nTRAJECTORY:...",
        "q_value": -0.3,
        "q_visits": 1,
        "success": false,
        "strategy_build": "proceduralization",
        ...
      }
    }
  },
  ...
]
```

## What Needs Investigation

1. **Does `save_checkpoint_snapshot` save ALL memories in MemOS?**
   - After hot restart: loaded 1382 from checkpoint + built ~125 new
   - Will the round-end checkpoint contain all 1507?
   - Check `memrl.service.save_checkpoint_snapshot` implementation in the memrl package

2. **Does `_rebuild_caches_from_checkpoint` correctly match what `build()` would populate?**
   - `build()` sets `dict_memory[task_desc] = [mem_id]` and `query_embeddings[task_desc] = vector`
   - `_rebuild_caches_from_checkpoint` reads `payload.memory` as task_desc and `payload.id` as mem_id
   - Are these the same fields? Verify by checking what `build_memory()` returns vs what's stored

3. **Are Q-values preserved correctly?**
   - Q-values (`q_value`, `q_visits`) are in textual_memory.json metadata
   - `load_checkpoint_snapshot` should restore them into the RL system
   - The old replay was WRONG — it re-applied `update_values()` which double-counted rewards
   - Verify: after checkpoint load, does `retrieve_query()` use the stored Q-values?

4. **Does `dict_memory` need ALL memories or just unique task_descriptions?**
   - Multiple memories can share the same task_description (same vulnerability, different rounds)
   - `dict_memory[task_desc]` is a list of mem_ids — all should be included
   - Check: does textual_memory.json have duplicate task_descriptions with different mem_ids?

5. **Thread safety of `build()` during batch execution**
   - `build()` now holds `_state_lock` around `build_memory()` + cache update
   - `asyncio.to_thread(memrl.build, ...)` with concurrency=32 means up to 32 threads
   - Only one thread can call `build_memory()` at a time (serialized by lock)
   - Embedding pre-computation runs outside lock (parallel)
   - Is this correct? Or does it bottleneck batch execution?
   - The original concurrency error ("cannot commit - no transaction is active") was from SQLite

## MEMRL Package Location

The `memrl` package is installed system-wide on the benchmark server:
```
/usr/local/lib/python3.13/dist-packages/memrl/
```

Key files to check:
- `memrl/service/memory_service.py` — `MemoryService.build_memory()`, `save_checkpoint_snapshot()`, `load_checkpoint_snapshot()`, `retrieve_query()`
- `memrl/providers/` — LLM and embedding providers
- `memos/` (MemOS) — underlying storage layer, Qdrant integration, SQLite user manager

## Server Info

- **Benchmark server**: `ssh wsl` (Host `10.180.161.222:2222`, User `shwii`) — SSH often times out
- **Code path**: `/inspire/hdd/project/multi-agent/niexiaohang-25130061/holos_synergy_experiments/ex_cybergym/`
- **MEMRL package**: `/usr/local/lib/python3.13/dist-packages/memrl/`
- **MemOS package**: `/usr/local/lib/python3.13/dist-packages/memos/`
