# MemRL 引擎改动指南 (v4 论文对齐)

目标文件：`MemRL/memrl/service/memory_service.py`

共 4 处改动，2 个目的：
1. Q_init 统一为 0.0（去掉按 success/failure 分别初始化的逻辑）
2. 去掉 ε-greedy 随机探索，改为纯 greedy top-k 选择

---

## 改动 1/4：`build_memory()` — Q_init 统一化

搜索定位：`User-selected behavior: unknown success defaults to q_init_pos`

找到这段：

```python
                # User-selected behavior: unknown success defaults to q_init_pos.
                is_success = True if success is None else bool(success)
                base_meta |= {
                    "q_value": (
                        float(self.rl_config.q_init_pos)
                        if is_success
                        else float(self.rl_config.q_init_neg)
                    ),
                    "q_visits": 0,
```

替换为：

```python
                # Paper Table 8: unified Q_init=0.0 for all memories regardless
                # of success/failure. Q differentiates over time via RL updates.
                base_meta |= {
                    "q_value": float(self.rl_config.q_init_pos),
                    "q_visits": 0,
```

要点：删掉 `is_success` 那行，去掉三元表达式，直接用 `q_init_pos`。

---

## 改动 2/4：`prepare_memory()` — Q_init 统一化

搜索定位：同样搜 `User-selected behavior: unknown success defaults to q_init_pos`（文件中有两处，这是第二处，在 `prepare_memory` 方法内）

找到这段（和改动 1 结构完全一样）：

```python
            # User-selected behavior: unknown success defaults to q_init_pos.
            is_success = True if success is None else bool(success)
            base_meta |= {
                "q_value": (
                    float(self.rl_config.q_init_pos)
                    if is_success
                    else float(self.rl_config.q_init_neg)
                ),
                "q_visits": 0,
```

替换为：

```python
            # Paper Table 8: unified Q_init=0.0 for all memories regardless
            # of success/failure. Q differentiates over time via RL updates.
            base_meta |= {
                "q_value": float(self.rl_config.q_init_pos),
                "q_visits": 0,
```

---

## 改动 3/4：`_rebuild_caches_from_checkpoint()` — Q_init 统一化

搜索定位：`q_init_pos/q_init_neg according to the success flag`

找到这段：

```python
                # IMPORTANT: do NOT blindly override upstream metadata (runner may
                # provide q_value based on success/failure). Only fill missing
                # fields, and when q_value is missing/invalid, initialize from
                # q_init_pos/q_init_neg according to the success flag.
                meta.setdefault("success", bool(succ))

                rl_cfg = getattr(self, "rl_config", None)
                try:
                    q_init_pos = float(getattr(rl_cfg, "q_init_pos", 0.0))
                except Exception:
                    q_init_pos = 0.0
                try:
                    q_init_neg = float(getattr(rl_cfg, "q_init_neg", 0.0))
                except Exception:
                    q_init_neg = 0.0

                default_q = q_init_pos if bool(succ) else q_init_neg
                if "q_value" not in meta or meta.get("q_value") is None:
                    meta["q_value"] = default_q
                else:
                    # If provided but not castable, fall back to the default.
                    try:
                        meta["q_value"] = float(meta["q_value"])
                    except Exception:
                        meta["q_value"] = default_q
```

替换为：

```python
                # Paper Table 8: unified Q_init=0.0 regardless of success/failure.
                meta.setdefault("success", bool(succ))

                rl_cfg = getattr(self, "rl_config", None)
                try:
                    q_init = float(getattr(rl_cfg, "q_init_pos", 0.0))
                except Exception:
                    q_init = 0.0

                if "q_value" not in meta or meta.get("q_value") is None:
                    meta["q_value"] = q_init
                else:
                    try:
                        meta["q_value"] = float(meta["q_value"])
                    except Exception:
                        meta["q_value"] = q_init
```

---

## 改动 4/4：`retrieve_query()` — 去掉 ε-greedy

搜索定位：`epsilon-greedy sampling`

找到这段：

```python
            # -------- epsilon-greedy sampling --------
            topk = min(self.rl_config.topk, len(enriched_sorted))
            if not getattr(self, "dedup_by_task_id", False):
                if random.random() < self.rl_config.epsilon:
                    selected = random.sample(enriched_sorted, topk)
                else:
                    selected = enriched_sorted[:topk]
            else:
                # LLB-only: de-dup by task_id while keeping epsilon-greedy behavior.
                # - greedy: iterate score-desc
                # - epsilon: shuffle before taking unique tasks
                pool = list(enriched_sorted)
                if random.random() < self.rl_config.epsilon:
                    random.shuffle(pool)

                selected = []
                seen_tasks: set[str] = set()
                for cand in pool:
                    tid = cand.get("task_id")
                    # If task_id missing, treat as unique by memory_id to avoid collapsing unrelated entries.
                    key = (
                        str(tid)
                        if tid
                        else f"__missing_task_id__:{cand.get('memory_id')}"
                    )
                    if key in seen_tasks:
                        continue
                    seen_tasks.add(key)
                    selected.append(cand)
                    if len(selected) >= topk:
                        break
```

替换为：

```python
            # -------- Phase-B: greedy top-k₂ selection (paper Sec 4.2) --------
            # Paper uses pure score-ranked selection (equivalent to Boltzmann
            # optimal policy μ*(m|s) ∝ π_sim·exp(βQ), Appendix A.4.3).
            # No ε-greedy randomization — exploration is implicit in Q dynamics.
            topk = min(self.rl_config.topk, len(enriched_sorted))
            if not getattr(self, "dedup_by_task_id", False):
                selected = enriched_sorted[:topk]
            else:
                # De-dup by task_id: greedy iterate score-desc, skip duplicates.
                selected = []
                seen_tasks: set[str] = set()
                for cand in enriched_sorted:
                    tid = cand.get("task_id")
                    key = (
                        str(tid)
                        if tid
                        else f"__missing_task_id__:{cand.get('memory_id')}"
                    )
                    if key in seen_tasks:
                        continue
                    seen_tasks.add(key)
                    selected.append(cand)
                    if len(selected) >= topk:
                        break
```
