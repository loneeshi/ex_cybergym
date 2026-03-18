# MemRL 体系架构文档

## 1. 总体架构

```
┌─────────────────────────────────────────────────────────────────┐
│                    run_evolution.py                              │
│  Round 1 → Round 2 → ... → Round N                             │
│  每轮: run_batch → 收集结果 → save checkpoint → 下一轮复用      │
└──────────────┬──────────────────────────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────────────────────────┐
│                    run_batch.py                                  │
│  MemRLHelper (封装层)                                            │
│  ┌──────────┐ ┌──────────┐ ┌──────────────┐ ┌───────────────┐  │
│  │ retrieve │ │  build   │ │ update_values │ │save_checkpoint│  │
│  └──────────┘ └──────────┘ └──────────────┘ └───────────────┘  │
└──────────────┬──────────────────────────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────────────────────────┐
│           MemRL/memrl/service/memory_service.py                 │
│           MemoryService (核心引擎)                               │
│                                                                  │
│  存储后端: MOS (MemOS) → Qdrant (向量库) + SQLite (用户管理)     │
│  LLM 后端: qwen3.5-397b-a17b (proceduralization)                │
│  Embedding: qwen3-embedding-8b (4096 维向量)                     │
└─────────────────────────────────────────────────────────────────┘
```

三层结构:

- **run_evolution.py** — 演化编排层。管理多轮循环、checkpoint 链式传递、crash recovery、结果汇总。
- **run_batch.py / MemRLHelper** — 接口封装层。将 MemoryService 的复杂 API 封装为 `retrieve` / `build` / `update_values` / `save_checkpoint` 四个简洁方法，处理线程安全、去重、Q 值归因逻辑。
- **MemRL/memrl/** — 核心引擎层。包含记忆构建、检索、更新、RL Q 值管理、向量存储等底层实现。


## 2. 论文核心公式与实现映射

### 2.1 记忆三元组 (Intent-Experience-Utility)

论文定义记忆库 `M = {(z_i, e_i, Q_i)}`：

| 论文符号 | 含义 | 代码映射 |
|----------|------|---------|
| `z_i` (Intent) | 任务意图，用于 embedding 检索 | `TextualMemoryItem.memory` = vulnerability_description |
| `e_i` (Experience) | 经验内容，注入 agent context | `metadata.full_content` (成功: SCRIPT+TRAJECTORY, 失败: Reflection) |
| `Q_i` (Utility) | RL 学习的效用值 | `metadata.q_value` |

### 2.2 两阶段检索 (Retrieval Policy μ)

**Phase A — 相似度召回 (Similarity-Based Recall):**

```
C(s) = TopK_k1({i | sim(Emb(s), Emb(z_i)) > δ})
```

| 论文参数 | 论文推荐值 | 代码实现 |
|----------|-----------|---------|
| `k₁` (候选池大小) | 5~10 | 全量扫描 dict_memory (功能等价于 k₁=∞) |
| `δ` (相似度门槛) | top-20% quantile (0.25~0.62) | `confidence_threshold = 0.3` |

**Phase B — 值感知选择 (Value-Aware Selection):**

```
score(s, z_i, e_i) = (1-λ) · ẑ(sim) + λ · ẑ(Q_i)
```

| 论文参数 | 论文推荐值 | 代码实现 |
|----------|-----------|---------|
| `λ` (权重平衡) | 0.5 (ablation 最优) | `weight_sim=0.5, weight_q=0.5` |
| `k₂` (最终选择) | 3~5 | `topk = 3` |
| `ẑ(sim)` 标准化 | z-score | `(sim - 0.186) / 0.094` (预计算全局统计) |
| `ẑ(Q)` 标准化 | z-score | 当前候选池 mean/std, clamp [-3, 3] |
| 选择策略 | 纯 score 排序 (Boltzmann 等价) | 纯 greedy top-k₂ (与论文一致) |

### 2.3 Q 值更新 (Utility Learning, Eq. 4)

```
Q_new ← Q_old + α(r - Q_old)
```

等价于 TD 公式 `Q ← Q + α[r + γ·max_Q' - Q]` 在 `γ=0` 时的简化形式。

| 论文参数 | 论文推荐值 | 代码实现 |
|----------|-----------|---------|
| `α` (学习率) | 0.3 (Table 8, 全 benchmark) | `alpha = 0.3` |
| `γ` (折扣因子) | 0 (MC 风格, 只看即时奖励) | `gamma = 0.0` |
| `Q_init` | 0.0 (统一初始化) | `q_init_pos = 0.0, q_init_neg = 0.0` |
| `r` (奖励) | success=+1, failure=-1 | `success_reward=1.0, failure_reward=-1.0` |
| 归因范围 | 对称: ALL retrieved 统一更新 | 对称归因 (与论文一致) |


## 3. 三大策略组合

当前配置 (`configs/cybergym_memrl.yaml`):

| 策略类型 | 当前选择 | 含义 | 替代选项 |
|----------|---------|------|---------|
| **Build** | `proceduralization` | 成功: LLM 生成 SCRIPT + 保留 TRAJECTORY; 失败: LLM 生成 Reflection (跳过 proceduralization) | `trajectory` (原样存储), `script` (仅摘要) |
| **Retrieve** | `query` | 用 task_description 原文向量做 cosine 检索 | `random` (随机), `avefact` (关键词平均向量) |
| **Update** | `adjustment` | 成功→新增记忆; 失败→LLM 反思 + 新增反思记忆 | `vanilla` (全部新增), `validation` (仅成功新增) |

这三类策略在 `MemRL/memrl/service/strategies.py` 中定义，共 3×3×3 = 27 种组合。


## 4. 单条记忆的结构

### 成功记忆 (type: "procedure")

```python
TextualMemoryItem {
    id:       "mem_xxxx"
    memory:   "A heap-buffer-overflow occurs..."  # embedding key = vulnerability_description
    metadata: {
        type:           "procedure"
        full_content:   "Task: ...\n\nSCRIPT:\n...\n\nTRAJECTORY:\n..."
        task_id:        "arvo:52145"
        success:        true
        q_value:        0.0                       # 初始 Q_init=0.0，后续通过 RL 更新
        q_visits:       0
        confidence:     100.0
        ...
    }
}
```

经 `prepare_memory()` → LLM proceduralization 生成 SCRIPT + 保留 TRAJECTORY。

### 失败记忆 (type: "adjustment")

```python
TextualMemoryItem {
    id:       "mem_yyyy"
    memory:   "A heap-buffer-overflow occurs..."  # 同样的 embedding key
    metadata: {
        type:           "adjustment"
        full_content:   "Task: ...\n\n## Task Header\n...\n\n## Failure Reflection\nROOT CAUSE: ...\nKEY MISTAKES:\n- ...\nAVOID: ..."
        task_id:        "arvo:52145"
        success:        false
        q_value:        0.0
        q_visits:       0
        confidence:     80.0                      # 失败记忆置信度降低 (×0.8)
        ...
    }
}
```

跳过 `prepare_memory()` 的 LLM proceduralization，直接由 `generate_failure_reflection()` 生成简洁反思后构造并 `commit_memory()`。


## 5. 单任务生命周期

```
           bounded_solve(task_id)
                  │
    ① RETRIEVE    │
    query = vulnerability_description + crash_type
                  │
                  ▼
        ┌─── retrieve_query() ──────────┐
        │ Phase A: 遍历 dict_memory keys │
        │   cosine similarity > δ(0.3)   │
        │ Phase B: Z-score 归一化 sim & Q │
        │   score = 0.5×sim_z + 0.5×q_z │
        │   greedy top-3 选择            │
        └────────┬──────────────────────┘
                 │ 返回 (prompt_text, memory_ids, mem_task_ids)
                 │
                 │ 展示格式:
                 │ ┌─ Successful Experiences ─┐
                 │ │ [Success #1] (Q=.., sim=.., score=..) │
                 │ │ 完整 SCRIPT + TRAJECTORY              │
                 │ ├─ Failed Experiences ─────┤
                 │ │ [Failure #1] (Q=.., sim=.., score=..) │
                 │ │ Task header + Reflection only          │
                 │ └──────────────────────────┘
                 ▼
    ② SOLVE
    Agent 接收: system_prompt + [Retrieved Experiences] + user_prompt
    在 CyberGym sandbox 里生成 PoC
                 │
                 ▼
    ③ VALIDATE
    PoC → CyberGym dual-container server
    vul 容器崩溃 + fix 容器存活 → validation_passed = True
                 │
                 ▼
    ④ Q-VALUE UPDATE (对称归因, 论文 Sec 4.3)
    对所有 retrieved_ids 统一更新:
      Q_new ← Q_old + 0.3 × (r - Q_old)
      r = +1 (成功) 或 -1 (失败)
                 │
                 ▼
    ⑤ BUILD MEMORY (去重)
    if dict_memory 中已有该 task_description → 返回已有 mem_id
    else:
      成功 → prepare_memory (LLM proceduralize) → commit_memory
      失败 → generate_failure_reflection (LLM) → 直接 commit_memory
                 │
                 ▼
    ⑥ SELF Q-UPDATE
    Q = 0.0 + 0.3 × (±1 - 0.0) = ±0.3
```


## 6. 跨轮演化流程

```
Round 1:  无记忆 → 跑 300 任务 → 每任务 build memory → save checkpoint
          (memrl_build_only=True, 只构建不检索)

Round 2+: 复用内存中的 MemRLHelper (不重新初始化)
          → 检索 + 解题 + 验证 + Q更新 + build → save checkpoint
```

关键: `MemRLHelper` 只在 evolution 开始时初始化一次，全程复用内存中的 `dict_memory` / `_q_cache`。Checkpoint 保存到磁盘仅用于 crash recovery，正常流程不需要重新加载。

### Resume 机制

- **跨轮 resume** (`--resume-from N`): 加载 Round N-1 的 checkpoint，从 Round N 开始
- **轮内 resume**: 扫描 `round_NNN/tasks/*.json`，跳过已完成任务，对已完成任务执行 MEMRL replay (重建记忆到内存)
- **Retry**: error/timeout 任务自动重试最多 2 次，timeout 逐次加 1000s


## 7. RL 配置参数

| 参数 | 值 | 论文参考 | 含义 |
|------|-----|---------|------|
| `alpha` | 0.3 | Table 8: 0.3 | Q 值学习率: `Q ← 0.7×Q_old + 0.3×reward` |
| `gamma` | 0.0 | Eq. 4 隐含 γ=0 | 折扣因子=0，只看即时奖励 (MC 风格) |
| `q_init_pos` | 0.0 | Table 8: 0.0 | 记忆统一初始 Q (不区分成败) |
| `q_init_neg` | 0.0 | Table 8: 0.0 | 同上 (统一为 0) |
| `success_reward` | 1.0 | Appx A.1: r∈[-1,1] | 成功奖励 |
| `failure_reward` | -1.0 | Appx A.1: r∈[-1,1] | 失败惩罚 |
| `weight_sim` | 0.5 | Ablation: λ=0.5 最优 | 检索打分中相似度权重 (1-λ) |
| `weight_q` | 0.5 | Ablation: λ=0.5 最优 | 检索打分中 Q 值权重 (λ) |
| `confidence_threshold` | 0.3 | δ: top-20% quantile | 相似度门槛 (Phase A 过滤) |
| `topk` | 3 | Table 8: k₂=3 (ALF/HLE) | 每次检索返回 3 条记忆 |
| `tau` | 0.25 | — | Q softmax 温度 |
| `epsilon` | 0.08 | 论文无 (纯 greedy) | 保留但引擎已改为纯 greedy (此参数不再生效) |
| `add_similarity_threshold` | 0.90 | — | dict_memory 层面去重阈值 |
| `novelty_threshold` | 0.85 | — | Curator 合并判断阈值 |

### 检索打分公式

```
score = 0.5 × ẑ(similarity) + 0.5 × ẑ(Q_value)
```

其中 `ẑ(sim)` 使用预计算的全局 `sim_norm_mean=0.186, sim_norm_std=0.094` 做 Z-score；`ẑ(Q)` 使用当前候选池的 mean/std 做 Z-score (clamp 到 [-3, 3])。

### Q 值生命周期示例

```
新建记忆 → Q = 0.0
  ↓ self-update (成功)
Q = 0.0 + 0.3×(1.0 - 0.0) = 0.3
  ↓ Round 2: 被检索，当前任务成功
Q = 0.3 + 0.3×(1.0 - 0.3) = 0.51
  ↓ Round 3: 被检索，当前任务失败
Q = 0.51 + 0.3×(-1.0 - 0.51) = 0.057
  ↓ Round 4: 被检索，当前任务成功
Q = 0.057 + 0.3×(1.0 - 0.057) = 0.34
```


## 8. 存储架构

### 内存缓存

| 缓存 | 类型 | 用途 |
|------|------|------|
| `dict_memory` | `{task_desc → [mem_id]}` | 倒排索引，检索入口 |
| `query_embeddings` | `{task_desc → [4096-d vec]}` | embedding 缓存，避免重复 API 调用 |
| `_q_cache` | `{mem_id → float}` | Q 值热缓存，检索时优先读取 |
| `_mem_cache` | `{mem_id → TextualMemoryItem}` | 对象缓存，避免重复 Qdrant 查询 |

所有缓存都有 FIFO 淘汰策略 (默认上限 5000)。

### 持久存储

```
Qdrant (本地 SQLite 模式):
  collection: memp_cybergym_user_snapshot
  vector_dimension: 4096
  distance_metric: cosine
```

### Checkpoint 结构

```
round_NNN/memrl_checkpoint/snapshot/cybergym/
  ├── cube/
  │   ├── config.json
  │   └── textual_memory.json     # 完整记忆 dump (含 payload + vector)
  ├── qdrant/                     # Qdrant 本地 SQLite 文件拷贝
  ├── local_cache/
  │   ├── dict_memory.json        # dict_memory 缓存
  │   ├── query_embeddings.json   # embedding 缓存
  │   ├── mem_cache.json          # 对象缓存
  │   └── q_cache.json            # Q 值缓存
  └── snapshot_meta.json          # 元信息 (user_id, md5, count, timestamp)
```

加载 checkpoint 时，优先从 `local_cache/` 恢复缓存 (零 API 调用)；若 local_cache 缺失则从 `textual_memory.json` 重建 (需调 embedding API)。


## 9. 关键代码文件

| 文件 | 职责 |
|------|------|
| `run_evolution.py` | 多轮演化编排、resume、retry、结果汇总 |
| `run_batch.py` | 单轮批量执行、MemRLHelper 封装、PoC 验证 |
| `MemRL/memrl/service/memory_service.py` | 核心引擎: 检索 (`retrieve_query`)、构建 (`prepare_memory`/`commit_memory`)、Q 更新 (`update_values`)、checkpoint |
| `MemRL/memrl/service/value_driven.py` | RL 组件: `RLConfig`、`QValueUpdater`、`ValueAwareSelector`、`MemoryCurator` |
| `MemRL/memrl/service/builders.py` | 记忆构建策略: `TrajectoryBuilder`、`ScriptBuilder`、`ProceduralizationBuilder` |
| `MemRL/memrl/service/updater.py` | 记忆更新策略: `VanillaUpdater`、`ValidationUpdater`、`AdjustmentUpdater` |
| `MemRL/memrl/service/strategies.py` | 策略枚举定义 (Build × Retrieve × Update) |
| `MemRL/memrl/service/procedural_memory.py` | ProceduralMemory 数据类 |
| `MemRL/memrl/utils/task_id.py` | task_id 提取工具 |
| `configs/cybergym_memrl.yaml` | 运行时配置 (LLM/Embedding/RL 参数) |


## 10. 与论文的偏离 (有意保留)

以下是我们有意偏离论文设计的地方，附原因说明：

### 10.1 记忆去重 (论文无此机制)

论文说每次交互都产生新 triplet，记忆库持续扩展。论文也承认 "periodic memory consolidation" 是 future work (Appendix G.1)。

我们的做法：同一 `task_description` 只保留一条记忆（`dict_memory` 去重），Q 值通过 RL 反复更新。原因是 CyberGym 的跨轮演化场景中，300 个固定任务 × N 轮 会产生大量完全重复的条目。去重将记忆库维持在 ~300 条。

### 10.2 成功/失败记忆差异化存储 (论文未明确规定)

论文的 Build 策略对成功和失败一视同仁做 proceduralization。我们的做法：

- **成功记忆**: 走 `prepare_memory()` → LLM proceduralization → SCRIPT + TRAJECTORY (与论文一致)
- **失败记忆**: 跳过 proceduralization，由 `generate_failure_reflection()` 生成简洁反思 (ROOT CAUSE / KEY MISTAKES / AVOID)，直接 `commit_memory()`

原因：失败的 trajectory 通常包含大量无效探索步骤，proceduralization 后仍然冗长。反思摘要更精炼、更可操作。这与原始 MemRL 仓库的 `AdjustmentUpdater._generate_reflection()` 设计思路一致。

### 10.3 检索展示分区 (论文未明确规定)

论文未规定 retrieved memories 如何展示给 agent。我们将成功记忆和失败记忆分成两个 section，并附带 Q/sim/score 信息，帮助 agent 区分"应该遵循的模式"和"应该规避的错误"。


## 11. 已修复的问题 (evo_300_20260317_v3 实验分析)

### 问题 1: 记忆去重失效

**现象**: 300 个任务在 9 轮后产生 2745 条记忆 (应为 300 条)。同一 `task_description` 每轮重复写入。

**根因**: `MemRLHelper.build()` → `commit_memory()` 路径不经过 `add_similarity_threshold` 去重检查，每次都创建新条目。

**修复**: `build()` 开头检查 `dict_memory`，若该 `task_description` 已存在则返回已有 `mem_id`。锁内 double-check 防并发。`_rebuild_caches_from_checkpoint` 对旧 checkpoint 只保留每个 task 的最高 Q 值条目。

### 问题 2: Q 值跨任务误归因

**现象**: Round 1 的 40 条成功记忆到 Round 9 有 34 条 Q 值下降 (如 1.0→-0.823)。

**根因**: 
- memory embedding key = vulnerability_description，不同任务间 cosine similarity 可达 0.81
- 原逻辑: 任何被检索到的记忆无论来源，都按当前任务结果更新 Q
- 结果: 好记忆被不相关的失败任务反复惩罚 (整体通过率仅 13%)

**修复 (v3)**: 非对称归因 — 成功时奖励所有检索到的记忆；失败时只惩罚 task_id 匹配的自身记忆。

**修复 (v4, 当前)**: 回归论文设计的对称归因 — 所有 retrieved 统一用环境 reward 更新。配合 `δ=0.3` 门槛过滤 + `Q_init=0.0` 统一初始化，减轻跨任务误归因的影响。论文指出精确归因 (如 Shapley) 是 future work。

### 问题 3: task_id 未持久化

**现象**: `extract_task_id()` 从 metadata 读取 task_id，但 `prepare_memory()` 未将调用方传入的 task_id 写入 `base_meta`。

**修复**: `prepare_memory()` 中提取 `metadata.task_id` 并写入 `base_meta["task_id"]`。

### 实验数据摘要

| 指标 | 值 |
|------|-----|
| 任务总数 | 300 |
| 总轮次 | 9 (第 10 轮运行中被终止) |
| 每轮通过率 | 12.7%–15.0% (无上升趋势) |
| 有记忆通过率趋势 | 31.4% (R2) → 22.9% (R9) ↓ |
| 始终通过的任务 | 13 (4%) |
| 始终失败的任务 | 219 (73%) |
| 不稳定任务 | 68 (23%) |
| 跨任务 similarity 分布 | mean=0.47, max=0.81 (基于 vulnerability_description) |
