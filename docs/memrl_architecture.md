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


## 2. 三大策略组合

当前配置 (`configs/cybergym_memrl.yaml`):

| 策略类型 | 当前选择 | 含义 | 替代选项 |
|----------|---------|------|---------|
| **Build** | `proceduralization` | LLM 生成 SCRIPT (高层摘要) + 保留 TRAJECTORY (完整轨迹) | `trajectory` (原样存储), `script` (仅摘要) |
| **Retrieve** | `query` | 用 task_description 原文向量做 cosine 检索 | `random` (随机), `avefact` (关键词平均向量) |
| **Update** | `adjustment` | 成功→新增记忆; 失败→LLM 反思 + 新增反思记忆 | `vanilla` (全部新增), `validation` (仅成功新增) |

这三类策略在 `MemRL/memrl/service/strategies.py` 中定义，共 3×3×3 = 27 种组合。


## 3. 单条记忆的结构

```python
TextualMemoryItem {
    id:       "mem_xxxx"                          # 唯一 ID
    memory:   "A heap-buffer-overflow occurs..."  # embedding key = vulnerability_description
    metadata: {
        type:           "procedure" | "adjustment"
        full_content:   "Task: ...\n\nSCRIPT:\n...\n\nTRAJECTORY:\n..."  # 实际注入 agent 的内容
        task_id:        "arvo:52145"              # 来源任务 ID (用于 Q 值精确归因)
        success:        true/false                # 该次执行是否成功
        q_value:        0.6                       # RL Q 值
        q_visits:       3                         # 被访问/更新次数
        reward_ma:      0.2                       # reward 指数移动平均
        confidence:     100.0                     # 记忆置信度
        strategy_build: "proceduralization"
        strategy_retrieve: "query"
        strategy_update: "adjustment"
        source_benchmark: "unknown"
        ...
    }
    vector:   [0.012, -0.034, ...]                # 4096 维 embedding (qwen3-embedding-8b)
}
```

关键设计决策:
- `memory` 字段 (embedding key) 直接使用 vulnerability_description 原文，而非 full_content
- 检索时 query 也是 vulnerability_description，因此同任务 similarity ≈ 1.0
- `full_content` 包含 LLM 生成的 SCRIPT + 原始 TRAJECTORY，是实际注入给 agent 的内容


## 4. 单任务生命周期

```
           bounded_solve(task_id)
                  │
    ① RETRIEVE    │
    query = vulnerability_description + crash_type
                  │
                  ▼
        ┌─── retrieve_query() ───┐
        │ 遍历 dict_memory keys   │
        │ 计算 cosine similarity  │
        │ Z-score 归一化 sim 和 Q │
        │ hybrid score =          │
        │   0.4×sim_z + 0.6×q_z  │
        │ ε-greedy (8%) top-3    │
        └────────┬───────────────┘
                 │ 返回 (prompt_text, memory_ids, mem_task_ids)
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
    ④ Q-VALUE UPDATE (非对称归因)
    if 成功: 对所有 retrieved_ids → reward +1.0
    if 失败: 只对 task_id 匹配的记忆 → reward -1.0
             (跨任务记忆不受惩罚)
                 │
                 ▼
    ⑤ BUILD MEMORY (去重)
    if dict_memory 中已有该 task_description → 返回已有 mem_id
    else → LLM proceduralize → commit_memory → 注册 dict_memory
                 │
                 ▼
    ⑥ SELF Q-UPDATE
    对自身 mem_id 用本任务结果更新 Q 值
```


## 5. 跨轮演化流程

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


## 6. RL 配置参数

| 参数 | 值 | 含义 |
|------|-----|------|
| `epsilon` | 0.08 | 8% 概率随机探索 (不选最优) |
| `alpha` | 0.2 | Q 值学习率: `Q ← 0.8×Q_old + 0.2×reward` |
| `gamma` | 0.0 | 折扣因子=0，只看即时奖励 |
| `q_init_pos` | 1.0 | 成功记忆初始 Q |
| `q_init_neg` | -1.0 | 失败记忆初始 Q |
| `success_reward` | 1.0 | 成功奖励 |
| `failure_reward` | -1.0 | 失败惩罚 |
| `weight_sim` | 0.4 | 检索打分中相似度权重 |
| `weight_q` | 0.6 | 检索打分中 Q 值权重 |
| `topk` | 3 | 每次检索返回 3 条记忆 |
| `tau` | 0.25 | Q softmax 温度 |
| `add_similarity_threshold` | 0.90 | dict_memory 层面去重阈值 |
| `novelty_threshold` | 0.85 | Curator 合并判断阈值 |

### 检索打分公式

```
score = weight_sim × z_norm(similarity) + weight_q × z_norm(q_value)
```

其中 `z_norm` 使用当前 candidate pool 的 mean/std 做 Z-score 标准化 (clamp 到 [-3, 3])。Similarity 的 Z-score 使用预计算的全局 `sim_norm_mean=0.186, sim_norm_std=0.094`。


## 7. 存储架构

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


## 8. 关键代码文件

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


## 9. 已修复的问题 (evo_300_20260317_v3 实验分析)

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

**修复**: 非对称归因 — 成功时奖励所有检索到的记忆 (跨任务记忆可能有贡献)；失败时只惩罚 `task_id` 匹配的自身记忆 (跨任务记忆不背锅)。

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
