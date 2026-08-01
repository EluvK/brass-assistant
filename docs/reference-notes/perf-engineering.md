# 阶段 ② 性能工程 — 里程碑记录

## 目标

自对弈吞吐量 ≥ 1000 局/s（4 人局，heuristic AI），为后续 MCTS / Self-Play 铺路。

## 结果

| 配置 | 优化前 | 优化后 | 提升 |
| --- | --- | --- | --- |
| heuristic（16 核默认） | 75 局/s | **~1200 局/s** | **16x** |
| 2ply（16 核默认） | 19 局/s | **~280 局/s** | **15x** |
| 单线程 heuristic | 75 局/s | ~90 局/s | 1.2x |
| 并行扩展性 | — | 4→8→12→16 核：345→641→886→1006 局/s | 近线性 |

> 注：`main.rs` 支持第 4 个参数指定线程数，如 `brass-engine 4000 4 heuristic 8`。

## 优化项分解

### 1. 图连通性 BFS 重构（`graph.rs`）— 单线程 +50%
瓶颈：`connected_locations` / `find_coal_sources` 每次调用都对 39 条连接线性扫描、
`Vec::contains` 做 O(n) 去重、堆分配 `VecDeque`。被每个行动决策反复调用数十次。

修复：
- **静态邻接表** `adjacency()`：一次性构建 `(Loc, conn_id)` 邻接表，BFS 只遍历本节点邻居（`OnceLock` 缓存）
- **u32 位掩码 visited**：27 个位置用 `1u32 << loc` 去重，O(1) 而非 O(n)
- **定长栈数组队列** `[Loc; 27]`：去掉 `VecDeque` 堆分配
- `is_in_network` 用预计算 `loc_connections()` 只检查邻接连接

### 2. 构建目标合法性预计算（`rules.rs`）— 单线程 +19%
- `get_valid_build_targets` 内 `vacant_single_icon_exists` 在每 (slot, industry) 迭代里
  全表扫描 → 提升为每次调用只算 6 种产业一次。
- `calculate_build_cost_with_sources`：铁源（位置无关）每次 `get_valid_build_targets`
  调用只算一次；煤源每个位置只算一次（同一位置的多个槽位复用）。

### 3. Rayon 并行化（`main.rs`）— 15-16x
- 每局独立（各自 seed 的 `StdRng`），天然可并行
- `into_par_iter().map().reduce()` 聚合每局统计（wins/vp/built/flipped/links）
- 自建 `ThreadPool` 支持指定线程数
- 结果与串行完全一致（确定性 seed，每局同一 rng 序列）

## 正确性验证

- 11 个单测全过
- 8000 局 avg VP ≈ 32、四家胜率 ~25%，与优化前一致
- seed=7 replay 单局动作序列逐条一致
- clippy 无新增 warning（pre-existing 警告未动）

## 未做 / 后续可做

- `find_beer_sources` 中 `connected_locations` 已去重为一次（跨对手/商家复用）
- MCTS 需要更细粒度的并行（单局内并行模拟），而非仅局间并行
- 若需更高吞吐：`GameState` 内存布局、避免 2ply 中 state.clone() 的重复分配
- 2ply（search_ai）仍依赖 `state.clone()`，MCTS 场景可考虑轻量快照

## 对 MCTS 的意义

- 单局串行 ~90 局/s ≈ 1.1 万步/s，单步合法动作生成已足够快
- 16 核并行下 1200 局/s，可支撑自对弈数据生成
- 下一步 MCTS 时，模拟器（随机/启发式 rollout）可直接复用此优化后的引擎
