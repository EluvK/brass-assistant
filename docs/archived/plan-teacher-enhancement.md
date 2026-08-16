# Rust 老师增强与训练管线重构 — 执行计划

- 日期：2026-08-03（下班前整理，次日执行）
- 状态：**计划已定，未开工**（代码改动为 0）
- 目的：按既定战略顺序推进，避免再次"乱撞"式训练。

---

## 0. 背景与核心判断（本次停下来的原因）

近两天的训练暴露了根本问题：**我们没定义"好"的方向就反复训练**，得到的数据时好时坏但没有意义。

- **镜像对战崩盘**：step5_best 在 net-vs-net 高 sims 下多座 0 分（seed 10 曾 0/83/2/2）。根因：value 头是相对 z 归一化（`selfplay.py _normalize`），网络学"比烂"而非"打好"。
- **加了 econ_head 后**：灾难局消除（min VP 0→10），但贷款仍过度（seed 10 贷 21 次）、终局收入中位数仍低。纯回归式 econ_head 引导力不足。
- **关键盲点（本日发现）**：BC 训练一直用**单步启发式**（`selfplay.py:197 choose_heuristic`）当老师，从未用过已有的 **2-ply**（`search_ai.rs`，明显更强）。2-ply 互打胜者 mean 93.9 vs 单步 ~66——**网络从烂老师起步，起点天然低 15-20 分**。

**用户裁决的战略顺序（本计划严格遵守）**：

1. 继续增强 Rust 老师（可含分时代评价函数）
2. 优化 Rust 老师速度（给后续训练加速）
3. 设计更充分的 value 指标去训练
4. 精选更好的对局记录作为 BC 数据
5. 在能几乎一直出高分对局的情况下再考虑 self-play

---

## 1. 现状事实（已实测，非推测）

### 1.1 老师强度

| AI | avg 每玩家 VP | 胜者 mean | 局速 | 数据来源 |
| --- | --- | --- | --- | --- |
| 1-ply 启发式 | ~58-70 | ~66 | 240 局/s | `brass-engine.exe 50 4 heuristic` |
| **2-ply** | 73.4 | **93.9**（500 局 mean，max 152） | 42 局/s | `seed_scores_2ply_0_499.csv` |
| 2-ply 胜者 ≥130 | 4/500 | 胜者 ≥100 | 174/500 | 同上 |

- **目标差距**：胜者 mean 93.9 → 目标 130，差 36；玩家 mean 73.4 → 目标"平均 100+"，差 27。
- 2-ply 有完善的护栏（`end_of_turn_penalty`、era-aware ALPHA、贷款收入下限），不是逻辑烂，是 **1-ply/2-ply 深度不足 + 局部视角**。

### 1.2 训练成本（Python 侧实测）

| 老师 | BC 生成速度 | BC 2000 局 | 说明 |
| --- | --- | --- | --- |
| 1-ply（当前） | 0.46 s/局 | ~15 分钟 | 快但弱 |
| 2-ply | **3.41 s/局** | **~113 分钟** | 7.4× 慢 |

**结论**：直接切 2-ply 老师重训要花 2-3 小时，收益不确定。必须先让老师够强、够快，再训练。

### 1.3 老师速度瓶颈（explore agent 分析，有行号依据）

`choose_action_2ply` 每次决策内部：
- **~8 次 `candidate_actions_k` 调用**、**~30-50 次 `state.clone()`**、~10 次 apply_move
- 每次 `candidate_actions_k` 对全部 build/network/develop/sell/loan 候选**全量重算**，单 build 候选约 8-10 次 BFS/全板扫描
- 每决策估算 **5000-9000 次 BFS/扫描**，主导是重复计算，不是 clone

**三个提速优化（按性价比）**：
1. **决策内 memo cache（预估 4-6×）**：`connected_locations(loc)`/`find_coal_sources(loc)`/`find_iron_sources()` 按 loc/state 缓存。`GameState` 无缓存字段，跨候选零共享。
2. **砍 2-ply 内层嵌套贷款前瞻（1.5-2×）**：`score_loan_result → best_same_turn_after_loan` 递归 `candidate_actions_k(3)`，与 2-ply 顶层重复。
3. **预计算 `score_best_network_double` 单连接分数（1.3-1.8×）**：N² 配对重复计算。

> 目标：Python 侧 2-ply 从 3.4s/局 → ~1s/局，BC 2000 局从 113 分钟 → ~35 分钟。

---

## 2. 前置基建（建议第一步）：`sweep_scores.rs` 批量统计工具

**现状缺口**：`seed_scores_2ply_0_499.csv` 是临时手工生成，**没有可复用的批量统计 bin**。没有它，老师增强无法快速验证。

**新增 `src/engine/src/bin/sweep_scores.rs`**：
- 参数：`<start_seed> <end_seed> <policy(heuristic|2ply|mcts)> [sims]`
- 并行（rayon，参考 `main.rs`）跑 N 局，输出 CSV：`seed,p0,p1,p2,p3,avg`
- **追加经济列**：每玩家终局 income + money + 运河末收入（与方向 3 的 value 指标对齐）
- 这是方向 1/2 的唯一验收基准（固定 seed 段与 500 局 CSV 对齐）

---

## 3. 分步执行细节

### 方向 1：增强 Rust 老师（纯 Rust，秒级验证）

2-ply 起点：胜者 mean 93.9，目标 130。

**1A. 分时代评价函数增强**
- `evaluate_position`/`vp_equivalent`（`heuristic_ai.rs:183/269`）已有时代感知（canal 收入权重高、late-rail 收入归零），可强化：
  - 运河时代末：奖励"运河末收入高 + 翻面数"（用户洞察：一时代烂后面高不了）
  - 铁路时代：收入权重进一步压低、VP/翻面权重提高
  - 增加"运河末里程碑"显式信号
- 验证：`sweep_scores.rs` 对比 500 局前后分数 + 运河末经济

**1B. 权重与超参调优**
- `FIRST_ACTION_K`/`SECOND_ACTION_K`（3/2 → 试探 4/3）、`ALPHA`（0.6）
- 每轮秒级验证

**1C. 战术模块增强**（若 A/B 不够）
- 网络/卖货/啤酒时机，参考用户人类经验

### 方向 2：优化老师速度（给训练加速）

按 §1.3 的 3 个优化点实施，**先 memo cache（收益最大）**。
验证：`sweep_scores.rs` 跑 2-ply 测速（局/s）+ 确认分数不退化。

### 方向 3：设计更充分的 value 指标（Python，老师强了再动）

- **value 头多分量**：`(终局VP, 终局收入, 运河末收入, 破产标志)`，各自独立监督
- **分段监督**：运河样本→运河末收入（用户洞察：比终局更重要），铁路样本→终局
- **绝对 VP**：不再归一化成相对 z（避免"比烂"）
- **econ_head 定位修正**：上一轮纯回归式 econ_head 引导力不足，本轮改为"锚定健康区间 + 破产硬惩罚"，不只是预测
- **用户"只训练一时代"验收**：可先单时代（运河）训练验证老师经济，再全时代

### 方向 4：精选 BC 对局记录

- 用 `sweep_scores.rs` 从高分对局（胜者 ≥120 或经济健康）中**精选**作为 BC 数据，而非全部 2000 局
- 即使老师不够完美，也只教"打得好"的样本

### 方向 5：self-play 训练（最后）

- 仅当老师能"几乎一直出高分对局"（胜者稳定 120+、经济健康）后，再启动门控 self-play

---

## 4. 待用户裁决的关键问题（明日开工前确认）

1. **"130 分/平均 100+"的口径**：按 2-ply 500 局，胜者 mean 93.9、玩家 mean 73.4。"平均 100+"指每局 4 人平均还是胜者平均？决定增强目标。
2. **验证闭环工具（sweep_scores.rs）是否第一步先做**？（强烈建议，否则方向 1/2 盲调）
3. **分时代评价的具体形式**：(a) 强化现有 `evaluate_position` 时代权重，还是 (b) 新增独立"运河末健康评分"函数？
4. **执行顺序**：确认"工具先行 → 1 → 2 → 3 → 4 → 5"。

---

## 5. 验收标准（每个方向的出口条件）

| 方向 | 验收 |
| --- | --- |
| 前置工具 | sweep_scores.rs 输出与现有 CSV 对齐，含经济列 |
| 方向 1 | 2-ply 500 局胜者 mean 从 93.9 提升（每轮 diff），逼近 130；终局收入中位数接近 15-25 |
| 方向 2 | 2-ply 局速提升 4-6×，分数不退化 |
| 方向 3 | 训练后 net-all-vs-all：负收入占比↓、终局收入↑、benchmark 胜率不退化 |
| 方向 4 | BC 用精选高分对局后，网络贪心 VP 显著高于全量 BC |
| 方向 5 | 门控 self-play 训练不退化，镜像对战稳定 |

---

## 6. 环境与参考

- 引擎：`src/engine`，改后 `cargo build` / `maturin develop`
- 现有 CSV 对照：`src/engine/seed_scores_2ply_0_499.csv`
- 2-ply 逻辑：`src/engine/src/search_ai.rs`
- 启发式评分：`src/engine/src/heuristic_ai.rs`（`evaluate_position`/`vp_equivalent`/`candidate_actions_k`）
- 参考 AI：`reference/npow-brass-birmingham/js/aiPlayer.js`（结构与我们的启发式同源，仅作思路参考，不直接移植）
- 回放工具：`src/ai/experiments/replay_net.py`（`--all-net` 镜像对战、`--multi` 批量）
- 可靠评估：`src/ai/experiments/benchmark.py`（20 固定 seeds）

> 规则/架构背景见 `AGENTS.md`、`docs/architecture.md`、`docs/handoff/0803.md`。
