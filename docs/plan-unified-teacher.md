# Rust 统一老师重构计划 — 分时代策略引擎（目标：胜者 mean ≥130）

- 日期：2026-08-04
- 状态：**计划已定，未开工**
- 前置：`docs/plan-teacher-enhancement.md`（既有方向 1-5 的宏观顺序，本计划为方向 1 的落地细化）
- 目的：把 Rust 侧策略从"heuristic + 2ply 两层、~20 处散落 if-era 分支"重构成**单一默认老师 + 4 段时代 Profile**，并按用户讲述的人类策略重写评分，目标把 2-ply 老师胜者 mean 从 95.1 提升到 **130+**，同时守住速度。

---

## 0. 背景与关键决策（用户已裁决）

### 0.1 目标与口径
- **目标：胜者 mean ≥130**（500 局、4 人局，`sweep_scores.rs` 验收）
- 现状：2-ply 胜者 mean **95.1**、玩家 mean **75.6**（2026-08-04 修复引擎后实测）
- 速度：当前 2-ply 41 局/s；重构后目标 **≥35 局/s**（新增信号开销 ≤15%）

### 0.2 已裁决决策
| # | 决策 | 内容 |
| --- | --- | --- |
| 1 | 统一范围 | 老师 = 单一入口 `choose_action`（内含同回合 2-ply）；**MCTS 叶复用同一评估器**，不再有独立 `evaluate_position` 分支 |
| 2 | 阶段划分 | **每时代按 `state.round ≤4 / >4` 分两段** → 4 个 Profile：`Canal-Early / Canal-Late / Rail-Early / Rail-Late` |
| 3 | 研发限制 | 保留现有硬限制：`BAN_DEVELOP_IRON_LV2_PLUS`、运河 4 次/铁路 1 次开发上限 |
| 4 | 流派编码 | **量化生产计划**：选主攻产业 X* + 计划产出 k* + 啤酒需求 M，建/开发/修路给软加成 |
| 5 | 产/路权衡 | **统一 VP 当量比较**：产业价值 = VP×翻面概率×双计（啤酒可控才给高翻面概率）；路价值 = 预期得分+解锁价值+酒锁定 |
| 6 | 推倒方式 | 先写本计划文档，审阅后再动代码 |
| 7 | 铁研发禁令 | **运河时代保持禁令**（禁研发 2/3/4 级铁，运河时代基本是坏招）；仅"炸别人铁+市场无铁"联动才可能例外。**铁路时代是否放开待 Rail 阶段再定**（§7.1） |
| 8 | 启动贷款 | Canal-Early round 1-2、cash<£18 时启动贷款（配合现有收入下限护栏） |

---

## 1. 现状事实（代码层面，供重构对照）

### 1.1 代码结构
- `heuristic_ai.rs`（1925 行）：1-ply 贪心评估器
  - ~20 处 `if state.era == Canal/Rail` 时代分支散落各函数
  - 大量硬编码加成：`market_adjust` / `coal_spike_bonus` / `rail_coal_shortage_bonus` / `beer_bonus` / `level_adjust` / `canal_beer_drain_bonus` / `urgency_bonus` ...
  - 权重常量：`VP_WEIGHT=1.0` / `BASE_MONEY_WEIGHT=0.12` / `BASE_INCOME_WEIGHT=0.35` / `FLEX_WEIGHT=0.8`
  - 临时禁令：`BAN_BUILD_LV1_BREWERY=true` / `BAN_DEVELOP_IRON_LV2_PLUS=true`
- `search_ai.rs`（148 行）：2-ply 包装器
  - `ALPHA=0.6` / `LATE_RAIL_ALPHA=0.35` / `FIRST_ACTION_K=3` / `SECOND_ACTION_K=2`
  - `end_of_turn_penalty`（低现金护栏）+ `combo_alpha`（时代衰减）
- `mcts_ai`：用 `evaluate_position`（heuristic_ai.rs:183）做叶子评估

### 1.2 现状强度（修复引擎后，2026-08-04 实测）
- heuristic 500 局：玩家 mean **63.2**，income/人 7.0
- 2-ply 500 局：玩家 mean **75.6**，胜者 mean **95.1**（max 166），income/人 8.7，无非法动作
- 500 种子逐局：`seed_scores_2ply_0_499_fixed.csv`（含 income/money/canal_income 经济列）

---

## 2. 用户讲述的人类策略（重写评分的事实来源）

> 以下全部为用户 2026-08-04 口述，禁止自行推断替代。

### 2.1 运河前半段（Canal-Early, round 1-4）
1. **确立流派**：决定本局主攻产业方向（依据：初始手牌 + 市场随机 + 空闲单图标格）
2. **经济引擎**：先建煤/铁（或能立刻卖的初级板块），市场卖出回血，支撑并偿还启动贷款
3. **抢酒桶（几乎必抢）**：优先**开发 1 级酒厂** → 尽早建 2 级+酒厂，抢先占据酒桶位置
   - **保持理智**：只在**铁价 ≤£2** 时研发酒厂（造酒桶也要铁）；**铁价 ≥£3** 不研发，先想办法补铁
4. **适当开发**：顺带研发其他低级建筑

### 2.2 运河后半段（Canal-Late, round 5-8）
1. **主建流派产业牌**：重心转到建自己选定的产业（棉/陶/制造）
2. **补市场缺口**：市场煤/铁空缺过大时适度补建，高价卖出强化经济
3. **只修关键路**：只修"不修就卖不掉货 / 解锁下一动"的路，**不用多余动修断头路**
4. **末期贷款**：最终出货前，用多余行动贷款为铁路准备资金
   - 判据：**现金 <£30 且下一动卖货能立刻回高收入**（该动贷款 -3 收入等级无实质影响）

### 2.3 铁路前半段（Rail-Early, round 1-4）
1. **核心是修路**：围绕修路展开，网络权重最高
2. **补煤**：铁路需要煤+大量经济，中途不得不补建煤矿
3. **选路标准**：预测每条路最终得分抢占优先路；**低分但有战略价值的路也关键**——如连接两处乡村酒厂的路（锁定啤酒供应）
4. **双铁路**：条件允许时**酒桶 + 一动双铁路** 优于两动修两条——更贵但多赚酒厂产业分，通常更优

### 2.4 铁路后半段（Rail-Late, round 5-8）
1. **灵活收官**：不同流派做不同事
2. **有产业流派**：判断自己最终有**空余酒桶**（翻建低级酒产业 / 提前控制乡村酒桶 / 市场酒可用）确保能卖掉出的货 → **修建价值高于剩下路分的产业建筑** → 最后打"酒卖"收分
   - 争市场酒时：打"产业+卖"组合动作直接消耗市场酒
3. **无酒可用**：产业价值不够高或完全没酒桶 → **继续修路**，3-4 分的路也比没有分好
4. **补铁/煤是全局话题**：铁路建 4 级铁厂也有 9 分，是选项

---

## 3. 重构设计

### 3.1 架构：单一入口 + EraProfile

```
choose_action(state) -> Decision            // 唯一默认入口（老师 + MCTS 叶共用）
   ├─ era_profile(state) -> EraProfile       // 4 档：Canal-Early/Late, Rail-Early/Late
   ├─ plan(state, pid) -> Plan               // 流派：{X*, k*, M}，每个 era_profile 一次
   ├─ candidate_actions(state)               // 各动作统一生成 Top-K
   └─ evaluate_move(state, mv, profile, plan) -> f64
```

**EraProfile 结构**（收敛全部时代逻辑）：
```rust
struct EraProfile {
    income_w: f64,      // 收入权重（Canal-Early 最高，Rail-Late 归零）
    money_w: f64,       // 现金权重
    develop_w: f64,     // 开发权重（Canal-Early 高）
    build_w: f64,       // 建厂权重（Canal-Late / 啤酒门控时高）
    network_w: f64,     // 建网权重（Rail-Early 最高）
    sell_w: f64,        // 卖货权重（era 末冲刺高）
    loan_w: f64,        // 贷款权重（开局 + 运河末两峰）
    beer_gate: f64,     // 啤酒门控强度（Rail-Late 最高）
    double_vp: f64,     // 2级+板块双计乘数
    alpha: f64,         // 2-ply 组合折扣（替代 LATE_RAIL_ALPHA）
}
```
- **删掉 heuristic_ai 中 ~20 处 `if era` 分支**，全部由 profile 参数驱动
- `estimate_rounds_remaining` 仅保留用于非固定判据（如末日建厂护栏），阶段切分一律用 `state.round`

### 3.2 流派 = 量化生产计划（`plan()`）

```
plan(state, pid) -> Plan {
    // 对每个可卖产业 X（棉/陶/制造）：
    //   tiles_left(X)      = 剩余板块数
    //   slots_avail(X)     = 可到达空格数（在网络内 或 经关键路可达）
    //   flip_prob(X,loc)   = 翻面概率（必须含"啤酒可达"门控，见 §3.4）
    //   plan_score(X)      = min(tiles_left, slots_avail) * avg_vp * flip_prob
    // 选 X* = argmax plan_score
    // k* = 可实际翻面数量；M = 卖掉 k* 需啤酒桶数（= k* 对应的 beers_to_sell 之和）
    Plan { industry: X*, count: k*, beer_needed: M }
}
```
- **软加成**（不硬约束，防开局误判锁死）：
  - 建 X* 产业：`score += score * 0.10`
  - 开发朝向 X*：`+10%`
  - 修路连到 X* 可建/可卖城市：路价值 `+10~20%`
- 流派每 **era_profile 边界**重算一次（Canal-Early 开局、Canal-Late、Rail-Early、Rail-Late），非每动重算（省开销）

### 3.3 关键路编码（`critical_link_value`）

路的综合价值 = 三部分之和，替代现有 `hub_bonus` / `potential_link_vps`：
```
critical_link_value(conn) =
    route_score(conn)                 // 已翻图标 + 未来可翻图标（link_vp_potential 已有雏形）
  + unlock_value(conn)                // 连到的目标城市里可立即/下一步建的高价值产业（铁厂、流派产业）
  + sell_unlock(conn)                 // 连通到接受己方可卖产业的商家（不修就卖不掉）
  + beer_lock(conn)                   // 连到乡村酒厂（锁啤酒供应）
  + double_rail_sync(conn)            // 与另一条路组成双铁路的协同（Rail-Early 加权高）
```
- **断头路罚**：两端都不含己方可建产业空格、可卖商家、酒厂的路 → 重罚（Canal-Late / Rail-Late 无酒时尤其）

### 3.4 啤酒门控的翻面概率（产业价值核心）

改造 `estimate_flip_probability`（heuristic_ai.rs:348）的可卖产业分支：
- 只有"连通接受该产业的商家 **且** 啤酒可达"才给高翻面概率
- **啤酒可控性**：己方未翻面可卖板块的 `beers_to_sell` 需求 + 铁路建网缓冲 ≤ 可用酒桶（己方酒厂+乡村酒厂+市场桶），超出即降低翻面概率
- **产业价值 = VP × flip_prob × double_vp**，与路价值统一到同一 VP 当量比较（Rail-Late 的"建产业 vs 铺路"就用这个）

### 3.5 贷款节奏（两峰模型）

改写 `score_loan_result`（heuristic_ai.rs:1664）：
- **启动贷款峰**（Canal-Early, round 1-2, cash<£18）：贷款为经济引擎启动
- **运河末贷款峰**（Canal-Late, round 6-8）：**新增判据 = cash<£30 且 下一动卖货能回高收入** → 显著上调（+2~3）
- **中间段**：靠收入偿还，不主动贷
- **护栏保留**：收入下限 floor（-8 硬罚）、rich 惩罚、late_era 惩罚

### 3.6 统一老师入口 + MCTS 叶复用

- `choose_action`：1-ply 候选 + 同回合 2-ply 组合加分（保留 ALPHA 机制，alpha 进 profile）
- **MCTS 叶子**：不再用独立的 `evaluate_position` 快照，改为"执行候选动作后的 `evaluate_move` 结果 + 位置快照混合"，确保 MCTS 与老师用同一评价体系
- 删除 `evaluate_position` 里的时代特判，改用 profile 参数

---

## 4. 分步执行（每步用 sweep_scores 500 局验收）

| 步骤 | 内容 | 验收 |
| --- | --- | --- |
| 0 | 验证闭环就绪 | `sweep_scores.rs` 与 `seed_scores_2ply_0_499_fixed.csv` 对齐 ✅ |
| 1 | **架构重构**：EraProfile + 收敛 if-era 分支 + 统一入口 + MCTS 叶复用 | `cargo test` 绿；500 局胜者 mean 不退化（≥95） |
| 2 | **Canal-Early 策略**：流派 plan() + 启动贷款 + 经济引擎 + 铁价门控抢酒桶 | 500 局：运河末 income 中位数↑，胜者 mean ↑ |
| 3 | **Canal-Late 策略**：主建产业 + 补市场缺口 + 关键路 + 末期贷款 | 运河末收入健康、贷款两峰可见 |
| 4 | **Rail-Early 策略**：修路权重最高 + 补煤 + 双铁路协同 + 选路=预期得分+战略价值 | 铁路早期 network 占比↑，双铁路使用↑ |
| 5 | **Rail-Late 策略**：啤酒门控建产业 vs 铺路统一 VP 当量 + 补铁/煤收官 | 胜者 mean 逼近 130，零灾难局（min≥60） |
| 6 | **速度保障**：实测局速，不足 35 局/s 则加 memo cache（见 plan-teacher-enhancement §1.3） | ≥35 局/s，分数不退化 |

## 5. 验收标准

- **强度**：2-ply（即统一老师）500 局胜者 mean **≥130**（当前 95.1）
- **健康度**：终局收入中位数 ≥10，负收入玩家占比 <10%，无破产
- **行为**：4 Profile 动作分布符合 §2 描述（运河早期抢酒桶/经济引擎、运河末关键路+末期贷、铁路早修路/双铁、铁路晚啤酒门控）
- **正确性**：`cargo test` 全绿，无非法动作（sweep_scores 的 illegal 列=0）
- **速度**：≥35 局/s
- **MCTS 对齐**：MCTS 叶复用统一评估器后，`benchmark.py` 20 固定 seeds 胜率不退化

---

## 6. 风险与质疑点（推倒旧代码时保持警惕）

1. **硬限制 vs 新策略的冲突**：`BAN_BUILD_LV1_BREWERY`（禁 1 级酒厂）与"抢酒桶"兼容（抢的是 2 级+）；`BAN_DEVELOP_IRON_LV2_PLUS` 与"铁价≥£3 不研发"兼容；**但**新策略"补铁/煤是全局话题"（Rail 建 4 级铁厂 9 分）与 `BAN_DEVELOP_IRON_LV2_PLUS` 冲突——4 级铁厂需开发解锁，需用户裁决是否放开铁研发禁令
2. **流派量化过强 → 开局误判锁死**：X* 选错导致整局偏航。v1 用软加成（±10%），验证后再硬化
3. **统一 VP 当量的标定**：产业价值 vs 路价值量纲要一致，否则某段失衡。用 sweep_scores 逐段校准
4. **MCTS 叶复用评估器变贵**：若 evaluate_move 显著慢于 evaluate_position，MCTS sims/s 下降，需评估器带缓存或分层
5. **末期贷款判据**：cash<£30 + 下一动能回高收入——"下一动能回高收入"需近似（如已有可卖货+酒可达），不可过度模拟

---

## 7. 待用户裁决（开工前确认）

1. **铁研发禁令（运河已裁决：保持禁令）**：Rail 建 4 级铁厂（9 分）需开发解锁，与 `BAN_DEVELOP_IRON_LV2_PLUS` 冲突——**铁路时代是否放开"铁 ≥2 级可研发"？**（运河时代保持禁令已定）
2. **启动贷款额度（已裁决）**：Canal-Early round 1-2、cash<£18。
3. **流派重算频率**：era_profile 边界重算（4 次/局）是否可接受，还是每轮重算（更准但更贵）。

---

## 8. 环境与参考

- 引擎：`src/engine`，改后 `cargo build` / `maturin develop`
- 验收：`cargo run --release --bin sweep_scores -- 0 500 2ply`（对应当前 `seed_scores_2ply_0_499_fixed.csv`）
- 代码参考：`heuristic_ai.rs`（评分函数）、`search_ai.rs`（2-ply）、`mcts_ai.rs`（叶评估）
- 规则背景：`AGENTS.md`、`docs/rules/brass-birmingham-rules.md`
- 策略事实来源：本文 §2（用户口述，勿自行推断）
