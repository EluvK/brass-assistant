# Heuristic AI 深度重构计划

## 目标

把 ~2450 行散乱、魔法数字遍布、量纲不一的启发式 AI，重构为「**统一量纲 + 集中参数 + 收敛上下文**」的架构，让后续调参/改逻辑变得容易。行为允许改变（用户已确认）；对外 API 签名（`Decision`、`choose_action`、`candidate_actions_k` 的 Top-K 合同等）保持不变，mcts_ai/nn_mcts/replay/PyO3 桥不需要改动。

## 一、目标文件架构（include! 全部消灭）

```
engine/src/ai/heuristic_ai/
├── mod.rs          # 门面：对外 API + candidate_actions_k 编排（原 heuristic_ai.rs 主体）
├── config.rs       # HeuristicConfig：全部 100+ 参数，按主题分组，Default=迁移现值
├── context.rs      # EvalContext：一次候选批构建一次，承载轮次/era 便捷函数
├── value.rs        # 统一币种：vp_equivalent、PlayerVpEstimate（对齐 scoring.rs 只读版）、
│                   #   市场稀缺度/煤价热度（消除 14/10 硬编码与重复公式）
├── board.rs        # 公共盘面查询：merchant 可达（5处合一）、啤酒可用（4处合一）、
│                   #   resource_source_ratio、overbuild 损失、beer 需求等
├── probability.rs  # 唯一一套翻转概率模型（合并 estimate_flip_probability 与
│                   #   plan_flip_probability 两套互不相干的模型）
├── build.rs / network.rs / develop.rs / sell.rs / loan.rs / scout_pass.rs
│                   # 真子模块，每个 scorer 拆成命名的「评分因子」子函数
├── cards.rs        # card_keep_score 系列从主文件迁出、参数化
├── plan.rs         # 保留（流派计划），翻转概率改调 probability.rs
└── lookahead.rs    # 保留 2-ply，阈值改走 config/context
```

## 二、核心类型设计

### 1. HeuristicConfig（config.rs，模仿 MctsConfig 先例）

```rust
pub struct HeuristicConfig {
    pub value: ValueWeights,     // vp/money_base/income_base/flex 等全局换算基数
    pub era: EraWeights,         // 四个 Phase 的 profile 公式参数（折现率也按 phase 给）
    pub build: BuildWeights,     // market 尖峰、shortage、beer、性价比等系数
    pub network: NetworkWeights, // 探索、plan bonus、beer lock、双铁 tempo 等
    pub develop / sell / loan / scout: ...Weights,
    pub cards: CardWeights,      // card_keep_score 全部系数
    pub lookahead: LookaheadParams, // k、现金红线、end-turn 惩罚尺度
    pub guardrails: Guardrails,  // 3 个 bool 开关 + develop 守栏系数（不再是散落 const）
}
```

每个字段带 doc 注释（含义、调节方向）。`Default` 迁移现值，保证初始行为接近现状。

### 2. EvalContext（context.rs，解决「轮次逻辑繁杂」）

```rust
pub struct EvalContext<'a> {
    pub cfg: &'a HeuristicConfig,
    pub pid: usize,
    pub phase: Phase,           // era_phase() 收敛于此，唯一权威
    pub profile: EraProfile,    // 由 cfg.era 生成
    pub rounds_remaining: f64,
    pub era_frac: f64,          // rounds/8 公式唯一一处（现 plan.rs/lookahead.rs 两处）
}
impl EvalContext<'_> {
    pub fn money_value(&self, pounds: f64) -> f64;   // £→VP当量（替代 5 种散乱比例）
    pub fn income_value(&self, levels: f64) -> f64;  // 收入档→VP当量（替代 4 种比例）
    pub fn future_discount(&self) -> f64;            // 按 phase 的未来收益折现
                                                    // （替代 network.rs/sell.rs 两套
                                                    //   互不一致的 round 三段式硬编码）
    pub fn is_era_endgame(&self) -> bool;            // 统一收官判定（替代 sell/loan 各自阈值）
    pub fn is_canal(&self) -> bool; pub fn is_rail(&self) -> bool;
}
```

所有 scorer 统一签名 `fn score_xxx(state: &GameState, ctx: &EvalContext, ...) -> ScoreParts`。以后「不同轮次权重不同」= 改 context 方法或 config 的 phase 表，一处生效。

### 3. ScoreParts（统一量纲，解决「计分标准不一致」）

```rust
/// 所有动作类型共用的评分分量；total() 是唯一合成点。
pub struct ScoreParts {
    pub vp: f64,        // 期望 VP（flip 期望、link VP 期望）
    pub money: f64,     // £ 变化（负=支出）
    pub income: f64,    // 收入档变化
    pub flex: f64,      // 手牌灵活度变化
    pub strategic: f64, // 战略/位置价值：探索、啤酒锁、流派、tempo、尖峰机会（VP当量刻度）
    pub risk: f64,      // 惩罚：滞销、超限、饱和（负值）
}
impl ScoreParts { pub fn total(&self, ctx: &EvalContext) -> f64 { ... } }
```

迁移标定原则：能归入真实量纲（vp/money/income/flex）的归入；位置/tempo/协同归 `strategic`；惩罚归 `risk`。初值尽量等价迁移现值，明显错位的修正（Scout 升到可比量纲、Pass 不再用魔法 -5.0 而是自然≈0、Network 的手牌×0.6 不再冒充 VP 改归 flex、Develop 的运河 ×2.0 变成 strategic 里的命名权重）。`derive(Debug)`，加一个 explain 风格测试展示分量构成，方便调试。

### 4. VP 计算收敛（value.rs）

- `PlayerVpEstimate { flipped, unflipped_expected, link_current, link_potential }`：唯一只读 VP 估算，对齐 `scoring.rs` 的结算逻辑（link_vp + merchant 2 分 + via_farm）
- `evaluate_position`（MCTS 叶子）重写为基于它 + ctx 权重；「收入 ×3.0」变成 config 命名参数 `leaf_income_scale`
- build/network 各自的 VP 片段改调 value.rs 组件，消除 5 处各自为政

## 三、实施步骤（每步保持可编译）

1. **拆假模块**：`include!` 6 文件 → 真 `mod`（机械改动：补 use、修可见性、`pub(super)` → 正确路径），行为不变，cargo test 过
2. **基础设施**：新建 config.rs / context.rs / value.rs / board.rs / probability.rs（此步旧 scorer 不动，新旧并存）
3. **逐 scorer 迁移**：build → network → develop → sell → loan → scout_pass → cards，每个：签名改收 ctx、内部拆成命名的评分因子子函数（228 行的 `score_build_candidate` 拆成 ~6 个因子函数，如 `market_value()` / `beer_economy()` / `cost_efficiency()`）、输出 ScoreParts
4. **主编排迁移**：mod.rs 的 `candidate_actions_k` 构建 EvalContext 传下去；重写 `evaluate_position`；lookahead 阈值走 config；`pick_build_card` 比较器预计算（消除 O(n log n)×全量枚举）；`score_loan_result` 复用传入 plan
5. **清理**：死代码（loan.rs 注释块）、`partial_cmp().unwrap()` → `total_cmp`、`panic!` → `debug_assert!`+保守 fallback、`show_era_profile` 改成真断言测试
6. **文档同步**：更新 heuristic_ai 模块头注释与 docs/ 下 AI 相关文档（新架构、调参入口）

## 四、验证

- `cargo test`（engine_tests.rs 现有契约：候选合法且永不空、choose_action 返回有限分——这些是接口兼容的回归网）+ `cargo clippy` 全绿
- 每个 scorer 迁移后跑全部契约测试；新增 config/EvalContext/ScoreParts 单测
- 固定 seed 对局 smoke test（不 panic、动作合法）
- Python 侧接口未动无需改代码；teacher 分数分布会变（已确认接受），如需可后续重跑 bootstrap_imitation

## 五、明确不做（本次范围外）

- 不改 mcts_ai/nn_mcts/replay/PyO3 的任何调用代码
- 不做数值调优（ScoreParts 初值≈现状迁移，调参留给后续用 config + sweep_scores 进行）
- 不动 reference 目录与规则实现（scoring.rs 等）