# 阶段 ③ MCTS（ISMCTS + Determinization）— 里程碑记录

## 目标

在 Rust 引擎上实现带 Determinization 的 MCTS 搜索，并用 1-ply 启发式作为 Prior 与叶评估，
性能目标：单决策 5000-10000 次模拟 ≤ 0.5-2 秒；强度目标：固定 seed 下对战 2-ply 有胜率 / Avg VP 提升。

## 结果

| 指标 | 数值 |
| --- | --- |
| 性能（5000 sims / depth6 / OnePly） | **1.19s / 决策**（达标 ≤2s） |
| 性能（10000 sims） | 2.35s（略超预算，推荐用 5000） |
| MCTS vs 2-ply 座位胜率（40局旋转座位） | **27.5%**（4人局理论均衡 25%） |
| MCTS avg VP vs 2-ply avg VP | **43.2 vs 35.3（+7.9）** |
| 单测 | 15 个全过（新增 4 个 MCTS/弃牌/牌池测试） |

## 架构（`src/engine/src/mcts_ai.rs`）

```
每次模拟（simulation）：
  1. Determinize 一次：从隐藏牌池采样对手手牌（deck_composition − 己方已知手牌）
  2. 从根进行 depth-N 确定性浅搜索（默认 N=6）
  3. 叶节点：对每个玩家做位置评估（MaxN 向量）→ 归一化到 [-1,1]
  4. 反向传播
节点选择：PUCT（c_puct 需与叶值尺度匹配，默认 1.0 配合归一化）
候选动作：candidate_actions_k(k=3)，每动作类型 top-K（Build/Network 各 3，其余各 1）
叶评估：LeafEval::OnePly（默认，位置快照）｜LeafEval::TwoPly（根玩家跑 2-ply）
```

## 踩坑记录（重要）

### Bug 1：叶值未归一化 → PUCT 探索被吞没（决定性问题）
- 现象：根节点 2000 次模拟全部 visits 只落在 1 个子节点，MCTS 等于"只评估了 1 个动作"，
  对局 avg VP 只有 ~26（比 2-ply 的 ~42 低一大截）。
- 根因：`evaluate_position` 输出 ~10-70 的巨大值，首个被探索子节点拿到 q≈44，
  PUCT 的探索项（prior×c×√(lnN/n)≈0.5）被 q 完全压制 → 之后每步都重选同一节点。
- 修复：叶值除以尺度（OnePly /60，TwoPly /12）归一化到 [-1,1]。
- 修复后：树正常扩展（2000 sims 约 100+ 节点），avg VP 从 26 → 43。

### Bug 2：PUCT 平手选择最后候选
- 首访时所有子节点 uct=0，`max_by` 平手返回最后一个候选（Pass），导致 Pass 被过度探索。
- 修复：探索项用 `√(ln(N+1)/(1+n))` 避免首访为 0，并在平手时按 prior 打破。

### 教训
- **MCTS 的 c_puct 必须与叶值尺度匹配**。位置评估值天然大（10-70），不归一化会让
  探索彻底失效，树退化为贪心。
- **depth 太浅（4）时树扩展不足**（nodes≈28），无法区分相似动作；depth 6 更稳。
- 对手节点用 MaxN（各玩家最大化自己）而非对抗 min —— Brass 是非零和，对手各自为政。

## 命令

```
# 性能验证（单决策）
cargo run --release --bin bench_mcts -- 7 4 5000 10000

# 强度对比（MCTS 座位旋转 vs 2-ply，最后参数是 MCTS sims）
cargo run --release --bin brass-engine -- 40 4 mcts-vs-2ply 600

# MCTS 全对局 / 对 1-ply
cargo run --release --bin brass-engine -- 8 4 mcts 500
cargo run --release --bin brass-engine -- 30 4 mcts-vs-heur 400
```

## 已知限制 / 下一步

- OnePly 叶评估是位置快照，对"建厂 vs 建网"这类长期价值仍会偏向建厂；TwoPly 叶评估
  能纠正但慢一倍（当前仅作用于根玩家）
- 树在每次决策时重建（未复用历史树）——根并行 / 树复用是下一步提速点
- `k_candidates`、`c_puct`、`prior_temp`、`max_depth` 未做系统扫描，单状态调参噪声大，
  应以整局胜率为准
- 下一步可接 Rayon 做根并行（每线程独立树，聚合决策），把 10000 sims 压进 2s 预算
