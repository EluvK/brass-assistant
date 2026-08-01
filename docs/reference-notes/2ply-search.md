# 2-Ply 确定性前瞻（search_ai）— 里程碑记录

## 动机

1-ply 启发式（`heuristic_ai`）每步只看一步，无法看到《伯明翰》的核心机制——
**同回合两动天然构成 combo**（Build→Sell、Network→Build、Develop→Build）。
在 1-ply 视角下，"先铺路"这一步价值为 0（纯花钱），导致 AI 行为"看起来瞎玩"。

## 实现

`src/engine/src/search_ai.rs`：对 1-ply 候选动作集（`candidate_actions`，每种行动的最佳
一步）逐个模拟，找到同玩家回合内**最佳第二动作**，第一动作的价值为：

```
value(first) = score(first) + ALPHA * score(best second action after first)
```

- `ALPHA = 0.6`：锚定 1-ply 自身评分，保证决策与已校准的启发式一致，同时给能解锁强力后续
  的动作加 bonus。
- 纯位置快照评估（leaf eval）被否决：会过度奖励贷款（看不见长期收入损失）导致行为退化。

## 实测结果（4 人局，自对弈）

| 指标 | 1-ply | 2-ply | 提升 |
| --- | --- | --- | --- |
| Avg VP/人（800局） | ~31.5 | **~37.7** | +20% |
| 建厂/局 | 11.9 | **14.1** | +18% |
| 翻面/局 | 7.6 | **9.4** | +24% |
| 连接/局 | 11.7 | 12.5 | +7% |
| 胜率分布 | 21-28% | 23-26% | 更均衡 |

行为变化（seed=7 replay 对比）：
- 1-ply：首轮全员研发、大量过牌、贷款频繁、行动浪费
- 2-ply：`建网→建厂`、`建厂→卖货` 等 combo 直接出现，建厂更早更狠，翻面时机好

## 命令

```
cargo run --release --bin brass-engine -- 800 4 heuristic   # 1-ply 基线
cargo run --release --bin brass-engine -- 800 4 2ply        # 2-ply
cargo run --release --bin replay -- 7 4 2ply                # seed=7 2-ply 详细日志
```

## 已知限制 / 下一步

- `ALPHA` 为手工调参，可后续 grid search
- 只搜"最佳第二动作"，未穷举所有 combo（分支因子大，先保速度）
- 对手信息未建模（确定性搜索，非 ISMCTS）——这是阶段 ② 性能工程后的下一步
