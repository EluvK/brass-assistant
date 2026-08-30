# Birmingham AI 长期设计原则

本文只记录尚未落地、但会影响后续路线的设计原则。当前已实现的 candidate
policy、teacher imitation、MCTS bridge、replay 格式和运行命令见
[ai-tools.md](ai-tools.md)；分阶段的发展规划见 [roadmap.md](roadmap.md)。

## 问题本质

《工业革命：伯明翰》是一个不完全信息、随机、多智能体、长时序且动作空间具
组合性的 stochastic game。模型的最终目标不是预测当前局面有多“富”，而是为
当前玩家最大化最终第一名概率。

因此长期架构应保持以下边界：

```text
Engine: legal actions and state transitions
Policy: rank legal candidates
Value: estimate future competitive outcome
Search: allocate lookahead among candidates
```

## State Representation

当前 state encoder 已是"图 + 向量"混合结构，不应被误认为最终表示：

- **已是图结构**：地图连通性——board 格 scatter-pool 成地点节点、连接作为边、
  经农场节点消息传递（`net.py encode_state`，3 层边/节点交替更新；拓扑由
  Rust `encode.rs` 导出）。
- **仍是 flatten vector**：玩家全局量（现金、收入、VP、债务、公开行动）、
  手牌（地点/行业/万能牌的 35 维槽位编码）、历史（已出现卡牌、公开行动、
  时代/回合顺序）。

下一代表示应把剩余对象逐步迁移为 token 或结构化表示（如手牌 token、历史
action token）。优先级不是立刻引入 Transformer 或 GNN，而是先通过错误案例
证明当前 encoder 无法区分的关键局面，再增量升级。目标是让模型理解卡牌保留
价值和对手竞争，而不是只拟合局面统计量。

## Value And Objective

policy 的最终优化目标是最大化最终第一名概率。当前 value 目标已切到竞争结果：
rank 头（每座位终局名次 /n，MSE）与 winner 头（唯一冠军 one-hot，CE），搜索
与终局 backup 统一使用 `1 - rank` 尺度，见
[ai-action-encoding.md](ai-action-encoding.md) §5.2。

保留 income、era score 等辅助头（当前为按时代拆分的 econ 头）以改善表征学习；
但 policy 的最终优化目标保持 win/rank，而不是绝对现金、收入或 VP。

如果引入 reward shaping，应使用 potential difference：

```text
r_t = terminal_outcome + lambda * (Phi(s_{t+1}) - Phi(s_t))
```

其中 `lambda` 应随训练下降，避免人工经济指标永久改变“争取第一名”的目标。

## Imperfect Information

当前 search 通过 determinization 处理隐藏手牌，这是第一版近似。长期需要让
policy/value 利用公开历史形成 belief，而不是把未知手牌当作独立随机噪声：

- 已出现与未出现的卡牌；
- 对手动作对其手牌的约束；
- 抽牌与剩余牌堆；
- 对手风格与策略分布。

可从历史 action token + GRU/Transformer 开始。只有当固定 observation 下的
决策误差明确来自 hidden information 时，才投入这一层复杂度。
