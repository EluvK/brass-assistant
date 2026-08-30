# Birmingham AI 长期设计原则

本文只记录尚未落地、但会影响后续路线的设计原则。当前已实现的 candidate
policy、teacher imitation、MCTS bridge、replay 格式和运行命令见
[ai-tools.md](ai-tools.md)。

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

当前 state encoder 是可运行基线，不应被误认为最终表示。下一代表示应逐步把
以下对象从 flatten vector 迁移为 token 或图结构：

- 玩家：现金、收入、VP、债务、产业板、网络、公开行动；
- 产业 tile：owner、location、industry、level、flip 状态、收益、时代属性；
- 地图：城市节点、连接边、商家、资源市场和网络归属；
- 手牌：地点牌、产业牌和万能牌 token；
- 历史：已出现卡牌、公开行动和时代/回合顺序。

优先级不是立刻引入 Transformer 或 GNN，而是先通过错误案例证明当前 encoder
无法区分的关键局面，再增量升级。目标是让模型理解连通性、卡牌保留价值和
对手竞争，而不是只拟合局面统计量。

## Value And Objective

policy 的最终优化目标是最大化最终第一名概率。v4 已把 value 目标切到竞争结果：
rank 头（每座位终局名次 /n，MSE）与 winner 头（并列冠军的均匀分布，CE），搜索
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

## Training Roadmap

1. 建立 held-out teacher validation 和 full-candidate benchmark。candidate
   recall 指标（已训练策略在全合法集上的 top-1/top-3 是否落在 shortlist 内）已定义于
   [ai-action-encoding.md](ai-action-encoding.md) §7；首个 v4 checkpoint
   （`checkpoints/bootstrap-0830.pt`）已产出，待训练充分后测量。
2. 为 imitation shortlist 加入有限 hard negatives，缩小训练与 MCTS 推理的候选
   分布差异。两条路径的候选集本身一致（同一 `candidate_actions_k(4)`），
   真正的偏差来自候选生成器：被 shortlist 排除的备选可能优于 teacher 选择，
   生成器上限需以 candidate recall 等更强参照度量（同文档 §7）。
3. 校准 policy/value，确认 candidate encoder 能泛化后，完成最小 self-play run。
   v4 编码/网络头改造已落地（2026-08-29），首个 v4 full-legal bootstrap 已开始
   （`checkpoints/bootstrap-0830.pt`）；下一步是训练迭代与 recall 测量，再过渡到
   self-play。
4. 用 MCTS visit distribution 逐步替换 heuristic teacher，并保留历史 checkpoint
   opponent pool。
5. 增加 history/belief 表示（winner/rank value 已于 v4 落地）。
6. 仅在 self-play + search 已稳定后，评估 PPO/actor-critic 是否带来额外价值。

PPO 不是当前阻塞点；candidate representation、验证集和 value 目标更基础。
