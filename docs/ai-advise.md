# Birmingham AI 长期设计原则

本文记录约束当前实现与后续演进的设计原则：哪些边界不能动、哪些简化是刻意的、
下一步该由什么证据来驱动。具体的特征与价值契约见
[ai-action-encoding.md](ai-action-encoding.md)，运行命令见 [ai-tools.md](ai-tools.md)，
分阶段规划见 [roadmap.md](roadmap.md)。

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

观测是 102 个 token（棋盘格、连接、商家、座位、全局），动作是"类型 + 实体引用"，
网络按引用 id 取出被引用的实体（见 [ai-action-encoding.md](ai-action-encoding.md)）。
这套表示已经解决了两类结构性问题：

- **逐格细节不再被池化掉**：每个棋盘格是独立 token，槽位身份、资源数、归属与
  网络连通性都在 token 上，动作引用直接指向它们。
- **动作与状态的交互是结构保证的**：打分头看到的是"这一手实际引用的那些格子的
  当前状态"，不依赖人工特征复述后果。

仍然偏粗的是历史与信念：对手手牌目前只有一次 determinization 采样加公开已用牌
计数，没有跨回合的历史序列。下一步的升级应由错误案例驱动——只有当固定观测下的
决策误差明确来自隐藏信息或牌序记忆时，再引入历史 token 与 belief 表示。

## Value And Objective

policy 的最终优化目标是最大化最终第一名概率。价值尺度是 VP 效用：
`(vp - 桌均 VP) / VP_SCALE`，四人和恒为 0，跨局可比且保留分差；搜索的叶子与终局
backup 使用同一尺度，见 [ai-action-encoding.md](ai-action-encoding.md) §4。
winner 头作为辅助监督保留，`q` 头给出动作条件化的价值，用于在树内分辨兄弟招。

不用名次当唯一尺度：名次是 VP 的粗化，会把分差抹平。保留 income、era score 等
经济辅助头（按时代拆分的 econ 头）以改善表征学习，但主目标是终局竞争结果。

如果引入 reward shaping，应使用 potential difference：

```text
r_t = terminal_outcome + lambda * (Phi(s_{t+1}) - Phi(s_t))
```

其中 `lambda` 应随训练下降，避免人工经济指标永久改变“争取第一名”的目标。

## Imperfect Information

当前 search 通过 determinization 处理隐藏手牌，这是第一版近似；观测上标记了
哪些手牌是采样值（`SEAT_HAND_SAMPLED_FLAG`），避免网络把它当成真实信息。长期需要让
policy/value 利用公开历史形成 belief，而不是把未知手牌当作独立随机噪声：

- 已出现与未出现的卡牌；
- 对手动作对其手牌的约束；
- 抽牌与剩余牌堆；
- 对手风格与策略分布。

可从历史 action token + GRU/Transformer 开始。只有当固定 observation 下的
决策误差明确来自 hidden information 时，才投入这一层复杂度。
