# Candidate Policy: Remaining Work

当前实现和操作契约见 [ai-tools.md](ai-tools.md)。本文只保留 candidate policy
尚未解决的设计问题，避免重复记录已完成的迁移过程。

## 训练与搜索的候选分布差异

imitation replay 使用有界 teacher shortlist，以控制内存；MCTS 推理则评分完整
legal concrete action 集合。两者的差异是当前 policy 泛化的主要风险。

下一步应在不恢复全量 replay 的前提下加入 hard negatives：

- 保留 teacher 高分动作；
- 从 full legal actions 采样结构不同的低分动作；
- 覆盖其他 action type、地点、产业和资源来源；
- 将每个训练样本控制在约 16--32 candidates；
- target 只在 teacher-ranked actions 上有主要概率，negative 保持零或极低权重。

必须比较“仅 shortlist”和“shortlist + hard negatives”在 held-out full-candidate
MCTS benchmark 中的差异，而不只比较训练 top-k。

## Action Feature V2

v1 feature schema 是可运行基线，但会压缩部分 concrete action 信息。升级前必须
先做统计与错误案例分析：

- 不同 canonical actions 是否产生相同 feature；
- 这些 collision 是否集中在煤/铁来源、价格、free 标记或 Sell；
- collision 是否对应网络的高分错误或 MCTS 的错误扩展。

v2 候选补充方向：

- 资源来源的 tile/market identity、价格、free 标记；
- Sell 的 tile--merchant--beer source 对应关系；
- Scout 的牌组合与 card semantics；
- 双铁路中连接和资源来源的配对关系。

任何 schema 修改都必须升级 version，并使 replay/checkpoint 明确拒绝不兼容数据。

## Policy Evaluation

训练 replay 上的 top-1/top-3/top-5 只能衡量拟合，不能表示策略强度。需要固定：

- held-out teacher seeds；
- held-out candidate ranking metrics；
- policy-only vs heuristic；
- 不同 MCTS simulation budget 下的 policy-guided MCTS vs heuristic；
- 按 era、action type 和 full legal candidate count 分桶的错误分析。

四人局 benchmark 至少需要 100 局后再解释 win rate。20 局只适合发现崩溃或严重
退化。

## Candidate Scorer Evolution

当前 scorer 是 state embedding 和 action embedding 的 MLP 融合。后续只有在
error analysis 指出 pairwise interaction 不足时，才考虑：

- state token 对 action token 的 cross-attention；
- 图编码后的 location/connection embedding；
- action type conditional expert；
- candidate-set attention，用于比较互相竞争的候选动作。

不要为了“层级化”而恢复固定动作表或让 Python 生成合法动作组合；合法性始终由
Rust Engine 保证。
