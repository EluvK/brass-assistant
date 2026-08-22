# 卡片选择与动作表示临时规划

> 状态：临时规划，待后续实现前评审确认。
>
> 目标：修正 Rust 启发式 teacher 的出牌偏差，并使 Python 网络能够理解“打出的具体卡片”，同时保留完整的理论合法动作覆盖。

## 1. 当前结论

规则层 `legal_moves()` / `legal_candidates()` 会枚举完整的 concrete action，包括不同卡片、资源来源、卖出组合和商人啤酒路线。

但启发式 AI 与训练表示存在明显压缩和语义断裂：

- `Network`、`Develop`、`Sell`、`Loan`、`Pass` 等动作使用 `pick_any_card()` 固定取手牌第一个索引。
- `Build` 只优先选择第一张非万能合法牌，不比较不同卡片的长期机会成本。
- `Scout` 使用启发式固定的一组三张牌，而不是对所有组合评分。
- `heuristic_candidates()` 只保留有限 teacher shortlist，不是全量合法动作。
- 具体候选动作必须保留卡片和资源来源等执行差异。
- action feature v1 使用 `card_index` 的 8 维 one-hot，而不是卡片语义。
- 状态侧 `own_hand` 是无序的全量卡片编码，因此网络无法从 `card_index` 推断该位置对应哪张牌。

## 2. 主要风险

### 2.1 Teacher 偏差

固定打手牌第 0 张会让 bootstrap 数据产生强烈的位置偏差。网络可能学习到数组位置，而不是卡片价值。更重要的是，teacher 分数没有计入被消耗卡片的机会成本。

### 2.2 动作特征不可泛化

动作只编码“打第几个手牌位置”，状态却只编码“手里有哪些牌”。同一个 action feature 在不同局面中可能对应完全不同的卡片，网络无法稳定学习其含义。

### 2.3 训练分布不完整

MCTS 的 concrete candidate 路径可以看到全量合法动作，但 imitation replay 只看到 teacher shortlist。大量合法卡片组合不会进入 bootstrap 数据，teacher shortlist 的 softmax 也不是完整动作分布。

### 2.4 组合空间与等价动作

如果简单枚举所有 hand index，重复卡片会产生只在索引上不同的冗余动作；如果过早折叠，又可能丢失不同卡片语义的长期影响。需要明确“执行引用”和“网络语义”之间的分层。

## 3. 设计原则

1. `card_index` 保留为 Rust 执行层引用，但不能作为网络理解卡片的主要语义特征。
2. 合法性和动作生成继续由 Rust 唯一负责，Python 不重新实现规则。
3. 对网络可见的动作，必须能表达被消耗卡片的稳定语义：地点牌、产业牌、万能地点牌、万能产业牌。
4. 完全等价的重复牌可以去重，但不同卡片语义不能因为 slot 压缩而丢失。
5. teacher 的动作评分必须包含卡片机会成本和保留万能牌/关键牌的价值。
6. 完整合法动作路径与 teacher shortlist 路径必须使用同一套 concrete action 语义。

## 4. 分阶段修改方案

### 阶段 A：动作与数据审计

- 统计各类局面中“任意卡可打”的动作数量、卡片语义数量和重复数量。
- 对 concrete canonical action 与 action feature 做碰撞分析。
- 确认手牌顺序是否仅为抽牌顺序，不能被当作稳定语义。
- 增加回归测试，覆盖万能牌、重复牌、仅一张合法牌和多张不同语义合法牌。

### 阶段 B：卡片语义表示 v2

- 保留实际 `card_index` 用于执行和 replay 对齐。
- 新增稳定卡片语义特征：Location、Industry、WildLocation、WildIndustry。
- 升级 action feature schema version，并同步 Rust、Python adapter、replay 和 checkpoint 校验。
- 明确重复同语义卡片的去重规则，避免仅 index 不同的无意义候选爆炸。

### 阶段 C：Card-aware heuristic teacher

- 对每个结构动作枚举所有语义不同的合法出牌选择。
- 新增卡片机会成本评分：万能牌、关键地点牌、关键产业牌应有保留价值。
- `Pass`、`Loan`、`Sell` 优先消耗低价值卡，而不是固定索引。
- `Network` 和 `Build` 同时考虑当前收益与未来手牌灵活性。
- `Scout` 对候选三牌组合进行评分，并保留必要的合法组合覆盖。

### 阶段 D：Teacher shortlist 与训练数据

- 先生成完整 concrete legal actions，再进行评分和 shortlist。
- shortlist 保留高分动作，同时从完整合法集合加入少量结构不同的 hard negatives。
- 对候选按动作结构和卡片语义去重，不把不同语义动作错误合并。
- 保证 teacher 选中动作始终存在于候选集合中。
- 明确 imitation target 是局部 teacher 排序，不把它解释为完整合法动作概率。

### 阶段 E：网络与评测

- 优先沿用 concrete candidate scorer。
- 比较旧 v1 表示、新 v2 表示、shortlist-only 与 shortlist+hard-negatives。
- 建立按 action type、卡片语义、候选数量分桶的 held-out teacher validation。
- 检查网络是否仍偏好手牌位置，而不是卡片语义。

## 5. 暂不做的事情

- 不在 Python 中复制 Brass 规则或合法动作生成。
- 不直接把所有 full legal actions 无限制写入长期 replay；先评估内存和训练吞吐。
- 不在没有 schema、canonicalization 和测试方案前修改网络输入维度。
- 不把动作特征压缩成无法表达完整卡片选择的固定动作空间。

## 6. 完成标准

- 同一结构动作的不同卡片语义能够产生不同且稳定的 action feature。
- 网络不依赖手牌数组位置来识别被打出的卡片。
- 启发式在多个合法出牌之间会根据卡片机会成本做选择。
- 完整 concrete candidate 路径仍覆盖所有理论合法动作。
- teacher shortlist 不再系统性固定使用手牌第 0 张。
- replay、checkpoint 和 schema version 能拒绝旧格式与新格式混用。
