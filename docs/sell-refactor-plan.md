# Sell 动作重构实施计划

## 1. 目标与最终语义

本次重构不兼容旧模型、旧 canonical 或旧训练数据。目标是让 Sell 动作只表达真正会影响执行结果和后续状态的选择，并在生成阶段尽早完成合法性校验。

Sell 的语义定义为：

```text
翻面哪些待售板块
+ 消耗哪些具体酒桶
+ Gloucester 免费开发哪个行业（如果有）
+ 使用哪张牌
```

以下内容不再作为动作身份：

- 未使用商家酒时选择了哪个普通买家；
- 酒桶在不同板块之间的排列顺序；
- 同一组酒桶的逐板块分配顺序，只要最终执行结果相同。

商家酒仍然是有语义的选择，因为它会消耗商家酒并触发该商家奖励。商家酒的 `merchant_idx` 直接包含在 `BeerSource` 中。

## 2. 需要修改的核心数据结构

### 2.1 `ResolvedMove::Sell`

文件：`engine/src/model/move.rs`

将当前结构：

```rust
Sell {
    keys: Vec<usize>,
    merchant_indices: Vec<usize>,
    use_merchant_beer: Vec<bool>,
    beer_sources: Vec<Vec<BeerSource>>,
    free_develop: Option<IndustryType>,
    card_index: usize,
}
```

改为：

```rust
Sell {
    keys: Vec<usize>,
    beer_sources: Vec<BeerSource>,
    free_develop: Option<IndustryType>,
    card_index: usize,
}
```

`keys` 和 `beer_sources` 均必须按照稳定顺序保存：

- `keys` 按城市槽位 key 升序；
- `beer_sources` 按 `BeerSource` 的规范排序键升序。

排序只用于 canonical 和去重，不代表执行顺序。

### 2.2 `Move::Sell`

文件：`engine/src/model/move.rs`

同步删除：

- `merchant_indices`；
- `use_merchant_beer`；
- 对齐的二维 `beer_sources`。

`Move::Sell` 仍然保留 `card_candidates`，因为牌选择仍由独立的 card head 处理。

### 2.3 辅助类型

文件：`engine/src/gameplay/actions/sell.rs`

保留 `BeerSource` 作为实际酒桶身份。建议新增：

```rust
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SellIdentity {
    pub keys: Vec<usize>,
    pub beer_sources: Vec<BeerSource>,
    pub free_develop: Option<IndustryType>,
}
```

实现统一的规范化函数：

```rust
pub fn sell_identity(
    keys: &[usize],
    beer_sources: &[BeerSource],
    free_develop: Option<IndustryType>,
) -> SellIdentity
```

该函数负责复制、排序并构造唯一标识。所有 legal move、MCTS、heuristic、canonical 层统一调用，禁止各处自行拼接字符串或自行排序。

## 3. 酒桶规范排序与等价性

`BeerSource` 当前包含：

```rust
kind: BeerSourceKind,
key: usize,
farm_idx: Option<usize>,
merchant_idx: Option<usize>,
```

新增一个只用于排序的稳定键，例如：

```text
Own/Opponent city source: (kind, 0, key, farm_idx)
Own/Opponent farm source:  (kind, 1, farm_idx, key)
Merchant source:           (Merchant, 2, merchant_idx)
```

不要只按 `key` 排序，因为农场和商家使用 `usize::MAX`，且不同来源可能共享 key。

两个 Sell 计划只有在以下三项都相同才视为等价：

```text
排序后的 keys
排序后的 BeerSource 集合
free_develop
```

必须保留的差异：

- 自家酒厂桶与对手酒厂桶不同；
- 不同城市槽位或不同农场桶不同；
- 不同商家桶不同；
- `free_develop` 行业不同。

可以折叠的差异：

- 普通买家不同但没有使用商家酒；
- 同一酒桶集合的排列顺序；
- 相同最终资源消耗下的逐板块 assignment 顺序。

## 4. Sell 合法性校验 API

文件：`engine/src/gameplay/actions/sell.rs`

将当前分散在 `get_valid_sell_targets()`、`plan_sell_beer_sources()` 和 `execute_sell_inner()` 中的规则整理为两层 API。

### 4.1 单板块选项

新增内部类型：

```rust
struct SellChoice {
    key: usize,
    buyer: usize,
    beer_sources: Vec<BeerSource>,
}
```

单板块选项生成必须完成：

1. 板块存在、属于当前玩家、未翻面、行业可出售；
2. 买家接受该行业且与板块地点连通；
3. 如果使用商家酒，商家必须是该买家、仍有酒，并且板块确实需要酒；
4. 其余酒桶来自可访问的自家/对手酒厂；
5. 酒桶数量恰好满足该板块需求。

普通买家路线只作为校验和执行辅助，不进入 `SellIdentity`。

### 4.2 完整计划校验

新增统一入口：

```rust
pub fn validate_sell_plan(
    state: &GameState,
    pid: usize,
    keys: &[usize],
    beer_sources: &[BeerSource],
    free_develop: Option<IndustryType>,
) -> Result<SellExecutionPlan, String>
```

`SellExecutionPlan` 是执行阶段的内部结构，可以包含逐板块 buyer 和逐板块酒桶分配，但不作为动作身份：

```rust
struct SellExecutionPlan {
    assignments: Vec<SellAssignment>,
    free_develop: Option<IndustryType>,
}
```

校验必须使用 clone 后的模拟状态，并按确定性规则：

- 验证每个 key 不重复；
- 验证所有 key 的板块合法；
- 验证输入的每个 `BeerSource` 当前存在且可用；
- 验证每个板块得到恰好所需酒桶；
- 验证商家酒来源的商家同时是该板块合法买家；
- 没有商家酒时，为板块选第一个稳定排序的合法买家；
- Gloucester 商家酒最多出现一次；
- `free_develop` 与 Gloucester 奖励严格匹配；
- `free_develop` 必须是当前可开发行业。

生成器和最终执行器都调用该 API。执行器不得重新实现一套不一致的合法性逻辑。

## 5. Sell 计划枚举算法

文件：`engine/src/gameplay/legal_moves.rs`

删除当前“目标子集 × 商家 route，叶子节点再调用 `plan_sell_beer_sources()`”的实现。

改为回溯枚举：

```text
按 key 升序处理待售板块
├─ 跳过当前板块
└─ 选择当前板块的一个合法酒桶组合
   ├─ 酒厂/农场酒桶组合
   └─ 某个合法商家的商家酒 + 其余酒厂酒桶
      → 立即更新搜索状态
      → 继续下一个板块
```

搜索状态至少包括：

```rust
struct SellSearchState {
    simulated: GameState,
    keys: Vec<usize>,
    beer_sources: Vec<BeerSource>,
    free_develop_eligible: bool,
}
```

实现要求：

1. 每次加入板块后立即消耗模拟状态中的酒桶；
2. 剩余资源不足时立即剪枝；
3. 商家酒已消耗时立即剪枝；
4. 一个计划最多允许一个 Gloucester 免费开发奖励；
5. 不枚举 source 的排列，只枚举 source 的组合；
6. 每个完整非空计划调用 `sell_identity()` 并插入 `HashSet`；
7. HashSet 只保留每个 Identity 的一个可执行代表；
8. 最终调用 `validate_sell_plan()` 生成执行所需 assignment。

### 5.1 酒桶组合生成

为每个板块生成酒桶方案时，按以下顺序尝试：

1. 只用酒厂/农场酒；
2. 每个可接受商家分别尝试使用一桶商家酒；
3. 其余需求从酒厂/农场酒桶中补足。

每个来源组合只出现一次。不能继续使用当前 `find_beer_sources().take(n)` 作为唯一方案，因为它会丢失不同酒厂来源产生的不同后状态。

### 5.2 记忆化

如果基准测试显示回溯仍占用明显时间，在递归层加入 memoization。memo key 应包含：

```text
当前 target index
已消费的酒桶集合
已使用的 Gloucester 标志
已选择的板块集合
```

不要只按“剩余酒桶数量”记忆，因为不同酒厂/对手酒桶会造成不同后续状态。

## 6. 执行逻辑

文件：`engine/src/gameplay/actions/sell.rs`

新增或重写 `execute_sell_with_free_develop()`，入口先调用 `validate_sell_plan()`，得到 `SellExecutionPlan` 后再执行。

执行顺序固定为：

1. 校验牌索引；
2. 校验并解析完整 Sell 计划；
3. 按 assignment 消耗商家酒和酒厂酒；
4. 应用商家奖励；
5. 翻面所有板块并推进收入；
6. 应用免费开发；
7. 弃牌。

执行失败时恢复 snapshot。成功后不得再次根据 merchant route 或 `use_merchant_beer` 推断酒桶来源。

`execute_sell()` 的兼容辅助语义可以删除；如果仍保留，必须改为调用新的完整计划入口，不再制造“第一个可用商家”的隐含旧格式。

## 7. Move / legal move / canonical 修改

### 7.1 `Move::resolve`

文件：`engine/src/model/move.rs`

Sell resolve 只复制：

```rust
keys
beer_sources
free_develop
card_index
```

### 7.2 `legal_resolved_moves`

文件：`engine/src/gameplay/legal_moves.rs`

每个 canonical Sell Identity 只生成一个结构动作，再为该结构动作附加所有合法 card candidates。不能让 card index 或 route 排列扩大结构动作数。

### 7.3 `legal_moves` 的 structural key

`structural_from_resolved()` 的 Sell key 改为：

```text
sell:{sorted_keys}:{sorted_beer_sources}:{free_develop}
```

### 7.4 `operation_key`

文件：`engine/src/ai/heuristic_ai/mod.rs`

使用 `sell_identity()` 的结构化结果。不要继续把普通 merchant buyer 放入 operation key。

### 7.5 MCTS `MoveKey`

文件：`engine/src/ai/mcts_ai.rs`

将 Sell 的 MoveKey 改为：

```rust
Sell {
    keys: Vec<usize>,
    beer_sources: Vec<BeerSource>,
    free_develop: Option<IndustryType>,
}
```

不保留逐板块 merchant、beer flag 或 assignment 顺序。

## 8. canonical 编解码

文件：`engine/src/bridge/move_codec.rs`

新的 Sell canonical 建议为：

```text
Sell{keys:... ,sources:...,free:...,card:...}
```

删除字段：

- `merchants`；
- `beer`。

`sources` 为扁平、排序后的 `BeerSource` 列表。decoder 只解析动作身份；执行时由当前 state 调用 `validate_sell_plan()` 恢复逐板块 assignment。

这样 canonical 表达的是最终资源选择，而不是生成器内部的临时路线。

## 9. 动作特征重构

文件：`engine/src/bridge/action_features.rs`

可以完全重构 301 维，不保留旧布局。

建议布局：

```text
ACTION              7
CARD               35
LOCATION           27
CITY_SLOT           4
INDUSTRY_1          6
INDUSTRY_2          6
CONNECTION_1       39
CONNECTION_2       39
SELL_KEY           47
SELL_BEER_BOARD    49
SELL_MERCHANT_BEER  9
CONSEQUENCE        12
SUMMARY            12
```

Sell 专用部分：

- `SELL_KEY[key] = 1`：该板块会被翻面；
- `SELL_BEER_BOARD[key]`：从对应酒厂/农场格消耗的酒桶数，按 `/4` 编码；
- `SELL_MERCHANT_BEER[idx]`：从商家 idx 消耗的酒桶数，按 `/4` 编码；
- `INDUSTRY_1`：免费开发行业；
- `CONSEQUENCE`：翻面数量、自家/对手酒厂翻面等后果；
- `SUMMARY`：总酒桶数、商家酒数量、待售板块数量等汇总。

删除 Sell 的普通 `MERCHANT` 特征。普通买家是合法性约束，不是动作结果；商家酒来源已经由 `SELL_MERCHANT_BEER` 表达。

其他动作如果仍需要原 `DRAIN`，可以保留该物理区块但按动作类型解释；若追求语义清晰，建议将其拆为动作专用 source block，并同步调整总维度。

特征编码必须直接从 `ResolvedMove::Sell` 的扁平 `beer_sources` 统计，不再从逐板块 assignment 推导。

修改后提升：

```rust
pub const ACTION_FEATURE_SCHEMA_VERSION: usize = 5;
```

同时更新 `docs/ai-action-encoding.md` 的布局表、Sell 章节、碰撞说明和 schema 维护清单。

## 10. 启发式 AI 修改

文件：`engine/src/ai/heuristic_ai/sell.rs`

当前启发式逻辑依赖 `SellTarget.routes`、`merchant_indices` 和 `use_merchant_beer`，需要改为消费新的合法 Sell 计划。

建议流程：

1. 从完整合法 Sell 计划中读取 `keys` 和 `beer_sources`；
2. 商家奖励根据 `BeerSourceKind::Merchant` 计算；
3. 普通酒厂酒不产生商家奖励；
4. 免费开发奖励根据 `free_develop` 计算；
5. 评分仍可按板块价值、收入、商家奖励和时代紧迫度排序；
6. `SOURCE_VARIANTS` 改为按不同 `SellIdentity` 选择候选，而不是仅替换第一个板块的 merchant route；
7. 每个候选计划只做一次 validation，不再对每个 prefix 重复执行完整模拟。

`SellTarget` 可以删除。如果其他模块需要单板块合法信息，则改名为更明确的 `SellTileOptions`，不要将 route 与最终动作结构混用。

## 11. Python 边界

Python 没有独立的 Sell 规则编码器；Rust 通过 pyo3 输出 canonical 和 action features，Python 网络直接接收特征。

需要修改或确认：

- `python/brass_ai/net.py` 的输入维度配置；
- 特征 schema/version 校验；
- candidate canonical 对齐逻辑；
- 训练数据读取和缓存格式；
- 任何依赖 301 维的断言、模型构造和测试。

Python 不应重新推断买家或酒桶分配，只消费 Rust 已验证的候选动作。

## 12. 测试与验收标准

### 12.1 规则生成测试

增加或改写 `engine/tests/engine_tests.rs`：

1. 单板块无酒需求时，不因不同普通买家生成重复动作；
2. 单板块需要酒时，能够生成不同酒厂来源的合法计划；
3. 自家酒厂桶、对手酒厂桶、农场桶分别能被正确区分；
4. 不同商家酒桶会生成不同动作；
5. 商家酒只能与对应合法买家配对；
6. 多板块共享酒桶时，不生成资源超额的计划；
7. 同一酒桶集合的不同排列只生成一个动作；
8. 相同 keys/source、不同 `free_develop` 保留为不同动作；
9. 所有 `legal_resolved_moves()` 生成的 Sell 都能 `apply_move()`；
10. 所有生成的 Sell 都能 canonical round-trip。

### 12.2 编码测试

增加：

1. 不同酒厂来源的 Sell 特征不同；
2. 不同商家酒来源的 Sell 特征不同；
3. 普通 buyer 不同但最终 keys/source 相同的动作特征相同；
4. 不同 `free_develop` 的特征不同；
5. `SELL_KEY`、酒厂酒、商家酒和免费开发字段与 canonical 一致；
6. 所有特征值满足 0.25 步长和 uint8 压缩范围；
7. 新 schema 维度与 Python 网络输入维度一致。

### 12.3 性能基准

在至少 20 个固定 seed 和若干高密度 Sell 状态下记录：

- legal Sell 计划数量；
- 去重前后计划数量；
- `legal_resolved_moves()` 耗时；
- heuristic Sell 评分耗时；
- MCTS 根节点候选数；
- 每个状态的 clone 次数。

验收目标：

- 不再因为普通 buyer 产生等价 Sell 分支；
- 不再重复对每个 prefix 执行完整 source planning；
- 所有真正不同的酒桶消耗方案仍被保留；
- 高密度状态下生成时间不劣于当前实现；若 source 选择增加导致分支变多，必须通过组合去重、剪枝或 memoization 抵消。

## 13. 实施顺序

按以下顺序修改，避免中间状态跨模块不一致：

1. 在 `sell.rs` 增加 `SellIdentity`、规范排序和统一 validation；
2. 重写 Sell 酒桶组合生成与增量回溯；
3. 修改 `Move`/`ResolvedMove` 数据结构；
4. 修改 `legal_moves.rs` 和 Sell 执行路径；
5. 修改 `move_codec.rs`、replay formatter 和 MCTS `MoveKey`；
6. 修改 heuristic Sell scorer；
7. 重构 `action_features.rs` 并提升 schema version；
8. 同步 Python 输入维度、候选对齐和训练数据处理；
9. 更新 `docs/ai-action-encoding.md`；
10. 运行 Rust 单测、Python 测试、canonical round-trip 测试和性能基准。

## 14. 最终效果

重构前：

```text
待售板块子集 × 商家路线 × 确定性酒桶计划
```

重构后：

```text
合法待售板块集合 × 合法实际酒桶集合 × 免费开发选择
```

最终收益：

- 普通商家选择不再制造等价动作；
- 自家、对手、农场和商家酒桶的战略差异不会丢失；
- 合法性在枚举过程中提前检查并剪枝；
- canonical、MCTS key、动作特征和实际后状态使用同一套语义；
- 网络不需要学习无意义的 merchant assignment；
- 训练候选更少但信息密度更高；
- 新模型可以直接围绕重构后的动作空间重新训练。
