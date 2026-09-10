# AI 特征与价值契约

本文是 Rust 引擎与 Python 网络之间唯一的接口契约：观测张量、动作表示、
价值尺度、动作身份。引擎、训练、推理、搜索全部以本文为准；任何一侧与本文
不一致都属于缺陷。改动本文必须同时改动两侧实现与测试（见 §8）。

## 0. 设计原则

1. **引擎是唯一权威。** 合法性、实体 id、动作语义、终局分数全部由 Rust 定义并
   导出。Python 只做张量组装与网络前向，不重复实现规则。
2. **视角规范化。** 所有观测按行动方旋转编码，玩家 0 恒为"我"。网络永远不需要
   从数据里反推自己是几号位。
3. **动作是引用，不是特征。** 候选动作只声明它选了哪些实体；网络打分时按 id
   取出这些实体在**当前状态**下的 token，因此动作与状态的交互是结构保证的，
   不依赖人工特征去复述状态。
4. **价值是 VP 尺度。** 终局效用是"相对本桌平均分的 VP 差"，不是名次。名次是
   派生量，丢掉分差会让搜索无法分辨兄弟招。
5. **对手手牌只是采样。** 对手手牌只在 determinization 下可见，且必须与其他
   玩家区分标记。网络不得把采样值当成真实信息。

## 1. 实体 id 空间

| 实体 | 空间 | id 定义 |
| --- | --- | --- |
| 棋盘格 cell | 49 | 0–46 城市槽（前 20 个地点的槽位依次铺开），47/48 为南北农场 |
| 地点 location | 27 | `ALL_LOCATIONS` 顺序 |
| 连接 connection | 39 | `connections()` 顺序 |
| 商家 merchant | 9 | `state.merchants` 顺序，各自挂在 9 个商家地点上 |
| 行业 industry | 6 | `IndustryType::ALL` |
| 卡牌语义 card | 35 | 0–26 地点牌，27–32 行业牌，33 万能地点，34 万能行业 |
| 座位 seat | 4 | 旋转后 0 = 行动方 |

cell → location、cell → slot 的映射由 Rust 导出（`board_cell_locations()`、
`BOARD_CELL_SLOTS`）。Python 不复制地图常量。

## 2. 观测张量

观测是一组 **token**，不是扁平向量。token 总数 102：

| token 组 | 数量 | 说明 |
| --- | --- | --- |
| cell | 49 | 棋盘格：行业板块、资源、连通性 |
| link | 39 | 连接：建成状态、归属、时代可建性 |
| merchant | 9 | 商家：收货类型、啤酒存量 |
| seat | 4 | 玩家公开状态 + 手牌信息（§2.6） |
| global | 1 | 时代、轮次、市场、行动队列 |

每个 token 的字段拼接后过一层线性映射到 `d_model`，再加上类型 embedding（5 类）
与 §2.4 的身份 embedding。所有计数都除以本文给出的归一化常数，one-hot 为 0/1。

### 2.1 cell token（49）

静态：slot_index/4、is_farm、可建行业 multi-hot(6)。

动态：occupied、owner 相对 one-hot(4)、industry one-hot(6)、flipped、
resource_cubes/6、level/8、vp/20、income/7。

推导（相对行动方）：`in_net[4]`（该格所属地点是否在各自座位玩家的网络中，0 号位
即我；取自 `GameState::network_mask`）、到我的网络的图距离/6（不可达记 1.0）。

### 2.2 link token（39）

静态：canal 可建、rail 可建、via-farm 存在。

动态：built、owner 相对 one-hot(4)、is_canal。

推导：`in_net[4]`（相对座位）、`touch_net_me`（任一端点落在我的网络中）。

### 2.3 merchant token（9）

收货类型 one-hot(5)：Blank / Any / 棉纺厂 / 制造厂 / 陶器；`has_beer`。

### 2.4 token 身份

特征本身不足以识别 token：同一城市里两个能力相同的空槽、两条时代属性相同的连接、
两个收货类型相同的商家，特征行完全相同。网络必须能区分它们，否则引用到哪一个都
一样。因此：

- 每组 token 额外加一个**组内位置 embedding**（cells 49 / links 39 / merchants 9 /
  seats 4），提供精确身份。
- cells 与 links 再加一个**共享的 location embedding**：cell 用自己所属地点
  （`BOARD_CELL_LOCATIONS`），link 用两端地点（`CONNECTION_ENDPOINTS`）的均值。
  这样空间关系是显式的，而不是靠 id 碰巧学出来。

### 2.5 global token（1）

era、round/8、rounds_remaining/10、actions_remaining/actions_per_turn、
煤市场 one-hot(15)、铁市场 one-hot(11)、牌堆剩余/牌堆总量、弃牌堆张数/牌堆总量、
wild_location_pile、wild_industry_pile、行动队列 one-hot(4×4，旋转后)。

### 2.6 seat token（4，index 0 = 我）

公开量：money/200、income_space/99、income_level、vp/200、canal_links/14、
rail_links/14、hand_size/8、has_wild_location、has_wild_industry、
每个行业剩余板块数/该行业总数(6)、money_spent_this_round/100、is_current。

手牌量（三组，各自 35 维 bag-of-card-semantics）：

| 向量 | 0 号位（我） | 对手 |
| --- | --- | --- |
| `hand_real` | 真实手牌 | 全 0 |
| `hand_sampled` | 全 0 | 本次 determinization 采样出的手牌 |
| `hand_public` | 已打出的非万能牌计数 | 同左 |

外加 `SEAT_HAND_SAMPLED_FLAG` 标志位。手牌用 bag 而非序列：手牌顺序是执行产物，
语义相同的牌完全可互换，顺序不携带任何策略信息。

### 2.7 张量形状

Rust 按组导出特征张量，Python 把每组线性投影到 `d_model`，加上类型 embedding 与
§2.4 的身份 embedding，再拼成 102 个 token 的序列：

| 名称 | 形状 | 类型 |
| --- | --- | --- |
| `cells` | `(B, 49, F_cell)` | float32 |
| `links` | `(B, 39, F_link)` | float32 |
| `merchants` | `(B, 9, F_merchant)` | float32 |
| `seats` | `(B, 4, F_seat)` | float32 |
| `global` | `(B, F_global)` | float32 |

各组的特征宽度由 Rust 导出，Python 不硬编码。

### 2.8 归一化常数

| 字段 | 除数 |
| --- | --- |
| 板块资源块 | 6 |
| 板块等级 | 8 |
| 板块 VP | 20 |
| 板块收入 | 7 |
| 槽位序号 | 4 |
| 到我的网络的链接距离 | 6（不可达记 1.0） |
| 现金 | 200 |
| 收入格 | 99；收入等级为 `(level + 10) / 40` |
| 手牌 bag | 8 |
| 已打出的牌 bag | 16 |
| 本回合花费 | 100 |
| 建链数 | 14 |
| 牌堆 / 弃牌堆张数 | 64 |

## 3. 动作表示

一个候选动作 = 动作类型 + 一组实体引用 + 少量标量，由 Rust 编码成一行
`ACTION_FEATURE_DIM = 55` 个 float32，Python 只按偏移量读取：

```
[0]                 action kind（索引，不是 one-hot）
                   0 Build, 1 Network, 2 NetworkDouble, 3 Develop,
                   4 Sell, 5 Loan, 6 Scout, 7 Pass
[1]                 Build 目标槽位，其余为 0
[2..6]              numbers：市场买煤数、市场买铁数、商家啤酒数、支付牌数
[6]                 引用个数 ref_count
[7 + 3i + 0..3]     第 i 个引用：(ref_kind, id, weight)，i < ACTION_REF_CAP = 16
```

偏移量由 Rust 导出（`ACTION_OFF_*`），Python 不硬编码。`ref_kind` ∈
{cell, link, merchant, industry, card}，各类 id 的上界分别是 49 / 39 / 9 / 6 / 35。
`weight` 是该引用的强度（例如每个煤来源记 1.0，同一格被引用两次就累加）。
`ACTION_REF_CAP` 越界时 Rust 直接报错——越界说明引用表设计需要复核，不允许静默截断。

各类动作的引用内容：

| 动作 | 引用 |
| --- | --- |
| Build | 目标 cell、建造行业、每个煤来源的 cell、每个铁来源的 cell、支付牌语义 |
| Network | 目标 link、煤来源 cell（若有）、支付牌语义 |
| NetworkDouble | 两条 link、两个煤来源 cell、一个啤酒来源（cell 或 merchant）、支付牌语义 |
| Develop | 两个行业、每个铁来源的 cell、支付牌语义 |
| Sell | 每个待售 cell、每个啤酒来源（cell 或 merchant）、免费开发行业（若有）、支付牌语义 |
| Loan / Pass | 支付牌语义 |
| Scout | 三个弃牌语义 |

**动作表示不重复陈述状态。** 被引用的格子有几块煤、属于谁、是否翻面，一律由
被取出的 cell token 提供，不写入动作特征。

### 3.1 网络侧如何使用

cell / link / merchant 引用按 id 从当前状态 token 序列取出对应的 token；industry /
card 引用查静态 embedding 表。两者拼成一张实体表后用 `ref_kind` 的偏移量索引，
再按 `weight` 加权求和。

实现上不要把 `(N, REF_CAP, d)` 物化出来：那是一个候选 × 16 × d 的张量，反向传播
还要保留一份。正确做法是把权重累加进 `(N, 实体总数)` 的稀疏权重图，再与实体表做
一次矩阵乘——数学等价，显存差一个数量级。

## 4. 价值与目标

### 4.1 终局效用

对一局终局，令 `vp[p]` 为座位 p 的最终 VP，`mean_vp` 为同桌四人的 VP 均值。

```
utility[p] = (vp[p] - mean_vp) / VP_SCALE         VP_SCALE = 50
```

这是"相对本桌平均分的 VP 差"，四人和恒为 0。它跨局可比、保留分差，并在搜索里
让"未访问孩子的价值取 0"重新变得有意义（零均值尺度）。

Rust 的 `terminal_value` 与 Python 的训练目标必须使用同一公式与同一常数。
VP_SCALE 是唯一的全局尺度常量。

### 4.2 网络头

| 头 | 形状 | 目标 | 用途 |
| --- | --- | --- | --- |
| `value` | `(4,)` | `utility[p]`，MSE | 主价值；搜索叶子与终局 backup |
| `winner` | `(4,)` | 唯一冠军 one-hot，CE | 胜率；辅助 |
| `econ` | `(4,)` | 时代经济（收入等级、现金），MSE | 辅助 |
| `q` | `(N,)` | 行动方实际走出那一手的 `utility[me]`，MSE | 兄弟招排序；搜索用 Q 初始化未访问孩子 |

损失：`policy_CE + value_MSE + 0.5*winner_CE + 0.2*econ_MSE + 0.3*q_MSE + l2`。

名次不单设头：名次是 VP 的粗化，单独训一个名次头既不加信息，又会让它的尺度与
搜索的价值/探索项不可比。名次如需展示，由 VP 预测与破平局规则派生。

### 4.3 策略目标

候选上的分布按 canonical 字符串对齐之后再做 `log_softmax`，归一化只在合法候选上
进行（合法性由引擎构造性保证，网络不学习合法性）。目标分布来自 MCTS visit 或
teacher 分数。

## 5. 动作身份

动作身份用于跨边界对齐与树内复用，必须是**语义身份**，不能是手牌下标。

```
identity = (动作的结构选择：类型 / 目标 / 资源来源, 支付牌的语义多重集)
```

约束：

* Python 与 Rust 之间用 canonical 字符串对齐（`move_codec`）。
* 搜索树的节点保存 `ResolvedMove` 与它支付的语义卡牌，不把那一手的手牌下标当作
  身份。对手手牌每局重采样，手牌下标在树内不成立。
* 每次 simulation 在当前 determinization 的手牌里重新绑定下标（`rebind_cards`）：
  按语义找到等价的牌。找不到等价卡的分支直接剪枝，绝不"顺着下标"打出另一张牌。

## 6. Rust 必须导出

```text
尺寸：BOARD_CELLS, LINK_CELLS, MERCHANT_COUNT, SEAT_COUNT, INDUSTRY_COUNT,
      CARD_SEMANTIC_COUNT, LOCATION_COUNT, TOKEN_COUNT
宽度：F_CELL, F_LINK, F_MERCHANT, F_SEAT, F_GLOBAL, ACTION_FEATURE_DIM
版本：STATE_TOKEN_SCHEMA_VERSION, ACTION_SCHEMA_VERSION
拓扑：BOARD_CELL_LOCATIONS, BOARD_CELL_SLOTS, CONNECTION_ENDPOINTS,
      CONNECTION_VIA_FARMS
特征偏移：CELL_*, LINK_*, MERCHANT_*, SEAT_*, GLOBAL_*, ACTION_OFF_*,
          ACTION_REF_*
尺度：VP_SCALE, ACTION_REF_CAP, ACTION_KIND_COUNT, ACTION_NUMBERS,
      REF_KIND_COUNT
```

运行时调用：

* `GameState.state_tokens(perspective=None)` → 五个 token 组，按行动方旋转
* `GameState.legal_candidates()` → canonical 字符串 + `(N, 55)` 动作引用行
* `GameState.player_vps()` / `final_ranking()` → 终局 VP 与官方名次

## 7. 明确不做的事

* **不编码派生后果。** 翻面数、建满城市、新触及商家这类人工标量一律删除：网络能
  从被引用的 cell / link token 直接看到它们。
* **不编码手牌顺序。** 见 §2.6。
* **不把采样出的对手手牌当作真实信息。** 见 §2.6 的 `SEAT_HAND_SAMPLED_FLAG`。
* **不用名次作为唯一价值尺度。** 见 §4.1。
* **不用手牌下标作为树内动作身份。** 见 §5。

## 8. 改动维护清单

改动本文任一条，必须同步：

1. `engine/src/bridge/encode.rs`：token 特征与映射，bump `STATE_TOKEN_SCHEMA_VERSION`。
2. `engine/src/bridge/action_features.rs`：动作引用编码，bump `ACTION_SCHEMA_VERSION`。
3. `engine/src/bridge/pymod.rs`：导出常量与运行时调用。
4. `engine/src/ai/nn_mcts.rs`：终局效用、树内动作身份、Q 初始化。
5. `python/brass_ai/hierarchical_policy.py`：常量与运行时校验。
6. `python/brass_ai/net.py`：token 编码器与各头。
7. `python/brass_ai/train.py`：目标字段、损失权重与 `VP_SCALE`。
8. `python/brass_ai/selfplay.py`、`selfplay_loop.py`：样本形态与对齐。
9. 测试：`python/tests/`、`engine/src/bridge/` 单元测试、`engine/tests/engine_tests.rs`。

任何 schema 变更都会让旧 checkpoint 与旧样本失效。本项目不接受旧产物兼容。
