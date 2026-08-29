# 动作特征编码与网络头设计

本文描述当前 candidate-scoring 方案中**动作特征（235 维）的具体布局**与 **Python 网络各输出头（policy / value / econ / type）的设计**，并给出每类动作的实测编码示例。

对应实现：`engine/src/bridge/action_features.rs`（编码）、`python/brass_ai/net.py`（网络）、`python/brass_ai/train.py`（损失）。
当前版本：`ACTION_FEATURE_SCHEMA_VERSION = 3`，`STATE_FEATURE_SCHEMA_VERSION = 3`。状态张量（board/links/global/手牌）的布局见 `engine/src/bridge/encode.rs` 模块注释与 `docs/architecture.md`。

## 1. 总览：candidate-scoring 链路

本项目**没有固定长度的 policy 索引空间**。动作空间是 卡 × 位置 × 资源来源 的组合爆炸（满合法集可达数千甚至上万），因此采用逐候选打分（pointwise scoring）而不是 AlphaZero 式 flat policy 头：

```text
Rust legal_resolved_moves()          枚举完整可执行动作（含资源与卡牌选择）
  -> action_features::encode_move()  每个动作 -> 235 维特征行
  -> move_codec::encode()            每个动作 -> canonical 字符串（跨边界身份，无损可逆）
  -> net.py PolicyValueNet           state trunk + 每候选 [state, action_emb] -> logit
  -> masked log_softmax              只在合法候选上归一化（合法性由引擎构造性保证）
  -> MCTS visit 分布 / teacher 分布   按 canonical 字符串对齐回训练目标
```

关键性质：

- 网络只对引擎给出的候选打分，**从不学习合法性**（`net.py` 模块注释）。
- 动作的身份标识是 canonical 字符串（`move_codec`，roundtrip 有测试覆盖），不是位置索引；候选顺序在 Rust/Python 之间通过对齐函数（`selfplay._candidate_policy`）映射。
- `card_index` 只是执行引用；打牌语义从行动玩家手牌读取（见 §2 CARD 块），网络不学习手牌顺序。

## 2. 235 维布局

11 个块，总维度 7+35+27+4+6+6+39+39+47+9+16 = 235。

| 块 | offset | 宽度 | 语义 |
| --- | --- | --- | --- |
| `ACTION` | 0 | 7 | 动作类型 one-hot：0 Build、1 Network（含双铁路）、2 Develop、3 Sell、4 Loan、5 Scout、6 Pass |
| `CARD` | 7 | 35 | 打出牌的语义：0–26 位置牌（地点 id）、27–32 行业牌（行业 id，一张牌最多亮 2 个）、33 万能位置牌、34 万能行业牌 |
| `LOCATION` | 42 | 27 | Build 目标地点 one-hot |
| `CITY_SLOT` | 69 | 4 | Build 目标城市槽位（0–3） |
| `INDUSTRY_1` | 73 | 6 | 主行业 one-hot（Build 的建造行业 / Develop 的第一个移除行业 / Sell 的免费开发行业） |
| `INDUSTRY_2` | 79 | 6 | Develop 第二个移除行业 / 预留 |
| `CONNECTION_1` | 85 | 39 | 单铁路目标 / 双铁路第一条连接 one-hot |
| `CONNECTION_2` | 124 | 39 | 双铁路第二条连接 one-hot |
| `SELL_KEY` | 163 | 47 | Sell 的目标城市槽位（全局槽位 key），卖几个亮几个 |
| `MERCHANT` | 210 | 9 | Sell 的目标商家 one-hot |
| `SUMMARY` | 219 | 16 | 聚合资源/形状特征，逐项见下表 |

SUMMARY 16 项：

| 偏移 | 含义 | 取值方式 |
| --- | --- | --- |
| +0 | 煤总量 | Build/Develop 类：`个数/2`；单铁路 `0.5`；双铁路 `1.0` |
| +1 | 铁总量 | `个数/2` |
| +2 | 啤酒总量 | 双铁路 `1.0`；Sell 每张待售板块 `+0.25` |
| +3 | 市场煤 | 每 1 市场煤 `+0.5` |
| +4 | 矿区煤 | 每 1 免费矿区煤 `+0.5` |
| +5 | 市场铁 | 每 1 市场铁 `+0.5` |
| +6 | 自家酒馆啤酒 | 双铁路 own `1.0`；Sell 每张非商家啤酒板块 `+0.25` |
| +7 | 对手酒馆啤酒 | 仅双铁路 opponent `1.0`（Sell 不写此项） |
| +8 | 商家啤酒 | 双铁路 merchant `1.0`；Sell 每张 use_merchant_beer 板块 `+0.25` |
| +9 | 单铁路标志 | Network 为 1 |
| +10 | 双铁路标志 | NetworkDouble 为 1 |
| +11 | 免费开发标志 | Sell 附带 free_develop 为 1 |
| +12 | 卖出板块数 | `个数/4` |
| +13 | Scout 标志 | Scout 为 1 |
| +14/+15 | 未使用 | 恒 0 |

CARD 块的重要细节：三次 `card()` 调用（Scout 弃三张牌）写入的是**同一个 35 维块的按位叠加**——只留下"这组牌的语义并集"，丢掉每张牌的对应关系与重复张数（见 §7.4）。

## 3. 每类动作实测示例

以下示例全部来自真实对局（`GameState(seed=7, players=4)` 启发式自对弈，用 `legal_candidates()` dump）。每节列出一个候选动作的**全部非零位**。

### 3.1 Build

`Build 制造厂 at Derby`（运河时代第 1 轮）：

```text
canonical: ResolvedMove{operation:Build{loc:1,slot:1,ind:3,coal:K;max;4;0|K;max;5;0,iron:,card:1}}
  ACTION[0] = 1        # Build
  CARD[27]  = 1        # 行业牌：棉纺厂图标
  CARD[30]  = 1        # 行业牌：制造厂图标（一张双图标牌）
  LOCATION[1] = 1      # 目标地点 Derby
  CITY_SLOT[1] = 1     # 城市槽位 1
  INDUSTRY_1[3] = 1    # 制造厂
  SUMMARY[0] = 1       # 2 煤 / 2
  SUMMARY[3] = 1       # 2 煤全部来自市场（2 × 0.5）
```

解读：花 2 市场煤的建造。煤的市场花费可由 `SUMMARY[3]`（买几个）+ 状态张量里的煤市场 one-hot（当前价格档）推断——市场补足固定取最便宜档位、按升序付款（`game_state/graph.rs` find_coal_sources / `actions/common.rs` source_options），**玩家对市场没有选择空间**，所以编码里不需要市场档位信息。`SUMMARY[4]` 亮则表示有免费矿区煤（花费为 0）。

### 3.2 Network（单铁路）

`Network Burton-on-Trent - Cannock`（铁路时代）：

```text
canonical: ResolvedMove{operation:Network{conn:10,coal:M;17;0;1,card:0}}
  ACTION[1] = 1          # Network
  CARD[4] = 1            # 位置牌（loc 4）
  CONNECTION_1[10] = 1   # 连接 10
  SUMMARY[0] = 0.5       # 1 煤
  SUMMARY[4] = 0.5       # 1 免费矿区煤（花费 0）
  SUMMARY[9] = 1         # 单铁路标志
```

运河时代单铁路不需要煤，此时 `SUMMARY` 只有 `SUMMARY[9] = 1`。

### 3.3 NetworkDouble（双铁路，铁路时代）

`Network x2: Burton-on-Trent - Derby and Belper - Derby`：

```text
canonical: ResolvedMove{operation:NetDouble{c1:11,c2:0,coal1:M;17;0;1,coal2:M;17;0;1,beer:O;11;-;-,card:0}}
  ACTION[1] = 1           # Network（双铁路同属类型 1）
  CARD[3] = 1             # 位置牌（loc 3）
  CONNECTION_1[11] = 1    # 第一条连接
  CONNECTION_2[0] = 1     # 第二条连接
  SUMMARY[0] = 1          # 2 煤
  SUMMARY[2] = 1          # 1 啤酒
  SUMMARY[4] = 1          # 2 免费矿区煤
  SUMMARY[6] = 1          # 啤酒来自自家酒馆
  SUMMARY[10] = 1         # 双铁路标志
```

啤酒来源 kind（自家 `SUMMARY[6]` / 对手 `SUMMARY[7]` / 商家 `SUMMARY[8]`）三选一，但**具体是哪一家酒馆/哪个对手/哪个商家不编码**（见 §7.2）。

### 3.4 Develop

`Develop 棉纺厂 + 棉纺厂`（一次移除两个板块）：

```text
canonical: ResolvedMove{operation:Develop{ind1:0,ind2:0,iron:max;2;0|max;2;0,card:0}}
  ACTION[2] = 1
  CARD[4] = 1            # 位置牌（loc 4）
  INDUSTRY_1[0] = 1      # 第一个移除：棉纺厂
  INDUSTRY_2[0] = 1      # 第二个移除：棉纺厂
  SUMMARY[1] = 1         # 2 铁
  SUMMARY[5] = 1         # 2 市场铁
```

铁的规则与煤不同：任意未翻面铁厂都免费且无需连通（`docs/brass-birmingham-rules.md` §5.2），所以 `SUMMARY[5]` 亮说明场上没有可用铁厂（或免费铁不够），全部从市场购买。

### 3.5 Sell

`Sell 2 tile(s)`：

```text
canonical: ResolvedMove{operation:Sell{keys:4;9,merchants:7;8,beer:1;1,sources:P;18;-;-~,free:-,card:0}}
  ACTION[3] = 1
  CARD[11] = 1           # 位置牌（loc 11）
  SELL_KEY[4] = 1        # 卖出槽位 4
  SELL_KEY[9] = 1        # 卖出槽位 9
  MERCHANT[7] = 1        # 目标商家 7
  MERCHANT[8] = 1        # 目标商家 8
  SUMMARY[2] = 0.5       # 2 张待售板块（2 × 0.25）
  SUMMARY[8] = 0.5       # 2 张都用商家啤酒（2 × 0.25）
  SUMMARY[12] = 0.5      # 2 / 4
```

注意 canonical 里 `sources:P;18;-;-`（对手酒馆啤酒，编码字母 O=自家 / P=对手 / M=商家，见 `move_codec.rs:12`）——这是引擎为满足板块 2 啤酒需求**确定性补足**的酒厂啤酒，235 维里完全没有体现（Sell 分支用 `..` 忽略 `beer_sources`）。它不是玩家选择，而是状态的确定函数，但网络需要自行学会推断（见 §7.3）。

### 3.6 Loan / Scout / Pass

```text
Loan:  ACTION[4] = 1;  CARD[4] = 1        # 只有动作类型 + 打出牌
Scout: ACTION[5] = 1;  CARD[4] = 1; CARD[12] = 1; CARD[31] = 1;  SUMMARY[13] = 1
                                        # 三张弃牌语义并集叠加在同一 CARD 块
Pass:  ACTION[6] = 1;  CARD[4] = 1
```

Scout 的三张牌亮三个 CARD 位（此处：位置牌 loc 4、位置牌 loc 12、行业牌 Pottery）。

## 4. 数值约定与 uint8 压缩不变量

**所有 235 维特征值都是 0.25 的整数倍**（0 / 0.25 / 0.5 / 1 / 1.5 / 2…）。这是回放存储无损压缩的前提：`hierarchical_policy.compress_candidate_features` 按 `×4` 打包成 uint8，遇到非 0.25 步长的值会直接报错（`hierarchical_policy.py:60-70`）。**修改 schema 新增特征时必须维持该不变量**，否则回放数据无法无损存储。

## 5. 网络头设计（net.py / train.py）

共享 state trunk（图消息传递 + global + 双方手牌 -> 256 维）之上共有 4 个输出头：

### 5.1 Policy 打分头（主头）

```text
action_encoder: Linear(235 -> 128) + ReLU + Linear(128 -> 128) + ReLU
action_score:   Linear(256 + 128 -> 256) + ReLU + Linear(256 -> 1)
logits = score([state ⊕ action_emb]) 对每个候选独立打分
candidate_log_probs = log_softmax(masked_fill(padding, -inf))
```

- 逐候选 pointwise 打分，候选之间不互相作用；变长候选集由 `pad_candidate_features` padding + bool mask 处理。
- 损失：CE `-(Σ_a p_a · log q_a)`，padding 位填 0。
- 布局依赖：`train.compute_loss` 用 `candidates[..., :7]` 切片取 ACTION 块（见 §5.4）。

### 5.2 Value head（价值头）

- 结构：`Linear(trunk -> 4)`，**无 tanh**，直接回归。
- 目标：四名玩家终局 VP 的 z-score 向量 `(vp - mean) / max(std, ε)`（`selfplay._normalize`）； std 过小时（平局/极接近）目标为全零。
- 损失：对 4 维向量的 MSE。搜索时 MaxN/ISMCTS 按行动方取对应座位分量。

### 5.3 Econ head（经济辅助头）

- 结构：`Linear(trunk -> 2)`，输出 `(income_level, money)`。
- 目标归一化：income `(x + 10) / 40` clamp 到 [0,1]；money `x / 100` clamp 到 [0,1]；预测端同样归一化后算 MSE。
- 加权：收入为负的样本 MSE 权重 ×3（`econ_neg_weight`）——"负经济是真实风险"的先验，不施加收入最大化目标。
- 盖章规则：运河时代样本在 `finish_canal_era` 时盖 canal-end 经济；铁路时代样本盖终局经济（`selfplay.play_game_with_roles`）。
- 总权重 `econ_lambda = 0.2`。

### 5.4 Type 辅助头（动作类型）

- 结构：`Linear(trunk -> 7)`，作用在 state trunk 上（与具体候选无关）。
- 目标：由候选 policy 目标对 ACTION 块边缘化得到：`type_target = einsum("bn,bnt->bt", policy, candidates[..., :7])`——**依赖 ACTION 块位于 0..7 列的布局约定**，改动布局必须同步 `train.py`。
- 损失权重 0.1。设计意图：引导粗粒度动作选择，但不重新引入 flat slot 头。

### 5.5 损失汇总与默认超参

```text
L = policy_CE + value_MSE + 0.1 * type_CE + 0.2 * (inc_MSE + money_MSE) + 1e-4 * ||θ||²
```

| 超参 | 默认值 | 说明 |
| --- | --- | --- |
| `epochs` / `batch_size` | 5 / 256 | 每次调用训练的遍数与批大小 |
| `lr` | 1e-3 | AdamW；`weight_decay=0`（L2 单独算） |
| `l2` | 1e-4 | 显式 L2 |
| 调度 | CosineAnnealingLR | `T_max=100`，`min_lr=1e-5`，跨迭代保持 optimizer 状态 |
| `econ_lambda` / `econ_neg_weight` | 0.2 / 3.0 | 经济辅助权重 / 负收入加权 |
| `max_candidate_batch` | 65536 | 单个训练 micro-batch 的候选行预算，限制 padding 显存峰值 |
| `amp` | True | GPU 上 fp16 autocast |

## 6. 训练目标来源

| 模式 | policy 目标 | 候选集 |
| --- | --- | --- |
| MCTS self-play | visit 分布按 canonical 字符串对齐到当前候选（`selfplay._candidate_policy`） | 搜索所用候选 |
| Imitation（shortlist） | teacher 分数 softmax（温度 1.0） | heuristic top-k（每类 ≤4，总 ≤14） |
| Imitation（full-legal） | teacher 动作 one-hot；样本只存 snapshot + teacher canonical，训练前用 `materialize_sample` 实时物化候选 | 完整合法集 |

MCTS 搜索默认也用启发式 shortlist（`nn_mcts.rs` 的 `candidate_k=4`），这一训练/推理分布错配是已知问题（见 `docs/ai-advise.md` roadmap）。

## 7. 已知信息损失（实测）

以下数据来自 3 局启发式自对弈（seed 7/21/99，2026-08 测量），方法：对每个状态把合法候选按 235 维特征 hash 分组，再区分"同特征不同动作"的原因。

### 7.1 按设计折叠：手牌选择（不算缺陷）

同一结构动作的卡牌选择按**打牌语义**折叠（不同手牌位置的相同语义牌得到同一向量）。单局被折叠的冗余候选：Network ~13k、Develop ~18.7k、Scout ~3.4k、Build ~1k、Loan/Pass 各 ~730、Sell ~380。这是"网络不学习手牌顺序"的刻意设计；同一语义牌选哪张执行结果完全相同，折叠无信息损失。

### 7.2 真实损失：免费来源身份（影响"谁的板块被翻面"）

煤/铁免费来源（矿区/铁厂）的**取自哪一个 key** 是真实的玩家选择自由度（`actions/common.rs` source_options 明确枚举，注释原文："that choice decides whose building flips"——对手板块被取空会推进其翻面，相当于喂对手 VP/收入；啤酒同理）。但 235 维只编码"免费 vs 市场"的**数量**，不编码来源身份。

实测碰撞（同一 235 维向量、同一结构动作、只差来源身份）：

```text
[Network]  seed 7 round 2:
    coal:M;20;0;1   vs   coal:M;17;0;1        # 两座不同的连通煤矿，特征完全相同
[NetworkDouble] seed 7 round 2:
    beer:O;15;-;-   vs   beer:O;43;-;-        # 两家不同的自家酒馆，特征完全相同
[Build]    seed 21 round 5:
    coal:M;20;0;1   vs   coal:M;1;0;1
```

发生频率（启发式打法下）：

| 动作 | 存在 ≥2 个免费来源身份的状态 | 其中发生特征碰撞的状态 |
| --- | --- | --- |
| Network | 22 / 351 | 21 / 351 |
| NetworkDouble | 16 / 50 | 9 / 50 |
| Build | 4 / 340 | 1 / 340 |
| Develop | 0 / 367 | 0 |

解读：双铁路最暴露（约 1/3 的双铁路状态有身份选择，其中多数特征相同）；Network 次之；Build 在启发式打法下罕见但真实存在。一旦发生，网络给这些动作**完全相同的 prior**，MCTS 只能靠 visit/.value 差异区分，selfplay 的 policy 目标在它们之间退化为任意分配；最终 Top-3 推荐里同分动作的排序也是任意的。

### 7.3 未编码的派生信息（不是选择，但网络需自行推断）

- **市场档位**：市场补足固定取最便宜档、按升序付款（`graph.rs`），玩家无选择空间；成本可由"市场计数 + 市场状态 one-hot"推断，无需编码。
- **Sell 的酒厂啤酒**：`beer_sources` 由引擎按 `find_beer_sources` 顺序确定性补足（`sell.rs plan_sell_beer_sources`），不是选择；但编码完全忽略它（`..`），其 kind（自家/对手）与数量需要网络自行从盘面推断——对手酒馆被消耗推进对手翻面这一信息是隐式的。
- **SUMMARY[6..8] 的语义**：Sell 分支里它们按"板块是否用商家啤酒"计数，与双铁路分支的"啤酒 kind"语义不同（`SUMMARY[2]` 在 Sell 里是板块数不是啤酒数）。

### 7.4 Scout 弃牌多重集坍缩

Scout 把三张弃牌的语义并集叠加进同一 CARD 块（`action_features.rs:184-188`）：{A,A,B} 与 {A,B,B} 的弃牌组合会得到相同向量。弃牌进入牌堆顶影响后续牌序信息，属真实差异，但发生条件较苛刻（需要手牌有重复语义牌），影响面小。

### 7.5 迭代原则

如需升级编码（如为免费来源身份增加编码位、区分 Sell 啤酒 kind），按 `docs/ai-advise.md` 的既定原则：先由错误案例证明当前 encoder 无法区分关键局面，再增量升级；升级必须 bump `ACTION_FEATURE_SCHEMA_VERSION` 并接受旧 checkpoint/回放失效重采样。

## 8. 修改 schema 的维护清单

改动 235 维布局时，以下位置必须同步（当前均有测试或运行时校验兜底）：

1. `engine/src/bridge/action_features.rs`：布局常量 + `encode_move` + bump `ACTION_FEATURE_SCHEMA_VERSION`。
2. `python/brass_ai/hierarchical_policy.py`：`ACTION_FEATURE_SCHEMA_VERSION` / `ACTION_FEATURE_DIM` 常量（运行时强校验）。
3. `python/brass_ai/train.py`：`candidates[..., :7]` 切片（ACTION 块位置）；`/4.0` 还原（0.25 步长约定）。
4. 保持所有特征值为 0.25 的倍数（uint8 回放压缩不变量，违规会运行时报错）。
5. 相关测试：`python/tests/test_engine.py`（shape/schema 回归）、`test_hierarchical_policy.py`（schema 门禁、压缩无损、teacher 对齐）、`engine/src/bridge` 单元测试与 `engine/tests/engine_tests.rs`。
