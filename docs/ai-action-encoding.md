# 动作特征编码与网络头设计

本文描述当前 candidate-scoring 方案中**动作特征（301 维，schema v4）的具体布局**与 **Python 网络各输出头（policy / rank / winner / econ）的设计**，并给出每类动作的实测编码示例。v4 的设计动机与完整方案见 `ai-encoding-v4-design.md`。

对应实现：`engine/src/bridge/action_features.rs`（编码）、`python/brass_ai/net.py`（网络）、`python/brass_ai/train.py`（损失）。
当前版本：`ACTION_FEATURE_SCHEMA_VERSION = 4`，`STATE_FEATURE_SCHEMA_VERSION = 4`。状态张量（board 24 平面 / links / global 168 / 手牌）的布局见 `engine/src/bridge/encode.rs` 模块注释与 `docs/architecture.md`。

## 1. 总览：candidate-scoring 链路

本项目**没有固定长度的 policy 索引空间**。动作空间是 卡 × 位置 × 资源来源 的组合爆炸（满合法集可达数千甚至上万），因此采用逐候选打分（pointwise + 集合上下文）而不是 AlphaZero 式 flat policy 头：

```text
Rust legal_resolved_moves()          枚举完整可执行动作（含资源与卡牌选择）
  -> action_features::encode_move()  每个动作 -> 301 维特征行
  -> move_codec::encode()            每个动作 -> canonical 字符串（跨边界身份，无损可逆）
  -> net.py PolicyValueNet           FiLM 调制的动作嵌入 + 候选集上下文 -> logit
  -> masked log_softmax              只在合法候选上归一化（合法性由引擎构造性保证）
  -> MCTS visit 分布 / teacher 分布   按 canonical 字符串对齐回训练目标
```

关键性质：

- 网络只对引擎给出的候选打分，**从不学习合法性**（`net.py` 模块注释）。
- 动作的身份标识是 canonical 字符串（`move_codec`，roundtrip 有测试覆盖），不是位置索引；候选顺序在 Rust/Python 之间通过对齐函数（`selfplay._candidate_policy`）映射。**身份与表示严格分离**：canonical 字符串只做跨边界对齐（A == B?），从不直接作为网络输入。
- `card_index` 只是执行引用；打牌语义从行动玩家手牌读取（见 §2 CARD 块），网络不学习手牌顺序。
- **动作特征只编码"选择"，"身份"交给状态张量**：DRAIN 引用棋盘格（与 board 张量同索引），格上行业/归属/翻面由状态平面提供，网络在打分头 join。
- **集合上下文**（v4）：每个候选的打分输入包含候选集的 masked-mean 嵌入，可表达"集合里还有别的动作在竞争同一资源/位置"（O(N)，非 self-attention）。

## 2. 301 维布局

14 个块，总维度 7+35+27+4+6+6+39+39+47+9+49+9+12+12 = 301。

| 块 | offset | 宽度 | 语义 |
| --- | --- | --- | --- |
| `ACTION` | 0 | 7 | 动作类型 one-hot：0 Build、1 Network（含双铁路）、2 Develop、3 Sell、4 Loan、5 Scout、6 Pass |
| `CARD` | 7 | 35 | 打出牌的语义**计数**：0–26 位置牌、27–32 行业牌（一张牌最多亮 2 个）、33/34 万能牌。单牌动作 = 1.0（等价原 one-hot）；Scout 三张弃牌按语义累加 1.0/2.0/3.0，弃牌多重集可分辨 |
| `LOCATION` | 42 | 27 | Build 目标地点 one-hot |
| `CITY_SLOT` | 69 | 4 | Build 目标城市槽位（0–3） |
| `INDUSTRY_1` | 73 | 6 | 主行业 one-hot（Build 的建造行业 / Develop 的第一个移除行业 / Sell 的免费开发行业） |
| `INDUSTRY_2` | 79 | 6 | Develop 第二个移除行业 |
| `CONNECTION_1` | 85 | 39 | 单铁路目标 / 双铁路第一条连接 one-hot |
| `CONNECTION_2` | 124 | 39 | 双铁路第二条连接 one-hot |
| `SELL_KEY` | 163 | 47 | Sell 的目标城市槽位（全局槽位 key），卖几个亮几个 |
| `MERCHANT` | 210 | 9 | Sell 的目标商家 one-hot |
| `DRAIN` | 219 | 49 | **按棋盘格索引**（47 城市槽 + 2 农场，与 board 张量同索引）该动作抽取的资源量（cubes/4）。煤/铁/酒厂啤酒共用一个块：一格只有一种行业，**种类与归属由状态平面 join 提供**。市场购买无身份，不进本块 |
| `MERCHANT_BEER` | 268 | 9 | 从各商家抽走的啤酒量（cubes/4） |
| `CONSEQUENCE` | 277 | 12 | 解析后果：[0]建链数/2、[1]网络新触及地点/4、[2]新触及商家 0/1、[3]/[4]翻面数 自家/对手（各 /4）、[5]overbuild、[6]升级、[7]建满城市、[8]待售板块/4、[9]免费开发、[10]/[11]保留 |
| `SUMMARY` | 289 | 12 | [0]煤总量/2、[1]铁总量/2、[2]啤酒总量、[3]市场煤数、[4]市场铁数、[5]商家啤酒数、[6]待售板块/4、[7]单铁路、[8]双铁路、[9]免费开发、[10]Scout、[11]保留 |

三套索引系的边界（详表见 `ai-encoding-v4-design.md` §1.1）：**49 棋盘格** = 行业板块体系（DRAIN/SELL_KEY/board）；**9 商家板块**挂在商家地点上（MERCHANT/MERCHANT_BEER/状态商家块）；**市场**虚拟无身份（SUMMARY 计数 + global one-hot，无啤酒）。

## 3. 每类动作实测示例

以下示例全部来自真实对局（seed 7/21/99 启发式自对弈，teacher 所选候选，2026-08-29 v4 dump）。每节列出一个候选动作的**全部非零位**。

### 3.1 Build

`Build（Derby 槽 1，制造厂）`：

```text
ACTION[0]=1  CARD[27]=1  CARD[30]=1     # 行业牌（棉纺厂+制造厂双图标）
LOCATION[1]=1  CITY_SLOT[1]=1  INDUSTRY_1[3]=1
SUMMARY[0]=0.5   # 1 煤（/2）
SUMMARY[3]=1     # 1 个市场煤
```

无 DRAIN 位 → 煤全部来自市场；`SUMMARY[3]` 是市场煤个数（无身份），成本可由它 + global 的煤市场 one-hot 推断。DRAIN 有值 = 抽了免费矿区煤，翻面后果见 `CONSEQUENCE[3]/[4]`。

### 3.2 Network（单铁路，运河时代）

```text
ACTION[1]=1  CARD[28]=1  CONN_1[23]=1
CONSEQUENCE[0]=0.5   # 建 1 条链（/2）
CONSEQUENCE[1]=0.5   # 网络新触及 2 个地点（/4）
CONSEQUENCE[2]=1     # 新触及商家
SUMMARY[7]=1         # 单铁路标志
```

运河时代单铁路不需要煤，SUMMARY 资源位全零；铁路时代会亮 `SUMMARY[0]`（煤量/2）、`SUMMARY[3]`（市场煤）或 `DRAIN[key]`（免费矿）。

### 3.3 NetworkDouble（双铁路，铁路时代）

```text
ACTION[1]=1  CARD[27]=1  CARD[30]=1  CONN_1[20]=1  CONN_2[21]=1
DRAIN[18]=0.25               # 从棋盘格 18 的免费矿抽了 1 煤
CONSEQUENCE[0]=1             # 建 2 条链
CONSEQUENCE[1]=0.5           # 新触及 2 个地点
CONSEQUENCE[2]=1             # 新触及商家
CONSEQUENCE[3]=0.25          # 自家 1 个板块被抽空翻面（/4）
SUMMARY[0]=1                 # 2 煤（/2）
SUMMARY[2]=1                 # 1 啤酒
SUMMARY[3]=2                 # 2 个市场煤
SUMMARY[8]=1                 # 双铁路标志
```

"抽的是谁家的矿"不再编码在动作里：DRAIN[18] 与状态 board 第 18 格的 owner/industry/cubes/flipped 平面 join 即得。啤酒来源同理——酒厂啤酒进 DRAIN，商家啤酒进 `MERCHANT_BEER`。

### 3.4 Develop

```text
ACTION[2]=1  CARD[31]=1  INDUSTRY_1[5]=1  INDUSTRY_2[5]=1
SUMMARY[1]=1     # 2 铁（/2）
SUMMARY[4]=2     # 2 个市场铁
```

铁的规则与煤不同：任意未翻面铁厂都免费且无需连通（`docs/brass-birmingham-rules.md` §5.2），`SUMMARY[4]` 亮说明免费铁不够、全部从市场购买；若抽了免费铁厂，DRAIN 亮对应格且 `CONSEQUENCE[4]` 可能计翻面。

### 3.5 Sell

```text
ACTION[3]=1  CARD[28]=1  SELL_KEY[4]=1  MERCHANT[7]=1
CONSEQUENCE[8]=0.25   # 1 张待售板块（/4）
SUMMARY[6]=0.25       # 1 / 4
```

该例板块不需要啤酒。若需要，啤酒支付计划（引擎确定性补足）现在显式进 DRAIN / MERCHANT_BEER——不再是网络自行推断的隐式信息。

### 3.6 Loan / Scout / Pass

```text
Loan:  ACTION[4]=1; CARD[29]=1
Scout: 三张弃牌按语义累加计数（如两张同位置牌 → CARD[loc]=2），SUMMARY[10]=1
Pass:  ACTION[6]=1; CARD[..]=1
```

Scout 的计数式 CARD 使 {A,A,B} 与 {A,B,B} 可分辨（v3 的按位 OR 会坍缩）。

## 4. 数值约定与 uint8 压缩不变量

**所有 301 维特征值都是 0.25 的整数倍**（0 / 0.25 / 0.5 / 1 / 1.5 / 2…）。这是回放存储无损压缩的前提：`hierarchical_policy.compress_candidate_features` 按 `×4` 打包成 uint8，遇到非 0.25 步长的值会直接报错（`hierarchical_policy.py:60-70`）。

澄清两条容易误解的边界：

- **作用范围只有"候选动作特征"这一种张量**。状态张量（board/links/global/手牌）是 float32，不走 uint8 打包，不受此约束。
- **63.75 是"归一化后的特征值"的上限，不是游戏数值的上限**。游戏里的原始数量（钱 £200、VP 120、8 个煤）从不直接写进特征——每一维都是除以设计常数后的归一化值（钱 /200、煤量 /2、计数 /4、one-hot 1.0）。"8 个煤"编码为 8/2 = 4.0，×4 = 16，离 255 很远。新增特征时要做的是选一个足够大的归一化常数，使**归一化值 ≤ 63.75**（实践中 ≤3 就够）；0.25 步长对这些离散计数/one-hot 特征是无损的。

即新增特征必须同时满足两条约束：`值 × 4` 是整数（0.25 步长），且 `0 ≤ 值 × 4 ≤ 255`。

## 5. 网络头设计（net.py / train.py，v4）

共享 state trunk（图消息传递 + global + 双方手牌 -> 256 维）之上的输出头：

### 5.1 Policy 打分头（主头：FiLM + 集合上下文）

```text
action_encoder: Linear(301 -> 128) + ReLU + Linear(128 -> 128) + ReLU
film:           Linear(trunk -> 2*128)                    # γ, β 状态调制
a'              = γ ⊙ action_emb + β
ctx             = masked_mean(a' over candidate set)      # (B,128) 集合上下文
action_score:   Linear(3*128 -> 256) + ReLU + Linear(256 -> 1)
logits          = score([a' ⊕ ctx ⊕ (a' ⊙ ctx)])
candidate_log_probs = log_softmax(masked_fill(padding, -inf))
```

- FiLM 让动作-状态交互成为结构保证（乘性调制而非纯拼接依赖 MLP 自己学）。
- 集合上下文 `ctx` 让每个候选的分数依赖**候选集合**——可表达"集合里还有别的动作在竞争同一资源/位置"，仍是 O(N)（无候选间 self-attention）。
- 损失：CE `-(Σ_a p_a · log q_a)`，padding 位填 0。

### 5.2 Rank head + Winner head（替代 v3 的 VP z-score value 头）

- **rank head**：`Linear(trunk -> 4)`，目标 = 每座位终局名次 / n（并列取平均名次），MSE。**跨局可比、局内保序**。
- **winner head**：`Linear(trunk -> 4)`，softmax + CE，目标 = 并列冠军的均匀分布。
- 搜索尺度：网络叶子值与终局 backup（`nn_mcts.rs terminal_value`）都用 `score_p = 1 − rank_p/n`（并列同秩），MaxN 按行动方取自己的分量最大化。
- v3 的逐局 VP z-score 头已移除（跨局只保序不保距、margin 信息丢失是已记录的权衡，v4 以 rank/winner 直接对齐"第一名概率"这一最终目标）。

### 5.3 Econ head（经济辅助头，按时代拆分）

- 结构：**canal 与 rail 两个独立头**（各 `Linear(trunk -> 2)`），输出拼接为 (B,4)；样本按所处时代只训练自己时代的头——消除 v3"同一 head 两种目标时间定义"的混杂。
- 目标：canal 样本盖 canal-end 经济、rail 样本盖终局经济；归一化 income `(x+10)/40`、money `/100`。
- 负收入加权 `econ_neg_weight` 默认 **1.0（关闭）**，保留开关供 ablation（v3 的 ×3 是未经验证的强归纳偏置）。
- 总权重 `econ_lambda = 0.2`。

### 5.4 已删除：Type 辅助头

v3 的 type head 是 policy 目标的低维边缘（`P(type)=Σ_{a∈type}P(a)`，无新信息），且使 `train.py` 耦合 ACTION 块布局（`candidates[..., :7]` 切片）。v4 删除；`train.py` 不再依赖任何布局约定。

### 5.5 损失汇总与默认超参

```text
L = policy_CE + rank_MSE + 0.5 * winner_CE + 0.2 * econ_MSE(era-split) + 1e-4 * ||θ||²
```

| 超参 | 默认值 | 说明 |
| --- | --- | --- |
| `epochs` / `batch_size` | 5 / 256 | 每次调用训练的遍数与批大小 |
| `lr` | 1e-3 | AdamW；`weight_decay=0`（L2 单独算） |
| `l2` | 1e-4 | 显式 L2 |
| 调度 | CosineAnnealingLR | `T_max=100`，`min_lr=1e-5`，跨迭代保持 optimizer 状态 |
| `econ_lambda` / `econ_neg_weight` | 0.2 / 1.0 | 经济辅助权重 / 负收入加权（默认关闭，留 ablation 开关） |
| `max_candidate_batch` | 65536 | 单个训练 micro-batch 的候选行预算（候选行 uint8 存储可再省 4×） |
| `amp` | True | GPU 上 fp16 autocast |

## 6. 训练目标来源

| 模式 | policy 目标 | 候选集 |
| --- | --- | --- |
| MCTS self-play | visit 分布按 canonical 字符串对齐到当前候选（`selfplay._candidate_policy`） | 搜索所用候选 |
| Imitation（shortlist） | teacher 分数 softmax（温度 1.0） | 生成器 v4：Build/Network/NetDouble 各 ≤4 几何体 × ≤2 来源变体，Develop/Sell 各 ≤2 计划，Loan/Scout/Pass 各 ≤1（**实测均值 ~12.4、上限 22**） |
| Imitation（full-legal） | teacher 动作 one-hot；样本只存 snapshot + teacher canonical，训练前用 `materialize_sample` 实时物化候选 | 完整合法集 |

候选集事实（`heuristic_ai::candidate_actions_k` 是**唯一**的候选生成器）：

- MCTS（`nn_mcts.rs`，`candidate_k=4`）与 shortlist imitation（`heuristic_candidates`）调用**同一个** `candidate_actions_k(state, 4)`——两者候选分布一致。plain MCTS（`mcts_ai.rs`）也用同一生成器（默认 k=3）。
- 生成器 v4（`SOURCE_VARIANTS=2`）：同一几何体（连接/建造位）下来源身份不同的变体**成对进入候选集**，"抽谁的矿/酒馆"由搜索而非生成器决定。
- 整个管线仍没有默认越过生成器的路径；NN-MCTS 显式传 `candidate_k=0` 可全合法集展开。
- **checkpoint 资产现状（2026-08-29，v4 落地后）**：`checkpoints/` 下所有 bootstrap checkpoint 均为旧 schema（v2/v3 状态张量、235 维动作特征、z-score value 头）训练产物，已被 v4 门禁全部拒绝。当前**没有可用的已训练策略**，首次 v4 full-legal bootstrap 待跑。

## 7. 架构瓶颈：三层风险与实测（评审驱动）

把系统抽象为 状态编码器 → 动作编码器 → pointwise scorer → 候选 softmax → MCTS 后，风险自上而下分三层（评审 `review-encoding.md` 的总结论）：

```text
1. Candidate Generator  是否漏掉好动作？      ← 最大风险
2. Action Representation 是否把关键动作折叠？
3. Pointwise Scorer     是否需要候选间交互？
```

**测量口径**：3 局启发式自对弈（seed 7/21/99，372 个决策状态，2026-08-29）。§7.1–7.4 的数字是 **v3 编码器时代**的测量——它们是 v4 方案的立项依据；§7.5 是 v4 落地后的复测。

### 7.1 第一层：候选生成器上限（v3 时代测量）

- 规模：全合法集均值 317.6 / 最大 2655；shortlist 均值 11.1 / 最大 17——**搜索平均只看到 3.5% 的合法候选**。
- **来源身份由生成器内部代为决定**：单/双铁路 scorer 都取第一个煤来源，双铁路啤酒偏好 Own。备选来源从不作为候选出现——net 与搜索都无机会评估"抽谁的矿/酒馆"。
- teacher 覆盖是构造性的（teacher canonical 恒在 shortlist 内），"teacher top-1 Recall@K"以启发式自身为参照没有鉴别力；生成器上限必须用更强参照度量。

### 7.2 第二层：编码碰撞的分类与 value regret（v3 时代测量）

非"仅卡牌序"的碰撞组共 3054 个：Scout 弃牌组合 1535、NetDouble 来源身份 1096、Network 来源身份 320、Sell 商家-啤酒配对 100、Build 来源身份 3。当时没有任何碰撞组有 ≥2 个成员同时进入 shortlist（0/372）。

value regret（211 个有不同后继的采样组，配对确定性启发式续演到终局，z = 终局 per-game z-score 之差）：

| 类型 | 采样组 | 执行等价 | regret>1VP | regret>4VP | 最大 regret |
| --- | --- | --- | --- | --- | --- |
| NetDouble（啤酒/煤矿身份） | 80 | 0 | 40% | 34% | 19.5 VP |
| Network（煤矿身份） | 80 | 0 | 6% | 6% | 7.75 VP |
| Scout（弃牌多重集） | 40 | 6 | 6% | 3% | 6.25 VP |
| Sell（商家-啤酒配对） | 60 | 46 | 0% | 0% | 0 |
| Build（煤矿身份） | 3 | 0 | 0% | 0% | 0 |

解读：碰撞多数无差（中位数 0），但双铁路的来源身份是高赌注折叠——约 1/3 的组终局差超 4VP。口径限制：确定性启发式续演会把微小盘面差放大（策略混沌），是"固定续演策略下后果差"的测量，不是最优价值差。

### 7.3 第一层的直接测量：被排除动作的天花板探针（v3 时代测量）

144 个探针（36 状态，全部命中被截断的 Develop 备选）：**32% 的被排除动作终局 margin 反超 teacher 选择（>0.5VP），28% >2VP，最大 +28.5VP**。启发式静态打分对被截断类型的备选排序错误率可观——生成器偏差的 self-confirming 闭环有实测依据。

### 7.4 三个建议指标的当前状态

| 指标 | 状态 |
| --- | --- |
| candidate recall（已训练策略在全合法集上的 top-1/top-3 落在 shortlist 内的比例） | 已定义，**待训练**——v4 无已训练 checkpoint（§6） |
| collision value regret | **已测**（§7.2）；v4 复测见 §7.5 |
| pointwise vs set-aware scorer gap | **部分解决**：v4 的集合上下文（§5.1）以 O(N) 引入候选集依赖；完整 set-aware（self-attention）仍待需要时投入 |

### 7.5 v4 落地后的复测（同口径，2026-08-29）

- **来源身份碰撞清零**：Network 320 → 0、Build 3 → 0、NetDouble 1096 → 105（剩余全部是"两条链接交换煤矿"的**执行等价**变体——DRAIN 只看格子集合，交换后结果相同）。
- 非"仅卡牌序"碰撞组 3054 → **1766**，剩余全部为执行等价类：Scout 同语义不同牌序变体 1573（弃牌多重集坍缩已由计数式 CARD 修复）、Sell 商家顺序变体 88、NetDouble 来源交换 105。
- shortlist 内同组变体成对出现（生成器 v4 生效，372 状态 1 例直接观测；变体生成的覆盖由 Rust 测试跨 20 seed 验证）。
- shortlist 规模：均值 ~12.4、范围 3–22（k=4 几何体 × ≤2 变体的新语义）。

即：v4 之后，编码器对"执行不同但价值可能不同"的动作**不再有已知折叠**；剩余碰撞全部执行等价，不构成表达力损失。

### 7.5 评审指出的其他结构项（P2，记录不动）

- **SUMMARY 语义复用**：同一字段在不同 ACTION 下语义不同（如 SUMMARY[2] 在 NetDouble 是啤酒、在 Sell 是待售板块数）。网络可经 ACTION one-hot 学会区分，不是 bug，但提高学习难度；SUMMARY[14]/[15] 空置保留。
- Scout multiset→set 折叠与 card index 折叠见 §8。

## 8. 信息损失台账

### 8.1 按设计折叠：手牌选择（不算缺陷）

同一结构动作的卡牌选择按**打牌语义**折叠（不同手牌位置的相同语义牌得到同一向量）。这是"网络不学习手牌顺序"的刻意设计；同一语义牌选哪张执行结果完全相同，折叠无信息损失。

### 8.2 免费来源身份 —— v4 已修复

v3 只编码"免费 vs 市场"的数量、不编码来源身份，而"抽谁家的矿/酒馆"是真实的选择自由度（`actions/common.rs`："that choice decides whose building flips"）。v3 时代的实测：NetDouble 来源碰撞组 34% 终局差 >4VP、最大 19.5VP（§7.2）。v4 的 DRAIN/MERCHANT_BEER 按棋盘格与商家索引编码抽取量，配合生成器把来源变体暴露给候选集（§7.5 复测：来源身份碰撞清零）。

### 8.3 状态张量的商家信息 —— v4 已修复

v3 状态张量不含商家任何信息（`buys` 收货类型与 `has_beer`）。v4 的 global 增加商家块：9 商家 × 5 收货类型 one-hot（Blank/Any/棉纺/制造厂/陶器——商家只收制成品，煤/铁/酒进不了商家，见 `map::merchant_tile_mix`）+ 9 has_beer（`ai-encoding-v4-design.md` §3）。

### 8.4 未编码的派生信息（不是选择，无需编码）

- **市场档位**：市场补足固定取最便宜档、按升序付款（`graph.rs`），玩家无选择空间；成本可由 SUMMARY 市场计数 + global 市场状态 one-hot 推断。
- **Sell 的酒厂啤酒**：`beer_sources` 由引擎确定性补足，不是选择；v4 起它显式进 DRAIN/MERCHANT_BEER（从"网络自行推断"升级为输入）。

### 8.5 Scout 弃牌多重集 —— v4 已修复

v3 的按位 OR 使 {A,A,B} 与 {A,B,B} 坍缩；v4 的计数式 CARD 保留多重集（§2）。

### 8.6 迭代原则

后续升级仍按 `docs/ai-advise.md` 的既定原则：先由错误案例证明当前 encoder 无法区分关键局面，再增量升级；升级必须 bump 对应 schema version；动作特征新增项必须满足 §4 的 0.25 步长与 63.75 上限。当前版本 v4 的设计全文见 `ai-encoding-v4-design.md`。

## 9. 修改 schema 的维护清单

改动 301 维布局或状态张量时，以下位置必须同步（当前均有测试或运行时校验兜底）：

1. `engine/src/bridge/action_features.rs`：布局常量 + `encode_move` + bump `ACTION_FEATURE_SCHEMA_VERSION`。
2. `engine/src/bridge/encode.rs`：状态平面/长度 + bump `STATE_FEATURE_SCHEMA_VERSION`。
3. `python/brass_ai/hierarchical_policy.py`：`ACTION_FEATURE_SCHEMA_VERSION` / `ACTION_FEATURE_DIM` 常量（运行时强校验）。
4. `python/brass_ai/train.py`：`_to_batch` / `compute_loss` 的目标字段（rank/winner/econ）与头输出对齐；`/4.0` 还原（0.25 步长约定）。
5. 保持所有动作特征值为 0.25 的倍数且 ≤63.75（uint8 回放压缩不变量，违规会运行时报错）。
6. 相关测试：`python/tests/test_engine.py`（shape/schema 回归）、`test_hierarchical_policy.py`（schema 门禁、压缩无损、teacher 对齐）、`engine/src/bridge` 单元测试与 `engine/tests/engine_tests.rs`。

## 附录 A：评审意见处置索引

`review-encoding.md` 逐条处置（"接受/部分接受/不成立"均指本轮核实结论）：

| 评审条目 | 处置 | 去向 |
| --- | --- | --- |
| ① pointwise 表达力上限 | 接受，记录为选型权衡 | §1、§7.4 |
| ② 碰撞→表达力→prior→样本效率影响链 | 接受（表述修正） | §8.2 |
| ③ SUMMARY 语义过度压缩 | 接受（P2 记录） | §7.5 |
| ④ Scout 是未来状态操作 | 接受（P2 记录） | §8.5 |
| ⑤ uint8 隐含上界 63.75 | 接受（已补文档） | §4 |
| ⑥ value 逐局 z-score 与 backup 不一致 | **部分不成立**：backup 与训练目标同公式同尺度，局内一致；跨局保序不保距记录为权衡 | §5.2 |
| ⑦ econ head 双时间定义 | 接受（记录） | §5.3 |
| ⑧ 负收入 ×3 强偏置 | 接受（记录；ablation 待做） | §5.3 |
| ⑨ type head 是 policy 边缘、无新信息 | 接受（已明确写出） | §5.4 |
| ⑩ 候选生成器 self-confirming 闭环 | 接受（有实测支撑） | §7.1、§7.3 |
| ⑪ imitation(≤14) vs MCTS(k=4) 第二层错配 | **前提不成立**：两路径共用 `candidate_actions_k(4)`，无第二层错配 | §6 |
| ⑫ canonical 身份 ≠ 表示 | 已是现状（补记） | §1 |
| ⑬⑮ 动作后果未显式编码 / 信息三分类 | 接受为 schema v4 方向，本轮不实施 | §8.2 修复方向 |
| ⑭ trunk 可恢复性检查 | 已执行：发现 merchant `has_beer` 未编码 | §8.3 |
| ⑯ 碰撞分类矩阵（执行等价/值等价/值不同） | 接受，已按后继状态分类并测 regret | §7.2 |
| ⑰ policy/value regret 指标 | 部分执行：value regret 已测；policy regret 并入 recall 实验 | §7.2、§7.4 |
| ⑱ P0/P1/P2 优先级划分 | 采纳 | §7.4 |

本轮评审驱动实验的附带产出：修复了一个真实引擎 bug——`GameState::from_snapshot_bytes` 恢复状态时网络掩码缓存未重建（`engine/src/game_state/state.rs`，docstring 声称 "rebuilt on restore" 但实现缺失），导致恢复态在绕过枚举路径直接走 `apply_move` 校验时可能误拒合法动作；已补 `rebuild_network_masks()` 并新增回归测试 `snapshot_restore_rebuilds_network_masks`。
