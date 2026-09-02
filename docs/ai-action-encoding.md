# 动作特征编码与网络头设计

本文描述当前 candidate-scoring 方案中**动作特征（301 维，schema v5）的具体布局**与 **Python 网络各输出头（policy / rank / winner / econ）的设计**，并给出每类动作的实测编码示例。

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
- **集合上下文**：每个候选的打分输入包含候选集的 masked-mean 嵌入，可表达"集合里还有别的动作在竞争同一资源/位置"（O(N)，非 self-attention）。

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
| `MERCHANT` | 210 | 9 | 保留区；v5 的 Sell 不编码普通买家 |
| `DRAIN` | 219 | 49 | **按棋盘格索引**（47 城市槽 + 2 农场，与 board 张量同索引）该动作抽取的资源量（cubes/4）。煤/铁/酒厂啤酒共用一个块：一格只有一种行业，**种类与归属由状态平面 join 提供**。市场购买无身份，不进本块 |
| `MERCHANT_BEER` | 268 | 9 | 从各商家抽走的啤酒量（cubes/4） |
| `CONSEQUENCE` | 277 | 12 | 解析后果：[0]建链数/2、[1]网络新触及地点/4、[2]新触及商家 0/1、[3]/[4]翻面数 自家/对手（各 /4）、[5]overbuild、[6]升级、[7]建满城市、[8]待售板块/4、[9]免费开发、[10]/[11]保留 |
| `SUMMARY` | 289 | 12 | [0]煤总量/2、[1]铁总量/2、[2]啤酒总量、[3]市场煤数、[4]市场铁数、[5]商家啤酒数、[6]待售板块/4、[7]单铁路、[8]双铁路、[9]免费开发、[10]Scout、[11]保留 |

三套索引系的边界——地图上有三个"位置"体系和两个供应来源（商家位于 27 地点系内，但**不占行业槽**）：

| 对象 | 索引系 | 内容 | 编码去向 |
| --- | --- | --- | --- |
| 城市槽 + 农场 | 47+2 = **49 棋盘格**（board 张量同索引） | 行业板块的建造/翻面/资源 | 动作：DRAIN、CITY_SLOT、SELL_KEY；状态：board (24,49) |
| **商家** | **9 个商家板块**，挂在 9 个商家**地点**上 | 每局随机 `buys`（收什么货）+ `has_beer`（商家酒） | 动作：MERCHANT_BEER（抽商家酒）；状态：global 商家块 |
| **市场** | **虚拟**，无地点无身份 | 只有煤/铁供应（价格随存量浮动），**没有啤酒** | 状态：global 煤/铁市场 one-hot；动作：SUMMARY 市场煤/铁计数。永不进 DRAIN |

Sell 的普通买家仅是合法性约束，不是动作身份；商家酒的身份由 `MERCHANT_BEER` 唯一表达。啤酒的三个来源索引系不混用：自家/对手酒厂 → DRAIN（49 格），商家 → MERCHANT_BEER（9 板块），市场无啤酒。

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
ACTION[3]=1  CARD[28]=1  SELL_KEY[4]=1
CONSEQUENCE[8]=0.25   # 1 张待售板块（/4）
SUMMARY[6]=0.25       # 1 / 4
```

该例板块不需要啤酒。若需要，所选择的具体酒桶显式进 DRAIN / MERCHANT_BEER——不是网络自行推断的隐式信息。

### 3.6 Loan / Scout / Pass

```text
Loan:  ACTION[4]=1; CARD[29]=1
Scout: 三张弃牌按语义累加计数（如两张同位置牌 → CARD[loc]=2），SUMMARY[10]=1
Pass:  ACTION[6]=1; CARD[..]=1
```

Scout 的计数式 CARD 使 {A,A,B} 与 {A,B,B} 可分辨。

## 4. 数值约定与 uint8 压缩不变量

**所有 301 维特征值都是 0.25 的整数倍**（0 / 0.25 / 0.5 / 1 / 1.5 / 2…）。这是候选行 uint8 无损压缩的前提：`hierarchical_policy.compress_candidate_features` 按 `×4` 打包，遇到非 0.25 步长的值直接报错（`hierarchical_policy.py:60-70`）；uint8 行按 `/4` 还原发生在 GPU 上（`net.py` 的 `forward`，H2D 拷贝因此只有 float32 的 1/4；CPU 路径由 `train.py` 的 `_to_batch` 兜底）。

澄清两条容易误解的边界：

- **作用范围只有"候选动作特征"这一种张量**。状态张量（board/links/global/手牌）是 float32，不走 uint8 打包，不受此约束。
- **63.75 是"归一化后的特征值"的上限，不是游戏数值的上限**。游戏里的原始数量（钱 £200、VP 120、8 个煤）从不直接写进特征——每一维都是除以设计常数后的归一化值（钱 /200、煤量 /2、计数 /4、one-hot 1.0）。"8 个煤"编码为 8/2 = 4.0，×4 = 16，离 255 很远。新增特征时要做的是选一个足够大的归一化常数，使**归一化值 ≤ 63.75**（实践中 ≤3 就够）；0.25 步长对这些离散计数/one-hot 特征是无损的。

即新增特征必须同时满足两条约束：`值 × 4` 是整数（0.25 步长），且 `0 ≤ 值 × 4 ≤ 255`。

## 5. 网络头设计（net.py / train.py）

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

### 5.2 Rank head + Winner head

- **rank head**：`Linear(trunk -> 4)`，目标 = 每座位终局名次 / n，MSE。终局名次由 `scoring.rs::final_ranking` 按 VP → 收入等级 → 现金**确定性破平局**（无并列）。**跨局可比、局内保序**。
- **winner head**：`Linear(trunk -> 4)`，softmax + CE，目标 = 唯一冠军（破平局后第一名）的 one-hot。
- 搜索尺度：网络叶子值与终局 backup（`nn_mcts.rs terminal_value`）都用 `score_p = 1 − rank_p/n`，MaxN 按行动方取自己的分量最大化。
- 设计权衡：z-score 类逐局价值跨局只保序不保距、margin 信息丢失；rank/winner 直接对齐"第一名概率"这一最终目标。

### 5.3 Econ head（经济辅助头，按时代拆分）

- 结构：**canal 与 rail 两个独立头**（各 `Linear(trunk -> 2)`），输出拼接为 (B,4)；样本按所处时代只训练自己时代的头。
- 目标：canal 样本盖 canal-end 经济、rail 样本盖终局经济；归一化 income `(x+10)/40`、money `/100`。
- 负收入加权 `econ_neg_weight` 默认 **1.0（关闭）**，保留开关供 ablation。
- 总权重 `econ_lambda = 0.2`。

### 5.4 损失汇总与默认超参

```text
L = policy_CE + rank_MSE + 0.5 * winner_CE + 0.2 * econ_MSE(era-split) + 1e-4 * ||θ||²
```

网络没有动作类型辅助头，`train.py` 不依赖任何动作块布局约定。

| 超参 | 默认值 | 说明 |
| --- | --- | --- |
| `epochs` / `batch_size` | 5 / 256 | 每次调用训练的遍数与批大小 |
| `lr` | 1e-3 | AdamW；`weight_decay=0`（L2 单独算） |
| `l2` | 1e-4 | 显式 L2 |
| 调度 | CosineAnnealingLR | `T_max=100`，`min_lr=1e-5`，跨迭代保持 optimizer 状态 |
| `econ_lambda` / `econ_neg_weight` | 0.2 / 1.0 | 经济辅助权重 / 负收入加权（默认关闭，留 ablation 开关） |
| `max_candidate_batch` | 65536 | 单个训练 micro-batch 的候选行预算；样本按候选数排序后贪心装箱（一个超大候选样本只影响其所在块） |
| `materialize_rpc_chunk` | 32 | 每个跨进程物化任务携带的样本数（候选行由 Rust `materialize_snapshot` 直接产出 uint8 quarter-step，省 4× IPC 与 H2D） |
| `finiteness_check_interval` | 100 | CUDA 上逐参数 inf/NaN 深检的步数间隔（GradScaler 已跳过坏步；CPU 上每步都查） |
| `amp` | True | GPU 上 fp16 autocast |

## 6. 训练目标来源

| 模式 | policy 目标 | 候选集 |
| --- | --- | --- |
| MCTS self-play | visit 分布按 canonical 字符串对齐到当前候选（`selfplay._candidate_policy`） | 搜索所用候选 |
| Imitation（shortlist） | teacher 分数 softmax（温度 1.0） | 生成器 v4：Build/Network/NetDouble 各 ≤4 几何体 × ≤2 来源变体，Develop/Sell 各 ≤2 计划，Loan/Scout/Pass 各 ≤1（**实测均值 ~12.4、上限 22**） |
| Imitation（full-legal，bootstrap 默认） | teacher 动作 one-hot；样本只存 snapshot + teacher canonical，训练前由 Rust `materialize_snapshot` 单次调用实时物化候选（uint8 候选 + teacher 等价类 policy，一次 PyO3 调用完成，不再逐元素跨界） | 完整合法集 |

候选集事实（`heuristic_ai::candidate_actions_k` 是**唯一**的候选生成器）：

- NN-MCTS 默认 `candidate_k=0`，直接展开完整合法集；传入正数时才启用 `candidate_actions_k` 的 heuristic shortlist（用于受控性能实验）。imitation 训练统一使用 full-legal，`heuristic_candidates` 仅负责取得 teacher action。plain MCTS（`mcts_ai.rs`）仍使用同一生成器（默认 k=3）。
- 生成器 v4（`SOURCE_VARIANTS=2`）：同一几何体（连接/建造位）下来源身份不同的变体**成对进入候选集**，"抽谁的矿/酒馆"由搜索而非生成器决定。
- 整个管线没有默认越过生成器的路径；NN-MCTS 显式传 `candidate_k=0` 可全合法集展开。
- **checkpoint 现状**：checkpoint 属本地不入库资产（`checkpoints/` 已被 gitignore），文档不引用具体训练产物文件名。首个 v4 full-legal imitation 训练 run 已产出样本分片（`<ckpt>.imitation/imitation-*.pkl`），正式训练尚在早期；candidate recall 等指标待训练充分后测量（§7）。

## 7. 编码碰撞与候选生成器现状（v4 复测，2026-08-29）

把系统抽象为 状态编码器 → 动作编码器 → pointwise scorer → 候选 softmax → MCTS，风险自上而下分三层：

```text
1. Candidate Generator  是否漏掉好动作？      ← 最大风险
2. Action Representation 是否把关键动作折叠？
3. Pointwise Scorer     是否需要候选间交互？
```

**测量口径**：3 局启发式自对弈（seed 7/21/99，372 个决策状态）。v4 落地后的复测结论：

- **来源身份碰撞清零**：Network / Build / NetDouble 的"抽谁的矿/酒馆"变体在 v4 编码 + 生成器来源变体下全部可分辨（v3 时代分别有 320 / 3 / 1096 个碰撞组，是当时最大的表达力折叠）。
- 非"仅卡牌序"的碰撞组剩 **1766**（v3 时代 3054），且全部为执行等价类：Scout 同语义不同牌序变体 1573（弃牌多重集已由计数式 CARD 保留）、Sell 商家顺序变体 88、NetDouble 两条链接交换煤矿的等价变体 105。
- 即：编码器对"执行不同但价值可能不同"的动作**不再有已知折叠**；剩余碰撞全部执行等价，不构成表达力损失。
- **候选集**：shortlist 均值 ~12.4、范围 3–22（k=4 几何体 × ≤2 来源变体的语义）；shortlist 内同几何体的来源变体成对出现（单次实测直接观测 1 例；变体生成覆盖由 Rust 测试跨 20 seed 验证）。
- 候选生成器仍是最大风险：搜索只看到全合法集的一小部分，且 teacher 覆盖是构造性的（teacher canonical 恒在 shortlist 内），生成器上限必须用更强参照度量。

三个监控指标的状态：

| 指标 | 状态 |
| --- | --- |
| candidate recall（已训练策略在全合法集上的 top-1/top-3 落在 shortlist 内的比例） | 已定义；首个 v4 checkpoint 已产出（§6），待训练充分后测量 |
| collision value regret | v4 复测完成：剩余碰撞全部执行等价，不构成表达力损失 |
| pointwise vs set-aware scorer gap | 已部分解决：集合上下文（§5.1）以 O(N) 引入候选集依赖；完整 self-attention 待需要时投入 |

## 8. 有意不编码的信息

### 8.1 手牌选择（按设计折叠）

同一结构动作的卡牌选择按**打牌语义**折叠（不同手牌位置的相同语义牌得到同一向量）。这是"网络不学习手牌顺序"的刻意设计；同一语义牌选哪张执行结果完全相同，折叠无信息损失。

### 8.2 派生信息（不是选择，无需编码）

- **市场档位**：市场补足固定取最便宜档、按升序付款（`graph.rs`），玩家无选择空间；成本可由 SUMMARY 市场计数 + global 市场状态 one-hot 推断。
- **Sell 的酒桶选择**：扁平 `beer_sources` 是动作身份；引擎只在执行时确定其到板块的合法分配，结果显式编码进 DRAIN / MERCHANT_BEER。

### 8.3 迭代原则

后续升级先由错误案例证明当前 encoder 无法区分关键局面，再增量升级；升级必须 bump 对应 schema version；动作特征新增项必须满足 §4 的 0.25 步长与 63.75 上限。

## 9. 修改 schema 的维护清单

改动 301 维布局或状态张量时，以下位置必须同步（当前均有测试或运行时校验兜底）：

1. `engine/src/bridge/action_features.rs`：布局常量 + `encode_move` + bump `ACTION_FEATURE_SCHEMA_VERSION`。
2. `engine/src/bridge/encode.rs`：状态平面/长度 + bump `STATE_FEATURE_SCHEMA_VERSION`。
3. `python/brass_ai/hierarchical_policy.py`：`ACTION_FEATURE_SCHEMA_VERSION` / `ACTION_FEATURE_DIM` 常量（运行时强校验）。
4. `python/brass_ai/train.py`：`_to_batch` / `compute_loss` 的目标字段（rank/winner/econ）与头输出对齐；`/4.0` 还原（0.25 步长约定）。
5. 保持所有动作特征值为 0.25 的倍数且 ≤63.75（uint8 压缩不变量，违规会运行时报错）。
6. 相关测试：`python/tests/test_engine.py`（shape/schema 回归）、`test_hierarchical_policy.py`（schema 门禁、压缩无损、teacher 对齐）、`engine/src/bridge` 单元测试（action_features 块布局与 uint8 不变量、move_codec、encode、replay_fmt）与 `engine/tests/engine_tests.rs`。
