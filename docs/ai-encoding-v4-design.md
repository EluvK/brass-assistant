# 编码与网络头 v4 迭代设计（评审驱动，破坏性升级）

> 状态：已批准（2026-08-29），按 §5 顺序实施。前提：项目起步期，**不做兼容**——旧 checkpoint 全部作废（本就已被 v3 门禁拒绝），旧回放作废，`ACTION_FEATURE_SCHEMA_VERSION` 与 `STATE_FEATURE_SCHEMA_VERSION` 直接 bump 到 4。
> 依据：`ai-action-encoding.md` §7 的实测（生成器 3.5% 可见率、来源身份由生成器代选、NetDouble 来源碰撞 34% 组 >4VP、32% 被排除 Develop 备选反超 teacher、merchant has_beer 缺失）。

## 0. 总原则

1. **三个改动必须捆绑落地**：动作编码 v4（能表达来源身份）、候选生成器 v4（把来源备选暴露成候选）、网络头 v4（集合感知 + rank 目标）。只改编码不改生成器，网络依然看不到备选（实测 0/372 状态的 shortlist 里有同组来源变体）；只改生成器不改编码，同特征动作在网络眼里仍是同一个动作。
2. **编码玩家真实的选择自由度**（来源身份），**显式编码解析可得的后果**（翻谁的板块），**派生信息留给网络推断**（市场档位等无选择空间的信息）。
3. **动作特征只编码"选择"，"身份"交给状态张量**：动作引用棋盘格索引与数量，地点的行业能力/归属/翻面状态全部由 state 张量在同名索引上提供，网络在打分头做 join。不在动作特征里重复状态已有的信息。
4. 保持既有不变量：所有特征值为 0.25 倍数且 ≤63.75（uint8 回放压缩，见 `ai-action-encoding.md` §4 的澄清：该上限约束的是归一化后的特征值，不是游戏原始数值）。

## 1. 动作特征编码 v4（235 → 301 维）

| 块 | offset | 宽度 | 语义（相对 v3 的变化） |
| --- | --- | --- | --- |
| ACTION | 0 | 7 | 不变 |
| CARD | 7 | 35 | **改为语义计数**：单牌动作 = 1.0（与原 one-hot 等价）；Scout 三张弃牌按语义**累加计数**，值 = 同语义牌数（1.0 / 2.0 / 3.0，整数即 0.25 倍数）。修复 multiset→set 坍缩（{A,A,B} ≠ {A,B,B}） |
| LOCATION | 42 | 27 | 不变（Build 目标） |
| CITY_SLOT | 69 | 4 | 不变 |
| INDUSTRY_1/2 | 73/79 | 6/6 | 不变 |
| CONNECTION_1/2 | 85/124 | 39/39 | 不变 |
| SELL_KEY | 163 | 47 | 不变 |
| MERCHANT | 210 | 9 | 不变（Sell 目标商家） |
| **DRAIN** | 219 | 49 | **新增，核心**：按**棋盘格索引**（47 城市槽 + 2 农场，与 board 张量同索引系）编码该动作抽取的资源量（cubes /4）。煤/铁/酒厂啤酒共用一个块——一格上只有一种行业，**种类与归属不用编码**：打分头把 DRAIN 引用与 board 平面（owner one-hot、industry、cubes、flipped）做 join 即得"抽的是谁家的什么"。市场购买无身份，不进本块 |
| **MERCHANT_BEER** | 268 | 9 | **新增**：从各商家抽走的啤酒量 /4（商家不占行业槽，不在 49 格索引系内，单列） |
| **CONSEQUENCE** | 277 | 12 | **新增**：解析后果特征（见下） |
| **SUMMARY** | 289 | 12 | 瘦身为纯资源总量，逐项：[0]煤总量/2、[1]铁总量/2、[2]啤酒总量、[3]市场煤数、[4]市场铁数、[5]商家啤酒数、[6]待售板块数/4、[7]单铁路、[8]双铁路、[9]免费开发、[10]Scout、[11]保留 |

合计 **301 维**。所有取值 ∈ {0, 0.25, …, 63.75} 满足 uint8 不变量（本布局实际最大值 ≤3）。

### 1.1 三套索引系：商家与市场的编码边界

地图上有三个"位置"体系和两个供应来源，编码前必须分清（商家确实位于 27 地点系内，但**不占行业槽**）：

| 对象 | 索引系 | 内容 | 编码去向 |
| --- | --- | --- | --- |
| 城市槽 + 农场 | 47+2 = **49 棋盘格**（board 张量同索引） | 行业板块的建造/翻面/资源 | 动作：DRAIN、CITY_SLOT、SELL_KEY；状态：board (24,49) |
| **商家** | **9 个商家板块**，挂在 9 个商家**地点**上（地点在 27 系内） | 每局随机 `buys`（收什么货）+ `has_beer`（商家酒） | 动作：MERCHANT 9（Sell 目标）、MERCHANT_BEER 9（抽商家酒）；状态：global 商家块（见 §3） |
| **市场** | **虚拟**，无地点无身份 | 只有煤/铁供应（价格随存量浮动），**没有啤酒** | 状态：global 煤/铁市场 one-hot（15/11）；动作：SUMMARY 市场煤/铁计数。永不进 DRAIN |

Sell 打分的 join 由此完备：动作的 `MERCHANT[i]` one-hot × 状态里商家 i 的 `buys`/`has_beer` = "这个商家收不收我的货、有没有酒可喝"。"从商家抽酒"进 MERCHANT_BEER，"从酒厂抽酒"进 DRAIN——啤酒的三个来源（自家/对手酒厂 → 49 格；商家 → 9 板块）索引系不再混用。

CONSEQUENCE 12 项（全部由 state + ResolvedMove 解析计算，**不做模拟**）：[0]links_built /2、[1]new_reach_own（建链后行动者网络新触及地点数 /4，保持 0.25 步长）、[2]new_merchant_reach（0/1）、[3]flips_own /4、[4]flips_opp /4（被抽空翻面的板块数，按归属分计）、[5]is_overbuild、[6]is_upgrade、[7]city_completion（此建使该地点满槽）、[8]sell_tiles /4、[9]free_develop、[10]/[11]保留。

v3 → v4 明确**删除**的东西：SUMMARY[2]/[6..8] 的跨类型语义复用（拆进 DRAIN 与 SUMMARY 固定语义）、CARD 的按位 OR 叠加、SUMMARY[14]/[15] 空置位。

## 2. 候选生成器 v4（heuristic_ai::candidate_actions_k 配套改造）

- Build / Network / NetDouble 的 top-k 语义从"k 个动作"改为 **"k 个几何体 × 每几何体 ≤m 个来源变体"**（建议 k=4、m=2，shortlist 上限 3×8+5 ≈ 29）：同一连接/建造位下，煤/啤酒来源身份不同的两个变体**同时**进入候选集。
- `operation_key` 已把来源身份算进操作身份，dedup 无需改。
- Develop/Sell/Loan/Scout/Pass：scorer 改为可产出多计划（首版至少 Sell 产出前 2 个商家路线、Develop 产出前 2 个移除计划）。
- teacher 分数对来源变体接近同分是**预期行为**：softmax prior 均分、由 MCTS 的 value 分辨——这正是让搜索接管来源决策的机制。
- MCTS `candidate_k` 语义随之变为"几何体数"，默认仍 4（等效候选 ~29）；self-play 实验臂保留 `candidate_k=0`（全合法）用于 recall 测量。

## 3. 状态张量 v4（STATE_SCHEMA_VERSION 3 → 4）

动作特征改为"引用棋盘格"之后，state 张量必须提供 join 所需的全部背景。当前 trunk 的读法（`net.py:39-41,98-100`）是 per-cell 线性编码 17 个平面 + scatter 到 27 地点节点 + 地点位置 embedding——**空槽位不携带任何"这里能造什么"的信息**（industry one-hot 平面只描述已占用的板块），商家状态也完全缺失。v4 补三件事：

1. **静态槽位能力平面**：board 增 6 个平面 = 该槽允许的行业 one-hot（来自地图数据，静态）——空槽也知道自己是不是煤位。这同时是样本效率的大补：网络不再需要靠位置 embedding 死记 47 个槽的能力表。
2. **槽位序号平面**：board 增 1 平面 = 槽位在地点内的序号 /4，配合 CITY_SLOT/SELL_KEY 引用做精确 join。
3. **商家块**（`MerchantTile { loc, buys, has_beer }` 的完整补齐）：global 增 **9 商家 × 5 收货类型 one-hot（45 维：Blank / Any / CottonMill / Manufacturer / Pottery）+ 9 维 has_beer**。收货类型只有 5 种是游戏设定——商家只收制成品货物，煤/铁/酒是原材料进不了商家（`map::merchant_tile_mix` 的每局发牌表只有这 5 种，按人数 5/7/9 张）。这是比 has_beer 更大的缺口：商家每局的收货内容是随机 setup 信息，当前状态张量完全看不见，Sell 路线价值评估无从谈起。has_beer 仍需单列：setup 时 `has_beer = buys != Blank`，但被喝掉后翻转，是动态位。

即 board (17,49) → **(24,49)**，global 114 → **168**，其余布局不动。

## 4. 网络头 v4（net.py / train.py 重写）

### 4.1 Policy：FiLM 交互 + 集合上下文（仍是 O(N)，不是 O(N²) attention）

```text
a      = ActionMLP(action_features)          # 301 -> 128
γ, β   = FiLM_MLP(state_emb)                 # 状态调制动作表征
a'     = γ ⊙ a + β
ctx    = masked_mean(a' over candidate set)  # 集合上下文（O(N) 池化）
logit  = ScoreMLP([a', ctx, a' ⊙ ctx])       # 逐候选标量
P      = masked log_softmax(logits)
```

- 解决评审 ① 的核心抱怨（score 不依赖候选集合）的**廉价 80%**：`ctx` 让网络能表达"集合里还有别的动作在竞争同一资源/位置"，代价只是一次 mean pooling。满合法集 2655 候选下依然便宜。
- FiLM（乘性交互）替代纯拼接，动作-状态耦合从"MLP 自己学"变成结构保证。

### 4.2 Value：终局排名 + 胜者分布（替代逐局 VP z-score）

- **rank head**：`Linear(trunk → 4)`，目标 = 每座位终局名次 / n ∈ (0,1]（并列取平均名次），MSE。**跨局可比、局内保序**——两个目标一次满足，直接替换 z-score（z 跨局只保序不保距是已记录的权衡，v4 直接移除该头）。
- **winner head**：`Linear(trunk → 4)`，softmax + CE（并列时目标为并列者的均匀分布）。对齐 `ai-advise.md` "policy 最终优化目标转向 win/rank"。
- MCTS backup：节点存 per-seat `score_p = 1 − rank_p`，MaxN 逻辑不变（行动方最大化自己的分量）；终局 backup 用 `1 − rank(vps)`，与网络输出同尺度。
- 训练损失：`policy_CE + rank_MSE + 0.5·winner_CE + 0.1·econ_MSE + 1e-4·L2`。

### 4.3 删除与拆分

- **删除 type 辅助头**：它是 policy 目标的边缘（无新信息），且删除后 `train.py` 对 `candidates[..., :7]` 布局耦合随之消失。
- **econ head 拆成 canal/rail 两个头**（各 2 维），样本按所处时代走对应头——消除"同一 head 两种目标时间定义"的混杂；负收入加权 3× 降为 1×，留 config 开关供 ablation。

### 4.4 训练管线

- 候选特征在回放中按既有 uint8 压缩存储（301/4 ≈ 76 B/行，比 float32 省 4×），`max_candidate_batch` 65536 → 32768。
- bootstrap 主路径切到 **full-legal imitation**：来源身份进编码后，one-hot 目标的同特征冲突消失（原 5/372 状态、CE 下界 log 2 的问题不复存在）；shortlist 模式保留作对照。
- self-play 用生成器 v4（`candidate_k=4`，等效 ~29 候选），训练稳定后按 roadmap 逐步以 visit 分布替换 teacher。

## 5. 实施顺序与验收

1. **动作编码 v4**（Rust `action_features.rs` + Python 常量 + bump）：单测覆盖每类动作的亮位示例更新、DRAIN 矩阵与手推来源一致、uint8 往返无损。
2. **生成器 v4**：单测断言"同几何体来源变体成对出现"；重跑 §7 实验，预期 shortlist 内同组变体 > 0。
3. **状态 v4 + 网络头 v4**（net.py/train.py/selfplay.py）：冒烟训练 + 验证 rank/winner 目标分布、policy 损失收敛。
4. **重测三指标**：collision 的 policy 目标冲突数（预期 0）、candidate recall（有 checkpoint 后）、天花板探针复测。
5. 里程碑判据：full-legal bootstrap 后 teacher top-1 recall（全合法集上前 1 动作进入 shortlist 的比例）与探针胜率显著下降，即生成器偏差被策略接管。

## 6. 本轮明确不做

- 连接端点 one-hot（CONN_ENDPOINTS）：信息上是连接序号的确定函数，仅为动作编码器对 Network 类动作的地理推理样本效率服务；若错误案例显示 Network 选址系统性弱，再以 54 维（仅单铁路）或 108 维回归。
- trunk 结构重写（token/图注意力化）——等 v4 落地后的错误案例再动，是下一前沿。
- 候选间 self-attention（O(N²)）——`ctx` 池化不够时再上。
- PPO/actor-critic——roadmap 第 6 位的既有结论不变。
