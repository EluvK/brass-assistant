# 02 · 状态和动作怎么变成数字

> **本章目标**：神经网络只认数字（张量）。本章讲清楚"牌面局面"和"候选动作"分别是哪张张量、每个数是什么含义、有哪些刻意的编码设计。
>
> 主读代码/文档：`docs/ai-action-encoding.md`（动作 301 维的**权威规范**，本章不重复它的全部细节）、Rust 侧 `engine/src/bridge/encode.rs`（状态张量）、`engine/src/bridge/action_features.rs`（动作特征）。
> Python 侧只有校验与压缩逻辑：`python/brass_ai/hierarchical_policy.py`。

---

## 2.1 两个编码问题

把"一局牌"喂给网络之前，要回答两个问题：

1. **状态编码**：当前局面 → 一组张量（网络输入五件套）；
2. **动作编码**：每个候选动作 → 一行 301 维特征（网络逐个打分的对象）。

这两套编码都由 **Rust 引擎**负责（Python 只消费），并且都有**schema 版本号**（当前都是 4）。版本号跟着 checkpoint 一起存（04 章），加载时强校验——编码改一个字节，旧 checkpoint 立刻拒绝加载，避免"用错特征语义的旧权重"这种无声的灾难。

所有相关常量可以在 Python 里实测（`python/brass_ai/_engine.pyi` 声明，编译产物 `_engine` 提供）：

| 常量 | 值 | 含义 |
| --- | --- | --- |
| `BOARD_PLANES × BOARD_CELLS` | 24 × 49 | 棋盘：49 格 × 每格 24 个平面 |
| `LINK_PLANES × LINK_CELLS` | 7 × 39 | 铁路连接：39 条 × 每条 7 个平面 |
| `GLOBAL_LEN` | 168 | 全局信息（时代、市场、商家、分数等） |
| `HAND_LEN` | 35 | 每人手牌编码长度（0–26 位置牌、27–32 行业牌、33/34 万能牌） |
| `LOCATION_COUNT` | 27 | 地点数（城市 + 农场，图网络的节点数） |
| `ACTION_FEATURE_DIM` | 301 | 每个候选动作的特征维度 |
| `ACTION_FEATURE_SCALE` | 4 | uint8 压缩倍率（0.25 步长 ×4） |

---

## 2.2 状态五件套：`state_to_tensor()`

Rust 的 `GameState.state_to_tensor()` 一次返回五个数组（`selfplay.py` 里接收为 `board, links, g, oh, op`）。把它们想象成"给网络的读盘记录"：

### board `(24, 49)`：棋盘

49 个"格"= 47 个城市槽位 + 2 个农场，每格用 24 个平面描述（哪些平面对应什么，见 `engine/src/bridge/encode.rs` 的模块注释）。**"平面"（plane）直接类比图像的 RGB 通道**：图像是 3×H×W（3 个通道叠在网格上），这里是 24×49——24 个二值/计数通道叠在棋盘格上，比如"这格有没有板块""是谁家的""什么行业""翻面没有""剩几个资源"各自占一个通道。

这样编码的好处是**位置对齐**：第 18 格的煤量就在 `board[·, 18]` 的资源通道上，后面动作编码里的 DRAIN 块也用同一个 49 格索引——网络在打分头里把"动作抽了第 18 格"和"第 18 格的状态"拼在一起理解（这个 join 是 301 维布局的核心思想，见 2.3）。

### links `(7, 39)`：铁路连接

39 条可建连接，每条 7 个平面（是否已建、谁建的、运河还是铁路等）。棋盘上两地的"连通性"——Brass 的核心机制——就编码在这里。

### global `(168,)`：全局信息

一维拼接向量：当前时代、轮次、四个玩家的公开状态（钱、收入等级、分数）、煤/铁市场的档位、9 个商家的收货与啤酒、待售板块等。凡是"不属于某个棋盘格"的公共信息都进这里。

### own_hand `(35,)` + opp_hands `(105,)`

己方手牌是 35 维计数向量（每种牌有几个）；三个对手的手牌拼成 `(105,)`。**对手手牌按理说是隐藏信息**——本项目训练/采样时引擎以"上帝视角"编码（self-play 里四个座位都是自己人，数据里没有真正的秘密），实战搜索时靠 ISMCTS 的确定性重采样处理信息隐藏（05 章）。

> 顺带解释一个背景：**为什么状态由 Rust 编码而 Python 不重写一遍？** 单一事实源。Rust 引擎是规则与状态的唯一持有者，Python 只做特征格式校验（`hierarchical_policy._feature_width()` 检查 schema 版本），从源头上杜绝两套实现不一致。

---

## 2.3 动作编码：每个候选一行 301 维

本项目**没有固定长度的动作索引空间**（"第 837 号动作"这种）。动作 = 卡牌 × 位置 × 资源路径的组合爆炸，全合法集可达数千。所以网络不做"输出几千维、挑最大的"的 flat policy，而是**逐候选打分**：引擎枚举每个具体合法动作，编码成 301 维特征行，网络给每行打一个分。

301 维分 14 个块。完整表格见 [ai-action-encoding.md §2](../ai-action-encoding.md)，这里给出便于建立直觉的精简版：

| 块 | 宽度 | 直觉理解 |
| --- | --- | --- |
| `ACTION` | 7 | 动作类型 one-hot（Build/Network/Develop/Sell/Loan/Scout/Pass） |
| `CARD` | 35 | 打出的牌的**语义计数**（不指明手牌第几张） |
| `LOCATION` / `CITY_SLOT` | 27 / 4 | 建在哪：地点 + 城市槽位 |
| `INDUSTRY_1` / `INDUSTRY_2` | 6 / 6 | 行业 one-hot（建造/移除的行业） |
| `CONNECTION_1` / `CONNECTION_2` | 39 / 39 | 建哪条连接（双铁路两条） |
| `SELL_KEY` / `MERCHANT` | 47 / 9 | 卖给哪个商家、卖哪些槽位的货 |
| `DRAIN` | 49 | **按棋盘格索引**：从哪个格抽了几个资源（×1/4 计数） |
| `MERCHANT_BEER` | 9 | 从各商家抽走的啤酒 |
| `CONSEQUENCE` | 12 | 解析后果（建几条链、新触及哪些地点/商家、翻面数等） |
| `SUMMARY` | 12 | 资源汇总（用了几煤几铁几啤酒、各来自哪里） |

用一个真实例子建立体感（运河时代的 Build，摘自 [ai-action-encoding.md §3.1](../ai-action-encoding.md)）：

```text
Build（Derby 槽 1，制造厂）的非零位:
ACTION[0]=1  CARD[27]=1  CARD[30]=1     # 类型 = Build;打出棉纺厂+制造厂双图标牌
LOCATION[1]=1  CITY_SLOT[1]=1  INDUSTRY_1[3]=1   # 建在 Derby 1 号槽,制造厂
SUMMARY[0]=0.5   # 用了 1 个煤(计数 ÷2 → 0.5)
SUMMARY[3]=1     # 这 1 个煤来自市场(市场无身份,只记个数)
```

### 三条贯穿性的设计思想

**① 只编码"选择"，不编码"派生事实"。** 成本、归属、市场档位这些可以由状态推导的信息不进动作特征。例：`DRAIN[18]=0.25` 只说"从第 18 格抽了 1 煤"，抽的是谁家的矿、当时市价多少，由状态张量 `board[·,18]` 的归属平面 + global 的市场平面提供，网络在打分时自己 join。这让特征向量只承载"决策自由度"，信息不重复。

**② 网络从不学习合法性。** 候选集由引擎构造性生成（每个都是可执行动作），网络只在合法候选上打分归一化（04 章的 masked softmax）。合法性检查是规则引擎的活，不是统计学习的活。

**③ 身份与表示分离：canonical 字符串。** 每个动作另有一个 canonical 字符串（Rust `move_codec` 生成，如动作的"身份证号"），用于跨 Rust/Python 边界**对齐**——把 MCTS 的 visit 计数、老师的选择对回到候选行上（01 章 `_candidate_policy`）。它**从不进入网络输入**：网络只看 301 维特征，字符串只做"A 是不是 B"的恒等判断。

---

## 2.4 uint8 压缩：0.25 步长不变量

所有 301 维特征值都是 **0.25 的整数倍**（one-hot 是 1.0，"1 个煤 ÷2" 是 0.5，"1 个板块 ÷4" 是 0.25……）。这意味着特征可以无损存成 uint8：`值 × 4` 必为 0–255 的整数。

```python
# hierarchical_policy.py
scaled = array * ACTION_FEATURE_SCALE        # ×4
if not np.allclose(scaled, np.rint(scaled)): # 不是 0.25 步长 → 直接报错
    raise ValueError(...)
return np.rint(scaled).astype(np.uint8)
```

使用场景与还原位置：

- **跨进程传输 / 磁盘存储**（01 章 shard 物化、04 章训练管线）：uint8 传输量是 float32 的 1/4；
- **GPU 上还原**：`net.forward` 开头遇到 uint8 输入就地 `float().div_(4)`——省的是昂贵的内存→显存拷贝，还原发生在显存里，几乎免费；
- CPU 路径由 `train.py._to_batch` 兜底做同样的 `/4`。

两条容易误解的边界（[ai-action-encoding.md §4](../ai-action-encoding.md) 有更详细的澄清）：这个约束**只作用于候选动作特征**，状态张量始终是 float32；63.75（=255/4）是"归一化后特征值"的上限，不是游戏数值上限——原始数值都先除以设计常数再进特征。

---

## 2.5 图拓扑常量：给网络用的"地图"

02 章最后埋一个 03 章的钩子。网络要理解"棋盘是一张图"（27 个地点、39 条连接、部分连接绕行农场），但**图的结构是常量**——每局都一样。引擎把它作为常量表导出：

```python
# net.py: 注册为 buffer(随模型保存、随 .to(device) 搬运,但不是可学习参数)
self.register_buffer("cell_locations", cell_locations)      # 49 格 → 属于哪个地点(27)
self.register_buffer("edge_endpoints", endpoints)          # 39 条连接 → 两端地点
self.register_buffer("edge_via_farms", via_farms)          # 连接是否绕行农场
```

03 章的网络会拿这三张表做**图上的消息传递**（message passing）——信息沿着"连接"在"地点"之间流动。

## 练习

1. 实测常量：
   ```python
   import sys; sys.path.insert(0, "python")
   from brass_ai import _engine as be
   print(be.BOARD_PLANES, be.BOARD_CELLS, be.GLOBAL_LEN, be.HAND_LEN, be.ACTION_FEATURE_DIM)
   ```
2. 生成 2 局 shortlist 模仿数据（01 章练习的代码），挑一个 Build 样本，打印 `candidates[0][42:69]`（LOCATION 块）的 argmax——它应该指向老师建在的地点。
3. 对照 [ai-action-encoding.md §2](../ai-action-encoding.md) 的 14 块表，验证 7+35+27+4+6+6+39+39+47+9+49+9+12+12 = 301。
