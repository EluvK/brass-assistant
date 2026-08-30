# 03 · 网络逐层拆解：`net.py`

> **本章目标**：把 `python/brass_ai/net.py`（201 行）从第一行读到最后一行，每一层都知道**它是什么、为什么这里需要它、进出的 tensor 形状是什么**。
> 前置：00 章（Linear/ReLU/Embedding/softmax）、02 章（状态五件套与 301 维动作特征）。
> 这是教程的核心章——其他章节的代码都可以"意会"，这一章值得逐行。

---

## 3.1 任务的形状合同

`PolicyValueNet` 对外只有两个入口。先把**输入输出形状**背下来，之后所有代码都在维护这个合同：

```text
输入:
  batch = {
    "board":     (B, 24, 49)   # 02 章的五件套,B = 批大小
    "links":     (B, 7, 39)
    "global":    (B, 168)
    "own_hand":  (B, 35)
    "opp_hands": (B, 105)
  }
  action_features: (B, N, 301)      # N = 每个局面的候选动作数,可变!
  candidate_mask:  (B, N) bool      # True = 真候选,False = padding 占位

输出 dict:
  "candidate_logits":     (B, N)    # 每个候选一个原始分(越大越好)
  "candidate_log_probs":  (B, N)    # masked log_softmax 之后的对数概率
  "rank":                 (B, 4)    # 四人终局名次预测(≈0.25~1.0)
  "value":                (B, 4)    # 1 − rank,搜索用的"越大越好"尺度
  "winner_logits":        (B, 4)    # 冠军预测的原始分
  "econ":                 (B, 4)    # [canal 收入, canal 现金, rail 收入, rail 现金]
```

N 是**可变的**（合法动作数随局面从几个到上千），这是本项目网络与教科书网络最大的不同：教科书网络输入输出形状全部固定，这里的核心工程就是"变长候选集怎么进固定形状的网络"。

> **关于 batch 维 B**：训练时 B 是 batch_size；推理（MCTS）时 B 是"攒起来一起问网络的面局数"。board/links 等五件套在最前面多一维 B，动作特征多两维 B 和 N。

---

## 3.2 全家福：forward 的数据流

```text
board (B,24,49)──transpose──board_enc──(B,49,128)──scatter 到 27 地点──┐
links (B,7,39) ──transpose──links_enc──(B,39,64)──+位置embedding───────┤
                                                                       ▼
                              3 轮图消息传递(边↔节点互传信息)  [3.4 节]
                                                                       │
              node (B,27,128) ──mean+max──(B,256) ─┐                   │
              edge (B,39,64)  ──mean+max──(B,128) ─┤                   │
global (168)+own(35)+opp(105)──────────────────────┴─concat─(B,692)    │
                                            │                          │
                                          trunk MLP                    │
                                            ▼                          │
                                    state 向量 (B,256) ◀──(图信息已汇入)─┘
                                            │
        ┌───────────────────────────────────┼──────────────────────────┐
        ▼ (FiLM 调制)                       ▼                          ▼
action_features (B,N,301)──action_encoder──(B,N,128)              四个输出头
        │                          │                             rank (B,4)
        ▼                          ▼                             winner (B,4)
γ,β = film(state) (B,128)   ctx = masked_mean(候选集)            econ (B,4)
        │                          │
        ▼                          ▼
   modulated = γ⊙a+β  ──与 ctx 拼接──▶ action_score ──▶ logits (B,N) [3.6 节]
```

三大块：**状态编码**（把局面压成 256 维向量，3.3–3.5）、**动作打分**（给 N 个候选各打一分，3.6）、**输出头**（3.7）。

---

## 3.3 入口：逐格投影 + 位置名片

```python
self.board_enc = nn.Sequential(nn.Linear(be.BOARD_PLANES, self.cfg.board_emb), nn.ReLU())
self.links_enc = nn.Sequential(nn.Linear(be.LINK_PLANES, self.cfg.links_emb), nn.ReLU())
self.node_position = nn.Embedding(be.LOCATION_COUNT, self.cfg.board_emb)   # 27 × 128
self.edge_position = nn.Embedding(be.LINK_CELLS, self.cfg.links_emb)       # 39 × 64
```

`encode_state` 的开头：

```python
board_cells = self.board_enc(batch["board"].transpose(1, 2))
node = self._scatter_mean(board_cells, self.cell_locations, be.LOCATION_COUNT)
node = node + self.node_position.weight.unsqueeze(0)
edge = self.links_enc(batch["links"].transpose(1, 2))
edge = edge + self.edge_position.weight.unsqueeze(0)
```

逐行翻译（记 `B` 为批大小）：

1. `batch["board"]` 形状 `(B, 24, 49)`（通道 × 格）。`transpose(1,2)` 换成 `(B, 49, 24)`——让"每一格"成为独立的样本行，好逐格过同一个线性层。`board_enc` 把每格的 24 个平面压成 128 维"格子摘要"，输出 `(B, 49, 128)`。
2. 49 个格其实属于 27 个地点（一个城市有多个槽位）。`_scatter_mean` 把同属一个地点的格子摘要**取平均**，聚成"地点摘要" `(B, 27, 128)`——这就是图网络的**节点**（node）。
3. `+ node_position.weight`：给每个地点加上一张可学习的"位置名片"（Embedding，27×128）。不加它，网络无法区分"Derby"和"Coalbrookdale"——散列平均之后的信息只描述"这是个有几格的地点"，名字本身要靠名片携带。`unsqueeze(0)` 把 `(27,128)` 变成 `(1,27,128)` 以广播相加到整个 batch。
4. 连接同理：39 条连接各自 7 平面 → 64 维，加 39×64 的位置名片，得到图网络的**边**（edge）表示 `(B, 39, 64)`。

> **parameter vs buffer**：`node_position` 是 `nn.Embedding`——**可学习参数**；下面马上出现的 `cell_locations` 等是 `register_buffer`——**常量表**（随模型保存、随 `.to(device)` 搬运，但训练不更新）。棋盘拓扑每局都一样，当然是 buffer。

### `_scatter_mean`：按索引分组求平均

```python
@staticmethod
def _scatter_mean(values, indices, size):
    batch, _, dim = values.shape
    out = values.new_zeros((batch, size, dim))
    expanded = indices.view(1, -1, 1).expand(batch, -1, dim)
    out.scatter_add_(1, expanded, values)          # 按 indices 把 values 加到 out
    counts = values.new_zeros((batch, size, 1))
    counts.scatter_add_(1, ..., values.new_ones(...))  # 顺便数每个桶收到几条
    return out / counts.clamp_min(1.0)             # 总和 ÷ 个数 = 平均;除零防 clamp
```

`indices` 就是 02 章那张"49 格 → 27 地点"的常量表。手算一个小例子（dim=1 时）：

```text
values = [[10], [20], [30]]      indices = [0, 0, 1]      size = 2
scatter_add 后 out = [[30], [30]]   counts = [[2], [1]]
mean    = [[15], [30]]
```

这就是"格 → 地点"的聚合，也是后面"边消息 → 节点"复用的同一把工具。

---

## 3.4 核心：三轮图消息传递（手写 GNN）

Brass 的灵魂是**连通性**：建网络要连通、卖货要走铁路到商家、铁路时代运煤过运河。02 章的状态张量只是"每格每条边各自的信息"，网络必须让它**沿图流动**才能推理"我建这条连接后能不能够到那个商家"。这就是图神经网络（GNN）的消息传递（message passing）：

> **一句话直觉**：每条边看看自己两端的地点现在"知道什么"，更新自己的认知；每个地点再看看挂在它上面的所有边"知道什么"，更新自己的认知。重复 3 轮，信息就能传播 3 步远。

先取出常量拓扑（02 章的 buffer）：

```python
a, b = self.edge_endpoints[:, 0], self.edge_endpoints[:, 1]   # 每条连接的两端地点编号
via_valid = self.edge_via_farms < be.LOCATION_COUNT           # 是否绕行农场
via = self.edge_via_farms.clamp_max(be.LOCATION_COUNT - 1)    # 无效行填 0(配合掩码归零)
```

然后 3 轮（`self.graph_layers = 3`），每轮两步：

```python
for edge_update, node_update in zip(self.edge_updates, self.node_updates):
    # 第一步:边看节点 —— 拼接[自己, 端点a, 端点b, 农场节点]再线性变换
    via_node = node[:, via] * via_valid.view(1, -1, 1)        # 无农场的连接此项全 0
    edge = edge_update(torch.cat([edge, node[:, a], node[:, b], via_node], dim=-1))
    #              Linear(64 + 128 + 128 + 128 = 448 → 64) + ReLU

    # 第二步:节点看边 —— 每条边把自己的新信息发给它的两个端点(和农场)
    incident = torch.cat([a, b, via[via_valid]], dim=0)        # 收件人地点编号列表
    messages = torch.cat([edge, edge, edge[:, via_valid]], dim=1)  # 对应的消息
    node = node_update(torch.cat([
        node, self._scatter_mean(messages, incident, be.LOCATION_COUNT)
    ], dim=-1))
    #        Linear(128 + 64 = 192 → 128) + ReLU
```

shape 走查（每一轮）：

- `edge_update` 的输入 `(B, 39, 448)`：39 条边并行，每条 448 维 = 自己 64 + 两个端点各 128 + 农场 128。
- `node_update`：`_scatter_mean` 把 78+ 条"边消息"按收件人归组取平均（同一把 3.3 的工具，复用在 27 个地点上），输出 `(B, 27, 64)`；与节点自身 `(B, 27, 128)` 拼成 `(B, 27, 192)`，线性层输出新节点 `(B, 27, 128)`。

**为什么正好 3 轮**：1 轮后每个地点只知道直接邻居，3 轮后"隔着 3 条连接的地点"也能互相影响——对"我这一步建网/卖货能触及什么"这类推理够用了。层数是 `NetConfig.graph_layers`，属于可调超参。

> **这不是 PyTorch 内置的 GNN 库**，是 30 行手写的定向消息传递。好处是没有依赖、完全可控；代价是每加一种图关系都要手写。读代码时抓住模式即可：**边吸收两端 → 节点吸收邻边 → 循环**。

---

## 3.5 汇聚成局面向量：池化 + trunk

```python
board = torch.cat([node.mean(dim=1), node.max(dim=1).values], dim=1)   # (B, 256)
links = torch.cat([edge.mean(dim=1), edge.max(dim=1).values], dim=1)   # (B, 128)
return self.trunk(torch.cat(
    [board, links, batch["global"], batch["own_hand"], batch["opp_hands"]], dim=1
))
```

1. **双池化**：`mean(dim=1)` 把 27 个地点平均成一个向量（"整体盘面怎么样"），`max(dim=1)` 取逐维最大值（"最突出的特征是什么"，比如"存在某个地点资源极多"）。两种池化拼接，兼顾全局印象与局部尖峰。
2. 拼接非图信息：全局 168 + 己方手牌 35 + 对手手牌 105，加上 board 256 / links 128，得到 `(B, 692)`。
3. `trunk` 是两层 MLP（`Linear(692→256) + ReLU + Linear(256→256) + ReLU`），输出 **state 向量 `(B, 256)`**——整个局面的最终摘要，后面所有输出头都从它出发。

---

## 3.6 主头：给 N 个候选打分（FiLM + 集合上下文）

这是本项目最有原创性的部分。输入是 `(B, N, 301)` 的变长候选特征（02 章），目标是 `(B, N)` 的分数。

### 第一步：动作编码

```python
self.action_encoder = nn.Sequential(
    nn.Linear(301, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(),
)
actions = self.action_encoder(action_features)      # (B, N, 301) → (B, N, 128)
```

每个候选动作独立压成 128 维"动作摘要"。N 个候选共享同一套权重（同一个函数，逐行调用）。

### 第二步：FiLM 调制——让状态决定"怎么解读每个动作"

```python
gamma, beta = self.film(state).chunk(2, dim=-1)     # film: Linear(256 → 256),拆成两个 (B,128)
modulated = gamma.unsqueeze(1) * actions + beta.unsqueeze(1)   # (B,N,128)
```

`gamma`（γ）和 `beta`（β）是从 state 向量算出来的两组 128 维系数。对每个候选：

```text
调制后动作 = γ ⊙ 动作摘要 + β     (⊙ = 逐元素相乘)
```

**为什么用乘法而不是把 state 和 action 拼接后丢给 MLP？** 乘性调制是"状态对动作的**结构性**影响"：γ 的某一维是 0.1，就表示"当前局面下，动作的这一维特征不重要，压到 1/10"；γ 是 3，表示"这一维现在非常关键"。数字例子：

```text
动作摘要 a   = [2.0, 0.5, 1.0]
γ(局面)     = [0.1, 3.0, 1.0]
β(局面)     = [0.0, 1.0, 0.0]
modulated   = [0.2, 2.5, 1.0]
             # 第 1 维被压制 → "钱数这个因素,现在不重要"
             # 第 2 维被放大 → "酒够不够,现在是决定性的"
```

同一动作在不同局面得到不同表示——这正是"这个动作好不好**取决于局面**"的数学表达。拼接方案理论上也能学到这种交互，但 FiLM 把它做成了结构保证。

### 第三步：集合上下文——打分要看整个候选集

```python
weights = candidate_mask.unsqueeze(-1).to(modulated.dtype)      # (B,N,1),padding 为 0
ctx = (modulated * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)  # (B,128)
ctx = ctx.unsqueeze(1).expand(-1, modulated.shape[1], -1)       # 广播回 (B,N,128)
```

`ctx` 是**所有真候选调制后表示的平均**（padding 位置权重为 0，不参与平均——这就是"masked mean"）。它把"这次有哪些可选"压成一个 128 维向量，广播给每个候选。

**为什么需要它**：一个动作的好坏是相对的。"在 Derby 建制造厂"单独看不错，但如果候选集里同时存在"在 Derby 建玻璃厂且能顺路连通商家"，它的分数就该打折。没有集合上下文的网络对两个候选各自独立打分，表达不了"它们在竞争同一个槽位"。有了 ctx，分数变成 `score(动作, 局面, 其他选项)`。

### 第四步：三路拼接打分

```python
self.action_score = nn.Sequential(
    nn.Linear(3 * 128, 256), nn.ReLU(), nn.Linear(256, 1),
)
logits = self.action_score(
    torch.cat([modulated, ctx, modulated * ctx], dim=-1)   # (B,N,384)
).squeeze(-1)                                              # (B,N,1) → (B,N)
```

每个候选的打分输入是三段拼接：**自己的调制表示 ⊕ 集合上下文 ⊕ 两者的逐元素乘积**。最后一段（乘积）让"这个动作与整体环境的交互"直接进入打分函数。输出 `(B, N)` 的 logits。

### 第五步：masked log_softmax

```python
log_probs = torch.log_softmax(logits.masked_fill(~candidate_mask, float("-inf")), dim=1)
```

把 padding 位置的 logits 填成 −∞，再按候选维做 log_softmax。−∞ 经 softmax 的 `exp` 变成**精确的 0 概率**——padding 位永远分不到概率质量，归一化只在真候选上进行。这就是"网络不学习合法性"（02 章）的落实处：非法性不是学出来的，是被 mask 硬保证的。

> 整段打分头是 **O(N)** 的（每个候选独立过打分函数），没有候选两两之间的 self-attention（那是 O(N²)）。集合上下文用"一次平均"廉价地引入了集合级依赖，是精度与成本的折中——这也是 [ai-action-encoding.md §1](../ai-action-encoding.md) 说的"pointwise + 集合上下文"方案。

---

## 3.7 输出头：价值、冠军与经济

state 向量 `(B, 256)` 除了喂给打分头，还直连四个小头：

```python
self.rank_head      = nn.Linear(256, 4)
self.winner_head    = nn.Linear(256, 4)
self.econ_canal_head = nn.Linear(256, 2)
self.econ_rail_head  = nn.Linear(256, 2)
```

### rank 头：预测四个人终局名次

输出 `(B, 4)`，目标是"名次 ÷ 人数"（第 1 名 0.25、第 4 名 1.0，01 章）。用 MSE 训练。

**为什么价值要这样设计，而不是预测"我的胜率"？** 三个理由（[ai-action-encoding.md §5.2](../ai-action-encoding.md)）：

1. **跨局可比**：0.25~1.0 的归一化名次在任何一局里含义一致；
2. **局内保序**：同一次预测里，第 2 名的值 < 第 1 名的值，四人的相对关系被结构保留；
3. **与搜索对齐**：搜索需要的"某座位的价值"取 `value = 1 − rank`（第 1 名 0.75，第 4 名 0.0）——**越大越好**，且 Rust 侧 MCTS 的终局回传值用同一个定义（`terminal_value`），两边尺度无缝衔接。MCTS 是"每个座位最大化自己的 value"的 MaxN 搜索，所以 value 必须按座位给出 `(B,4)` 而不是单个标量。

### winner 头：预测冠军

`(B, 4)` 的 logits，目标唯一冠军的 one-hot，交叉熵训练。它和 rank 头信息高度重叠，但目标形式不同（"名次分布" vs "谁拿第一"），0.5 的权重让它作为轻量补充。

### econ 头：经济辅助任务

canal / rail 两个独立的 `Linear(256, 2)`，输出拼成 `(B, 4)`：前 2 位预测运河时代的 `(收入, 现金)`，后 2 位预测铁路/终局的。训练时样本按自己的时代只训练对应的头（04 章）。

这是一个**辅助任务（auxiliary task）**：网络主任务（打分）很难直接从稀疏的终局信号中学到"钱很重要"，让一个免费的小头顺便预测经济，等于强迫 state 向量里保留"当前经济状况"的信息，反过来帮助打分。总权重只有 0.2——辅助任务，喧宾不夺主。

---

## 3.8 forward 的入口细节

回到 `forward` 开头，两处工程处理：

```python
if action_features.ndim == 2:
    action_features = action_features.unsqueeze(0)     # 兼容单局面调用
if action_features.dtype == torch.uint8:
    action_features = action_features.float().div_(ACTION_FEATURE_SCALE)  # uint8/4 → float
```

- uint8 入参（01 章：跨进程传输的压缩格式）**在 GPU 上**还原成 float——数据进显存时是 1/4 大小，还原在显存里完成，几乎零成本；
- 形状/掩码随后有一组显式校验（`ValueError`）：batch 数不一致、mask 形状不对、"每个局面至少要有一个真候选"都在入口拦下。防御性编程在这份代码里很常见，训练跑几个小时才炸的 shape 错误，值得在入口就报清楚。

---

## 3.9 推理包装：`policy_value()`

```python
def policy_value(self, batch, action_features, candidate_mask=None):
    was_training = self.training
    self.eval()
    try:
        with torch.no_grad():
            return self.forward(batch, action_features, candidate_mask)
    finally:
        self.train(was_training)
```

搜索时（05 章）MCTS 每批问一次网络：`eval()` 切评估模式，`no_grad()` 不建计算图（省显存省计算），`finally` 恢复训练模式——保证这包装饰可以在训练循环中随时穿插调用而不污染训练状态。

---

## 3.10 参数都在哪：63.6 万的账单

实测 `sum(p.numel() for p in net.parameters()) == 636045`。粗略构成（四舍五入）：

| 部件 | 参数量级 | 说明 |
| --- | --- | --- |
| trunk（692→256→256） | ~24 万 | 最大的单块：局面汇总 MLP |
| action_score（384→256→1） | ~10 万 | 打分头 |
| action_encoder（301→128→128） | ~5.5 万 | 每候选共享 |
| film（256→256） | ~6.5 万 | FiLM 调制 |
| 图消息传递（3 轮 edge+node） | ~16 万 | 448→64 与 192→128 各三层 |
| 位置 Embedding + 池化入口 | ~1 万 | 27×128 + 39×64 等 |
| 四个输出头 | ~0.2 万 | 小头几乎不占参数 |

感受一下量级：这比现代 LLM 小 7 个数量级，单卡轻松训练；但比"几个全连接层"大得多——大头花在"把 692 维拼盘压成 256 维局面理解"和"逐候选打分"上。

---

## 3.11 小结：一段话复述 forward

> 棋盘格先各自线性投影、按地点聚合，加上可学习的地点名片；边和节点做 3 轮"边看两端、节点看邻边"的消息传递，让连通性信息流动起来；双池化成全局印象，与 global、双方手牌拼接，trunk 压成 256 维局面向量。每个候选动作特征独立编码成 128 维，被局面向量经 FiLM 乘性调制；全部候选的调制表示取平均得到"集合上下文"，与自身、交互项拼接后打分，masked softmax 归一化。局面向量同时预测四人名次、冠军和两段经济——四路人马共享同一个"对局面的理解"。

## 练习

1. 实例化网络并核对形状合同：
   ```python
   import sys; sys.path.insert(0, "python")
   import torch
   from brass_ai.net import PolicyValueNet
   net = PolicyValueNet()
   B, N = 2, 5
   batch = {"board": torch.randn(B, 24, 49), "links": torch.randn(B, 7, 39),
            "global": torch.randn(B, 168), "own_hand": torch.randn(B, 35),
            "opp_hands": torch.randn(B, 105)}
   af = torch.rand(B, N, 301) * 2           # 随手造的假特征,仅看形状
   mask = torch.ones(B, N, dtype=torch.bool); mask[1, 3:] = False   # 第二个局面只有 3 个真候选
   out = net(batch, af, mask)
   for k, v in out.items(): print(k, tuple(v.shape))
   ```
   确认 `candidate_log_probs[1, 3:]` 全是 −inf（padding 被 mask）。
2. 打印 `sum(p.numel() for p in net.parameters())`，应为 636045；再把 `NetConfig(graph_layers=1)` 传进去，看参数量变化来自哪里。
3. 思考题：如果把第三步的 `ctx` 去掉（每个候选独立打分），哪个场景最先露馅？（提示：候选集里存在"竞争同一格/同一资源"的互斥动作时。）
