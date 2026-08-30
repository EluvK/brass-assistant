# 00 · 神经网络最小知识包

> **本章目标**：不借助任何外部资料，建立一个足以读懂本项目全部 AI 代码的神经网络直觉。
> 学完的验收标准：能逐行读懂 `python/brass_ai/train.py` 文件头注释里的损失公式，并能亲手跑通 [toy_nn.py](toy_nn.py)。
>
> 配套脚本：[toy_nn.py](toy_nn.py)（约 90 行，只依赖 PyTorch，和本项目代码无关）。

---

## 0.1 本项目里的"神经网络"到底是什么

抛开所有术语，本项目训练的东西是**一个可调参的打分函数**：

```text
输入: 一个牌面局面 + 一个候选动作  →  输出: 一个分数(这个动作有多好)
                                    →  外加: 这局最后四个人大概的名次
```

"可调参"的意思是：函数内部有 **636,045 个数字**（实测 `PolicyValueNet` 的参数总数，见 03 章），每个数字就是一个"旋钮"。训练 = 反复微调这 63.6 万个旋钮，直到打分函数给出的分数和"标准答案"足够接近。

旋钮初始是随机的，所以刚初始化的网络是"瞎打分"。训练的全部意义就是：给它看大量"局面 + 标准答案"的例子，让它自己把旋钮拧到合适的位置。

> **类比**：一个函数式编程工程师可以这样理解——网络就是 `score = f_θ(state, action)`，其中 `f` 的结构（几层矩阵乘法）是我们写死的，`θ`（63.6 万个参数）是让机器自动搜出来的。传统软件里我们手写 `f` 的逻辑；神经网络里我们只写 `f` 的"骨架"，逻辑本身从数据中学出来。

---

## 0.2 最小单元：线性层（`nn.Linear`）

神经网络的基本积木是**线性层**：把输入向量做一次矩阵乘法再加偏置。

```python
self.board_enc = nn.Sequential(nn.Linear(be.BOARD_PLANES, self.cfg.board_emb), nn.ReLU())
```

`nn.Linear(24, 128)` 的含义：输入一个 24 维向量，输出一个 128 维向量，内部持有参数矩阵 `W`（形状 128×24）和偏置 `b`（形状 128），计算 `y = W x + b`。

**手算一个 2→2 的例子**（所有层都是它的放大版）：

```text
W = [[1, 2],      b = [0, 1]      x = [3, 4]
     [0, 1]]

y = W x + b
y[0] = 1*3 + 2*4 + 0 = 11
y[1] = 0*3 + 1*4 + 1 = 5
```

`W` 和 `b` 就是"旋钮"：训练会不断修改它们。一个 128×24 的线性层有 128×24 + 128 = 3392 个旋钮——本项目 63.6 万个旋钮主要就是这样一层层堆出来的。

> **shape 规则**：`nn.Linear(in, out)` 接受 `(batch, in)` 的输入，输出 `(batch, out)`。`batch` 这一维是"一次同时算多少个样本"，网络代码里到处都是它。本项目里 batch 常记作 `B`。

---

## 0.3 为什么需要非线性：ReLU

如果只把线性层叠起来，数学上等价于一层更大的线性层（矩阵乘矩阵还是矩阵乘），网络再深也只能表达"直线关系"。所以每层之后都要接一个**非线性函数**，本项目用的是最简单的 ReLU：

```text
ReLU(x) = max(0, x)      负数清零，正数不变
ReLU([3, -1, 0, 7]) = [3, 0, 0, 7]
```

就这么个"负数清零"的操作，堆叠起来却能拟合任意复杂的曲线（toy 脚本 A 部分用两层 `Linear+ReLU` 拟合出了 sin 曲线）。这就是代码里到处可见的固定搭配：

```python
nn.Linear(a, b), nn.ReLU()      # 一层"带激活的全连接"
```

---

## 0.4 Embedding：给每个"类别"学一张名片

棋盘上有 27 个地点。我们希望每个地点有一个自己的 128 维向量（"名片"），而且名片内容可以学习——这就是 **Embedding**：

```python
self.node_position = nn.Embedding(be.LOCATION_COUNT, self.cfg.board_emb)
# LOCATION_COUNT = 27 → 一张 (27, 128) 的可学习查找表
```

它本质就是一张 `27×128` 的参数表，输入地点编号 `i`，返回第 `i` 行。和"先构造 27 维 one-hot 再乘一个 27×128 矩阵"数学上完全等价，只是查表更快。

> **什么时候用 Embedding**：特征是"类别编号"（地点、玩家、词表单词）而不是"数值"（钱数、资源量）时。数值直接进网络，类别先过 Embedding。

---

## 0.5 学习是怎么发生的：loss、梯度、优化器

### loss：给"错得有多离谱"打分

训练需要一个标量来衡量"这次答错了多少"，这就是**损失函数（loss）**。本项目用到两种：

**MSE（均方误差）**——预测连续值时用（本项目：预测终局名次）：

```text
MSE = mean((预测 - 答案)²)
预测 [0.3, 0.9], 答案 [0.5, 0.7]  →  ((−0.2)² + (0.2)²) / 2 = 0.04
```

**交叉熵（Cross-Entropy, CE）**——预测"概率分布"时用（本项目：policy 打分、冠军预测）。
它分两步。第一步 **softmax** 把任意分数（logits）变成合法概率分布：

```text
softmax([2, 1, 0]):
  e² ≈ 7.389,  e¹ ≈ 2.718,  e⁰ = 1      总和 ≈ 11.107
  → [0.665, 0.245, 0.090]               全为正、总和 = 1
```

第二步，CE = `−Σ 答案ₐ × log(预测ₐ)`。当答案是 one-hot（只有正确项为 1）时，它退化成一句话：**"正确动作的预测概率越接近 1，loss 越接近 0"**：

```text
答案 = [0, 1, 0]，预测 = [0.2, 0.5, 0.3]
CE = −log(0.5) ≈ 0.69      预测对正确项越自信，这个数越小
```

### 梯度下降：蒙眼下山

loss 是关于 63.6 万个旋钮的函数。训练目标：调整旋钮让 loss 变小。

梯度（gradient）就是"每个旋钮往哪边调、调多少能让 loss 下降最快"的方向指示。你不需要会推导它——PyTorch 的 `loss.backward()` 一行就自动算出所有 63.6 万个旋钮的梯度（这叫**反向传播**，backpropagation，自动微分的一种）。

拿到方向后，每个旋钮朝反方向（下坡方向）挪一小步：

```text
θ_new = θ_old − lr × gradient
```

`lr`（learning rate，学习率）是步长，本项目默认 `1e-3`。太大→来回震荡甚至发散；太小→学得极慢。这是训练中最常调的旋钮。

### batch 和 epoch

- **batch（批）**：一次同时喂给网络的样本数（本项目 `batch_size=256`）。损失取批内平均。用 batch 而不是单样本，是因为：① GPU 擅长大矩阵并行，批越大吞吐越高；② 256 个样本的"平均梯度"比单样本的梯度更稳、噪声更小。
- **epoch（遍）**：全部训练数据完整过一遍叫一个 epoch。本项目 bootstrap 默认对同一批数据训 10 遍（`--epochs 10`）。（同一个数据集一轮轮学下去会发生什么、何时停，见 04 章 4.10 的训练动力学。）

### optimizer：聪明的调旋钮策略

最朴素的更新就是上面的"梯度反方向挪一步"（SGD）。本项目用 **AdamW**：每个旋钮额外维护自己的"历史梯度统计"，自动放大步长小的维度、抑制震荡，类似一个记住每个旋钮近期调整历史的智能调节器。工程上只需记住：**AdamW 是可靠的默认选择，一般不用换**。

### 正则化：L2 / weight decay

`L = … + 1e-4 × Σθ²` 这一项的含义：所有旋钮的平方和也算进 loss。它惩罚"把某个旋钮拧得特别大"的极端解，防止网络死记硬背训练数据（**过拟合**，overfitting），让学到的打分函数更平滑。本项目在 `compute_loss` 里显式计算它（`l2: float = 1e-4`）。

---

## 0.6 一次真实的 PyTorch 训练循环（全文背下来也就 5 步）

所有训练代码——包括本项目 500 多行的 `train.py`——核心都是这个循环：

```python
model = ...                                   # 网络(一堆可学习参数)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

for batch in data:                            # 每个 batch 是一批"输入+答案"
    optimizer.zero_grad()                     # 1) 清掉上一轮的梯度
    pred = model(inputs)                      # 2) 前向传播:算出预测
    loss = loss_fn(pred, target)              # 3) 用损失函数打分
    loss.backward()                           # 4) 反向传播:自动算出全部梯度
    optimizer.step()                          # 5) 优化器按梯度更新全部参数
```

[toy_nn.py](toy_nn.py) 就是这个循环的两个实例。**现在就运行它**（在仓库根目录）：

```bash
./.venv/Scripts/python.exe docs/tutorial/toy_nn.py
```

预期输出（数字可能略有差异）：

```text
== A 回归:拟合 sin(x)(MSE,对应 rank 头)==
A  step    0  mse = 0.5896
A  step  500  mse = 0.0000
A  完成:最大误差 |sin(x) - model(x)| = 0.0048

== B 分类:3 选 1(softmax + 交叉熵,对应 policy/winner 头)==
B  step    0  ce = 1.1798  acc = 30.1%
B  step  500  ce = 0.1591  acc = 96.5%
B  完成:accuracy = 100.0%
```

A 部分演示 MSE（对应本项目 rank 头），B 部分演示 softmax+交叉熵（对应 policy 和 winner 头）。两段代码的骨架和 `train.py` 的 `train_on_batch` 完全同构。

---

## 0.7 训练与推理是两种状态

```python
out = net.policy_value(batch, action_features, candidate_mask)
```

这是本项目搜索时用的推理入口（`net.py`）。它做了两件惯例性的事：

- `self.eval()`：切换到评估模式。对含 Dropout/BatchNorm 的网络会改变行为（本项目两者都没用，纯属规范习惯）。
- `with torch.no_grad()`：告诉 PyTorch"不用记录计算图"。推理不需要求梯度，不记录能省大量内存和计算。

反过来，训练前要 `net.train()`（`train_on_batch` 第一行），且**必须**在有梯度的模式下跑 `forward → backward`。

---

## 0.8 GPU 与张量搬运

PyTorch 的张量（tensor）就是"带设备属性的 numpy 数组"。本项目所有重活都在 GPU 上做：

```python
batch = {k: v.to(device) for k, v in batch.items()}   # CPU → 显存
```

两张关键卡的带宽有限（数据从内存拷进显存很贵），这解释了本项目两个看似奇怪的优化，后面章节会展开：

- 动作特征以 uint8 传到 GPU，在 GPU 上再转 float（省 4× 传输量）；
- MCTS 攒一批局面（`batch_size=64`）一起问网络，而不是每步问一次。

---

## 0.9 本项目的网络全景（预览）

后面三章会把每个方框讲透。现在只需记住形状合同：

```text
状态五件套                            动作特征 (N, 301)   ← 每个候选动作一行
  board      (24, 49)  棋盘 49 格 × 24 平面                    │
  links      (7, 39)   39 条连接 × 7 平面                     ▼
  global     (168,)    全局信息                    ┌─────────────────────┐
  own_hand   (35,)     己方手牌                    │   PolicyValueNet    │
  opp_hands  (105,)    三个对手的手牌               │   (63.6 万参数)      │
                                            └─────────────────────┘
                                                    │
        ┌───────────────┬───────────────┬───────────┴───────┐
        ▼               ▼               ▼                   ▼
   policy: N 个     rank: 4 人终局   winner: 4 人冠军     econ: 经济
   候选各一个分数    名次预测(MSE)    概率(交叉熵)          辅助预测
```

- **policy** 是主头：给当前局面的每个合法候选动作打分，分数经 masked softmax 变成概率。
- **rank / winner** 是价值头：不看好动作，只看"这个局面发展下去最后谁赢"。
- **econ** 是辅助头：顺带预测经济状况，帮网络建立"钱和收入很重要"的内部概念（权重只有 0.2）。

---

## 0.10 验收：读懂 train.py 的损失公式

打开 `python/brass_ai/train.py`，文件头有这样一段注释：

```text
Loss (per sample, head set v4):
  L = -sum_a p_a * log_softmax(score(s, a))_a     (policy CE over concrete
      Engine-generated legal candidates; padding is masked only for batching)
    + ||rank_4 - target_4||^2                     (MSE on per-seat normalized
      final rank; the search value for a seat is 1 - rank)
    + 0.5 * winner_CE                             (official winner one-hot CE)
    + 0.2 * econ_MSE                              (era-split auxiliary heads)
    + l2 * ||theta||^2
```

逐行翻译：

| 公式 | 含义 | 用到的知识 |
| --- | --- | --- |
| `−Σ pₐ log qₐ` | policy 交叉熵：网络对"应该走哪"的概率分布 `p` 与预测分布 `q` 的差距 | 0.5 节 CE |
| `‖rank − target‖²` | 名次预测的均方误差 | 0.5 节 MSE |
| `0.5 × winner_CE` | 冠军 one-hot 交叉熵，权重 0.5 | 0.5 节 CE |
| `0.2 × econ_MSE` | 经济辅助头（权重 0.2） | MSE |
| `l2 × ‖θ‖²` | 正则项：旋钮别拧太大 | 0.5 节 L2 |

五项相加 = 一个 batch 的总损失。总损失越小 = policy 打分越准 + 名次/冠军预测越准 + 经济理解越对 + 参数不过分——训练就是在降这一个数。

如果你能看懂这张表，第 0 章的任务就完成了。接下来按顺序读 01 章。

## 练习

1. 跑通 [toy_nn.py](toy_nn.py)，把 A 部分的 `lr=1e-3` 改成 `5e-2` 再改成 `1e-5`，观察三种学习率下 MSE 下降速度的差异。
2. 把 A 部分网络改成 `nn.Sequential(nn.Linear(1, 1))`（去掉 ReLU 和隐藏层），确认它拟合不了 sin——这就是 0.3 节"没有非线性就不行"的实证。
3. 在 B 部分的 `loss.backward()` 前后各打印 `clf[0].weight.sum().item()`，确认参数真的被更新了。
4. （选做）把 `PolicyValueNet()` 实例化，数一数参数：`sum(p.numel() for p in net.parameters())`，应为 636,045。
