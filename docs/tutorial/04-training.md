# 04 · 损失函数与训练循环：`train.py`

> **本章目标**：读懂训练侧的全部代码——五项损失各在罚什么、"一个 batch 怎么进 GPU"、变长候选批的装箱问题、混合精度与 NaN 防护、学习率调度与 checkpoint，以及 **epoch 一轮轮学下去会发生什么、靠什么控制、何时停**（4.10，训练动力学）。
> 主读代码：`python/brass_ai/train.py`（545 行）。
> 前置：00 章（训练五步循环）、01 章（Sample 与物化）、03 章（网络输出的形状合同）。

---

## 4.1 `Trainer` 持有什么

```python
class Trainer:
    """Persistent optimizer + LR scheduler around a PolicyValueNet."""
    def __init__(self, net, cfg=None):
        self.net = net
        self.optimizer = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=0.0)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=100, eta_min=1e-5)
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp_enabled)
        self._pool = None        # 惰性创建的物化进程池(01 章)
        self._step_count = 0     # 用于周期性 NaN 深检计时
```

类注释里的 "persistent" 是重点：**optimizer 和 scheduler 跨 shard、跨 epoch 复用**。AdamW 为每个参数维护动量统计，如果每次训练都重建 optimizer，等于每轮都失忆重学——这份代码曾经踩过这个坑（模块 docstring 原话："previously each call rebuilt Adam and lost all state"），现在 Trainer 一旦创建就贯穿整个训练过程。

`TrainConfig` 的关键默认值（完整注释见代码）：

| 字段 | 默认 | 含义 |
| --- | --- | --- |
| `epochs` / `batch_size` | 5 / 256 | 每次调用训几遍 / 批大小 |
| `lr` | 1e-3 | 初始学习率 |
| `l2` / `weight_decay` | 1e-4 / 0 | 显式 L2（进 loss）/ AdamW 解耦衰减（关） |
| `amp` | True | GPU 上用 fp16 混合精度 |
| `grad_clip_norm` | 5.0 | 梯度范数裁剪上限 |
| `econ_lambda` | 0.2 | 经济辅助损失权重 |
| `max_candidate_batch` | 65536 | 单个训练块的候选行预算（见 4.5） |
| `materialize_workers` / `materialize_rpc_chunk` | 4 / 32 | 物化进程池（01 章） |
| `finiteness_check_interval` | 100 | CUDA 上 NaN 深检的步间隔（见 4.4） |

---

## 4.2 `compute_loss`：五项损失逐个看

总公式（`train.py` 文件头，00 章已翻译过一遍）：

```text
L = policy_CE + rank_MSE + 0.5·winner_CE + 0.2·econ_MSE + 1e-4·‖θ‖²
```

### ① policy 交叉熵（主损失）

```python
out = net(tensors, tensors["candidates"], tensors["candidate_mask"])
target = tensors["policy"]                                  # (B, N) 标准答案分布
policy_loss = -(target * out["candidate_log_probs"].masked_fill(
    ~out["candidate_mask"], 0.0
)).sum(dim=1).mean()
```

逐项拆解：

- `candidate_log_probs` 是网络预测的对数概率（03 章 masked log_softmax，padding 位是 −inf）；
- **为什么 padding 位还要 `masked_fill(…, 0.0)`？** 因为 `target` 在 padding 位是 0，而 `0 × (−inf) = NaN`。把 −inf 换成 0 后，`0 × 0 = 0`，padding 位干净地不产生损失。这是浮点世界的经典陷阱：数学上"0 乘任何数"成立，涉及 inf 就不成立。
- `.sum(dim=1)` 对一个样本 = `−Σₐ targetₐ · log qₐ`。答案是 one-hot 时退化为"−log(正确候选的预测概率)"；答案是 MCTS visit 分布时就是对整个分布的加权惩罚（01 章）。
- `.mean()` 对 batch 取平均，得到标量。

**手算例子**（00 章 B 部分就是这个）：答案 one-hot 指向候选 2，网络给它的概率 0.5 → loss = −log 0.5 ≈ 0.69；概率 0.9 → 0.105。训练把这个数往下压 = 让正确候选的概率往上顶。

### ② rank MSE

```python
rank_loss = F.mse_loss(out["rank"], tensors["rank"])
```

网络预测的四人名次 vs 真实名次/4（01 章）。MSE 对四个分量取平均。这项教网络"看局面评终局"——它就是搜索用到的 value 头（`value = 1 − rank`）的训练信号。

### ③ winner 交叉熵（×0.5）

```python
winner_loss = -(tensors["winner"] * F.log_softmax(out["winner_logits"], dim=1)).sum(dim=1).mean()
```

和 policy CE 同构，只是对象从"N 个候选"换成"4 个座位"。权重 0.5：它与 rank 头信息重叠，是轻量补充。

### ④ econ MSE（按时代拆分，×0.2）

```python
econ_target = tensors["econ"]                     # (B,2) = (income_level, money)
inc_t = ((econ_target[:, 0] + 10.0) / 40.0).clamp(0.0, 1.0)   # 归一化到 ~[0,1]
money_t = (econ_target[:, 1] / 100.0).clamp(0.0, 1.0)
econ = out["econ"]                                # (B,4) = [canal头|rail头]
era = tensors["era"]
for active, pred in ((era == 0, econ[:, :2]), (era != 0, econ[:, 2:])):
    mask = active.float()
    if mask.sum() == 0:
        continue
    inc_pred = (pred[:, 0] + 10.0) / 40.0         # 网络输出用同样的归一化
    money_pred = pred[:, 1] / 100.0
    ...
    era_loss = era_loss + (inc_loss + money_loss) * mask.mean()
econ_loss = econ_lambda * era_loss                # ×0.2
```

要点：**运河样本只训练 canal 头，铁路样本只训练 rail 头**（01 章的盖章规则与此对应：canal 样本的目标是运河末经济，rail 样本是终局经济——每种头只见一种目标定义，不会自相矛盾）。预测和目标都用 `(收入+10)/40`、`现金/100` 归一化到 0~1，和 rank 一样为了跨局可比。每个时代块的损失再按"该时代样本占批内的比例"加权，最后整体乘 0.2。

### ⑤ L2 正则

```python
l2_loss = sum(p.pow(2).sum() for p in net.parameters()) * l2
```

所有 63.6 万个参数的平方和 ×1e-4（00 章：防止旋钮拧过大）。注意 `AdamW(weight_decay=0)`——AdamW 自带的"解耦权重衰减"被关掉了，L2 显式写在 loss 里。两种正则数学上相似但路径不同（L2 走梯度并被 Adam 的自适应步长缩放；weight decay 直接缩参数），显式 L2 让正则强度在打印的 loss 明细里可见、可控。

### 汇总

```python
total = policy_loss + rank_loss + winner_weight * winner_loss + econ_loss + l2_loss
return total, policy_loss, rank_loss, winner_loss, econ_loss, l2_loss
```

返回六元组——训练日志里打印的每项损失都来自这里，方便诊断"哪一项出了问题"。

---

## 4.3 `train_on_batch`：一步训练的全过程

```python
def train_on_batch(net, batch, cfg, optimizer, scaler=None, deep_check=None):
    net.train()                                  # 切训练模式(00 章 0.7)
    optimizer.zero_grad(set_to_none=True)        # 1) 清梯度(set_to_none 省一趟清零)
    losses = compute_loss(batch, net, cfg.l2, cfg.device, ...)
    if not torch.isfinite(losses[0]):
        raise FloatingPointError("non-finite loss before backward")
    losses[0].backward()                         # 2) 反向传播:63.6 万个梯度一次算完
    gradients_finite = all(                      # 3) NaN 巡检(条件执行,见下)
        torch.isfinite(p.grad).all().item() for p in net.parameters() if p.grad is not None)
    torch.nn.utils.clip_grad_norm_(net.parameters(), cfg.grad_clip_norm)  # 4) 裁剪
    optimizer.step()                             # 5) AdamW 更新
```

这就是 00 章 0.6 节的五步循环，多了两道保险：

- **梯度裁剪**（`clip_grad_norm_(…, 5.0)`）：如果整批梯度的范数超过 5，就等比例压回 5。异常 batch（脏数据、边界局面）偶尔会产生巨大的梯度把参数甩飞，裁剪是"保险丝"——正常训练几乎不触发，出事时救命。
- **NaN 巡检**：`isfinite` 检查梯度和（深检时）参数。数值训练的失败模式不是报错而是"悄悄变成 NaN 然后全盘污染"，早发现早止损。

### 混合精度（AMP）路径

GPU 上默认走 fp16 分支（`cfg.amp=True` 时）：

```python
with torch.autocast(device_type=cfg.device):     # 前向自动用 fp16
    losses = compute_loss(...)
scaler.scale(total).backward()                   # loss 先 ×65536 再反向
scaler.unscale_(optimizer)                       # 梯度缩回真实尺度
... 裁剪 + 深检 ...
scaler.step(optimizer)                           # 若发现 inf/NaN 梯度 → 本步直接跳过
scaler.update()                                  # 动态调整放大倍数
```

原理一句话：fp16 比 fp32 快、省一半显存，但小数值容易**下溢**成 0（精度不够），所以把 loss 放大 6.5 万倍再反向，让梯度"够得着" fp16 的最小值，更新前再缩回来。`GradScaler` 还顺带提供一道免费保险：检测到坏梯度就跳过这一步（参数不动），训练不会因单个坏批崩溃。

这也解释了 `finiteness_check_interval=100`：逐参数 `isfinite` 每次都强制 GPU 同步（`.item()` 要等所有异步计算完成），把流水线卡成串行——CPU 上 `.item()` 便宜可以每步查，GPU 上改成每 100 步做一次"深检"，平时的坏步由 GradScaler 兜底跳过。

---

## 4.4 `_to_batch`：变长样本拼成一个 batch

03 章说过网络吃 `(B, N, 301)`，但每个 Sample 的 N 不同，怎么拼？**补齐（padding）+ 掩码（mask）**：

```python
def _to_batch(samples):
    b = np.stack([s.board for s in samples]).astype(np.float32)      # (B,24,49)
    ...
    rows = [s.candidates for s in samples]
    if all(row.dtype == np.uint8 for row in rows):
        # 快路径:按 uint8 补零,进 GPU 后由 net.forward 再 /4 还原
        max_n = max(row.shape[0] for row in rows)
        candidates = np.zeros((len(rows), max_n, 301), dtype=np.uint8)
        candidate_mask = np.zeros((len(rows), max_n), dtype=bool)
        for i, row in enumerate(rows):
            n = row.shape[0]
            candidates[i, :n] = row              # 真候选写进前 n 行
            candidate_mask[i, :n] = True         # 其余是 padding
    ...
    pol = np.zeros(candidate_mask.shape, dtype=np.float32)
    for i, sample in enumerate(samples):
        pol[i, :len(sample.policy)] = sample.policy   # policy 目标同样补零
```

关键点：

- 一批里取最大的 N 补齐；`candidate_mask` 标出哪些位置是真的。padding 位置在 log_softmax 前被置 −inf（03 章）、在 loss 里被清零（4.2），**从头到尾不产生任何影响**；
- policy 目标补零天然正确——padding 位答案是 0；
- uint8 快路径：整批按 uint8 补零、整块拷进 GPU，在 GPU 上转 float（02 章），省 4× 主机→显存带宽。

返回的 dict 的 key 与 `net.forward` 的 `batch` 参数一一对应（board/links/global/own_hand/opp_hands）再加目标（policy/rank/winner/econ/era），`compute_loss` 第一行把它们整体 `torch.as_tensor(…, device=…)` 搬上 GPU。

---

## 4.5 变长批的装箱：`_pack_candidate_chunks`

补齐有个隐性代价：**padding 也占显存**。候选矩阵大小 = 样本数 × 批内最大 N × 301。256 个样本里混进一个 N=1000 的"巨无霸"局面，整批就被撑到 256×1000×301×4B ≈ 300MB——还不算它引发的所有中间激活。

解法不是"压扁所有样本"而是**按大小分批装车**：

```python
def _pack_candidate_chunks(samples, batch_size, max_candidate_batch):
    ordered = sorted(samples, key=lambda s: len(s.candidates))   # 按候选数排序
    chunk = []
    for sample in ordered:
        if chunk and (
            (len(chunk) + 1) * len(sample.candidates) > max_candidate_batch  # 行预算
            or len(chunk) >= batch_size                                        # 样本数上限
        ):
            yield chunk                      # 装不下这箱了,先发车
            chunk = []
        chunk.append(sample)
    if chunk:
        yield chunk
```

直觉：把样本从小到大排好，贪心地往一辆"货车"（一个训练块）里装，装到**候选行总数**（样本数 × 当前最大候选数）触及 `max_candidate_batch=65536` 预算就发车。因为已排序，一辆车内的 padding 浪费最小；某个巨无霸只把"装它那辆车"撑大，不影响别的车。这是货真价实的**装箱问题（bin packing）**的贪心近似——后端工程师秒懂的模式。

`train_one_epoch` 里的完整流水线：

```python
idx = np.random.permutation(len(samples))            # 洗牌
batches = [samples[idx[start:start+batch_size]] ...]  # 切成批
for raw in stream_materialized_batches(pool, batches, ...):   # 子进程提前物化(01 章)
    for chunk in _pack_candidate_chunks(raw, ...):            # 货车装箱
        losses.append(train_on_batch(self.net, _to_batch(chunk), ...))
```

三个并行层叠在一起：物化 worker 与 GPU 训练重叠（01 章）、一个物化批拆成多个装箱块、GPU 按块吃。`batch_size` 在这里退化为"每辆车的样本数上限"，真实限制是候选行预算。

---

## 4.6 训练的两种组织方式

**`train_one_epoch(samples)`**：把给定的样本完整过一遍（洗牌 → 批 → 块 → 步），**不**动学习率调度器。bootstrap 对每个 shard 各调一次它，全部 shard 过完才 `scheduler.step()` 一次（见 06 章）——"逻辑 epoch"由调用方拼装。

**`train_on_samples(samples)`**：外面包了 `cfg.epochs` 遍 + 每遍一次 `scheduler.step()` + 结尾附带 `evaluate_policy` 指标。是"独立训练一轮"的便捷入口。

**`train_steps(samples, n_steps)`**：第三种方式，为未来的 self-play 循环准备——**有放回**地随机抽 `n_steps` 个 minibatch（经典 experience replay：新数据不断进来、旧数据被反复重抽），同样不碰调度器（`step_lr()` 单独调，让余弦周期跟随"迭代数"而不是"步数"）。当前 `run_loop` 是它的参考实现。

---

## 4.7 学习率调度：CosineAnnealingLR

```python
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=100, eta_min=1e-5)
```

学习率从 `1e-3` 出发，按**余弦曲线**在 100 个逻辑 epoch 内降到 `1e-5`：前期大步快速逼近，后期小步精细打磨。为什么不用恒定学习率？大 lr 到训练后期会在最优解附近来回震荡压不下去；小 lr 从头开始又太慢。余弦退火（cosine annealing）是两者兼得的工业标准配方。

调度器也进了 checkpoint（4.9）——恢复训练时"现在已经衰减到多少"必须跟着回来，否则 lr 跳回 1e-3 会把微调好的参数震坏。

---

## 4.8 `evaluate_policy`：训练质量的仪表盘

```python
metrics = evaluate_policy(net, shard_samples, device, ...)
```

纯推理（`eval()` + `no_grad()`，不碰训练状态）扫一遍样本，产出：

| 指标 | 含义 | 怎么读 |
| --- | --- | --- |
| `policy_top1` | 网络排第一的候选 = 老师所选（等价类摊平后 argmax）的比例 | 核心指标；模仿学习中可以理解为"和老师一致率" |
| `policy_top3` / `top5` | 老师选择落在网络前 3 / 前 5 的比例 | top1 不高但 top3 高 = 排序大致对、自信心过强 |
| `policy_entropy` | 预测分布的平均熵 | 掉得太快太低 = 网络过早"一言堂"（过早收敛/过拟合信号） |
| `winner_top1` | 冠军预测命中率 | value 侧质量；4 人局瞎猜基线是 25% |
| `candidate_count_mean` / `p95` | 每局面候选数的均值/95 分位 | 不是成绩，是理解其他指标的"分母背景" |

top-k 的实现值得一看：`argsort` 拿到网络排序，再检查老师位置是否进前 k——纯排序比较，与概率绝对值无关，所以跨局面（候选数不同）可比。

bootstrap 打印的那行 `policy eval: top1=… top3=…` 就是这些指标（06 章逐行解读）。

---

## 4.9 checkpoint：存什么、为什么、怎么防中断

`Trainer.state_dict()` 打包的**不只是模型权重**：

```python
{
  "model":     net.state_dict(),          # 63.6 万参数本身
  "optimizer": optimizer.state_dict(),    # AdamW 每参数的动量统计
  "scheduler": scheduler.state_dict(),    # 余弦衰减走到哪了
  "scaler":    scaler.state_dict(),       # AMP 放大倍数
  "epoch":     self.epoch_count,
  # 下面是 schema 门禁:加载时强校验(02 章)
  "action_feature_dim": ACTION_FEATURE_DIM,               # 301
  "action_feature_schema_version": ACTION_FEATURE_SCHEMA_VERSION,  # 4
  "state_feature_schema_version": be.STATE_FEATURE_SCHEMA_VERSION,
  "state_feature_shapes": {"board": (24,49), "links": (7,39), "global": 168, "hand": 35},
}
```

为什么 optimizer/scheduler/scaler 都要存：恢复训练若丢掉 Adam 动量，优化器要重新"热身"，表现为 loss 短暂回升震荡；丢掉 scheduler，学习率跳回初值，直接把收敛中的参数震坏。

`load_state_dict` 先校验 schema 再加载——特征编码改版后（维度 301 变了、状态布局变了），旧 checkpoint 会被**明确拒绝**并给出人话报错，而不是加载成功然后在训练里产生无声的垃圾梯度。

bootstrap 里的 `save_checkpoint()` 还有一层**原子写**保护：

```python
# 先写进临时文件,再原子性改名覆盖正式 checkpoint
torch.save(trainer.state_dict(), tmp_path)
os.replace(tmp_path, ckpt_path)
```

训练中途 Ctrl+C / 断电，最多丢一个 epoch，`checkpoints/bootstrap.pt` 永远是完整可恢复的（不存在"写了一半的坏文件"）。`os.replace` 在同一文件系统内是原子操作——这是后端领域"写临时文件再 rename"的同款套路。

---

## 4.10 训练动力学：同一个数据集，epoch 一轮轮学下去会发生什么

4.6 讲了"一轮 epoch 怎么组织"，本节讲**多轮 epoch 的宏观行为**——这是"模仿学习"阶段最核心的控制问题。

### 机械上每轮发生什么

以 bootstrap（06 章）为例，一个逻辑 epoch = 依次对每个 shard 调 `train_one_epoch`（每遍都重新洗牌——`np.random.permutation`，所以每轮的 batch 组合不同，但**数据本身不变**），全部 shard 过完后 `scheduler.step()` 一次（学习率沿余弦曲线降一档）+ 原子写 checkpoint。`--resume` 会把调度器进度一起恢复（4.9），所以"第 N 轮用什么学习率"是确定的、可续的。

### 宏观上会发生什么：三个阶段

**阶段一（前几轮）：快速学习结构。** loss 陡降。网络在学"结构性知识"——动作类型先验（Sell 大概率优于 Scout）、各阶段的大方向。这是收益最高的阶段。

**阶段二（中期）：收益递减。** 大方向已对，剩余误差来自具体局面的细粒度判断，loss 下降明显变缓。

**阶段三（后期）：过拟合风险。** 继续在**同一批**数据上训，网络开始**记忆样本**而非学习模式。三个信号：

- 训练 loss 仍在降，但 **held-out 验证集**（09 章 9.2②）指标停滞或回升——"背题"与"会做题"分道扬镳；
- **policy 熵快速坠落**（4.8 的表）——预测越来越"一言堂"。这不只是过拟合的症状，还会直接伤害后续 MCTS：先验过尖，搜索失去探索空间（08 章 8.8 反模式 3 的前奏）；
- **benchmark 胜率不再涨甚至回退**——最终裁判永远是对局，不是 loss。

### 一个理论地板：policy loss 降不到 0

交叉熵有分解 `CE(p, q) = H(p) + KL(p‖q) ≥ H(p)`——即使网络完美拟合（KL=0），loss 也只能降到**目标分布自身的熵 H(p)**。本项目的 imitation 目标不是 one-hot：等价类摊平（01 章 1.4）、教师分数 softmax 都让答案分布自带熵。所以 **"loss 降到平台" ≠ "训练出问题"**——可能只是贴到了理论地板，结合 06 章 6.4 的指标行判断，别在地板上空耗轮数。

### 控制旋钮及其相互作用

| 旋钮 | 位置 | 控制什么 |
| --- | --- | --- |
| `--epochs` | bootstrap | 在固定数据上花几轮预算 |
| `--lr` | bootstrap | 阶段一的推进速度（也是阶段三发散的头号嫌疑人） |
| `CosineAnnealingLR` 的 `T_max` / `min_lr` | TrainConfig | 学习率衰减的节奏 |
| `--resume` | bootstrap | 轮与轮之间的人工控制点（每轮一个完整 checkpoint） |

一个容易忽略的相互作用：**默认 `--epochs 10` 只用掉余弦周期（`T_max=100`）的前 10%**——10 轮后学习率约 9.8e-4，几乎没衰减。这不是疏漏：bootstrap 的设计意图是"你随时 `--resume` 续训，学习率沿曲线继续走"。但如果你想让"一次跑完 + 学习率充分衰减收尾"，要么把 `T_max` 调到≈计划轮数，要么接受更多轮数自然收尾。

反过来说，**逐轮递减的学习率本身就是一层内置控制**：即使无脑加大轮数，后期每轮的步子也越来越小，过拟合的破坏力被部分抑制——但它不改变"越训越贴近这批数据"的方向，该停还是要停。

### 何时停：本项目的实操判断

1. 按 06 章 6.5 的节奏：`--epochs 1` + `--resume` 循环，每轮结束都有完整 checkpoint 可评估、可回退；
2. 每轮后看三样：训练 loss 趋势（4.8 表）、top-k 与熵、（最好）held-out 指标；
3. **指标平台 + 熵仍在快速掉 → 停**。模仿阶段的使命是"会玩"（warm-start），不是"完美复刻老师"——老师的能力就是模仿的天花板（08 章 8.1），剩余提升空间属于 self-play；
4. 终审交给 benchmark：对 heuristic 的胜率不再涨，模仿就该收工，预算转投 08 章的闭环。

### 与 self-play 闭环的关系

进入 08 章的迭代循环后，"固定数据集 + epochs"这个控制面**整体消失**——数据持续流入，改由"回放缓冲 + `train_steps(n_steps)`"控制训练量，"epoch"被"迭代"取代。所以 `--epochs` 是模仿阶段特有的旋钮；两套控制范式的交接点就是 teacher 退役（09 章 9.3）。

---

## 4.11 小结

> `compute_loss` 把"policy 打分准 + 名次/冠军预测准 + 经济理解对 + 参数别太大"压成一个数；`train_on_batch` 用"前向 → 反向 → 裁剪 → AdamW 步进"降低它，fp16 + GradScaler + 周期深检是 GPU 上的速度与稳定套装；`_to_batch` 用 padding+mask 拼变长候选，`_pack_candidate_chunks` 按行预算装箱控显存；数据由物化进程池提前备好与 GPU 重叠；checkpoint 存全套训练状态并带 schema 门禁与原子写。

## 练习

1. 用假数据走一遍最小的 `compute_loss`（不依赖引擎）：
   ```python
   import sys; sys.path.insert(0, "python")
   import torch
   from brass_ai.net import PolicyValueNet
   from brass_ai.train import compute_loss
   net = PolicyValueNet()
   B, N = 4, 7
   batch = {"board": torch.randn(B, 24, 49), "links": torch.randn(B, 7, 39),
            "global": torch.randn(B, 168), "own_hand": torch.randn(B, 35),
            "opp_hands": torch.randn(B, 105),
            "candidates": torch.rand(B, N, 301) * 2,
            "candidate_mask": torch.ones(B, N, dtype=torch.bool),
            "policy": torch.full((B, N), 1.0 / N),
            "rank": torch.rand(B, 4), "winner": torch.zeros(B, 4),
            "econ": torch.tensor([[20.0, 50.0]]).repeat(B, 1),
            "era": torch.zeros(B, dtype=torch.int64)}
   total, *parts = compute_loss(batch, net, l2=1e-4, device="cpu")
   print("total", total.item())
   total.backward()          # 确认梯度能一路算回参数
   ```
   把 `policy` 改成 one-hot 指向候选 0：均匀目标下 `policy_loss` 等于"平均每个候选的意外程度"，量级约为 `log(7)≈1.95`（预测也均匀时恰为该值），训练会让它下降。
2. 把 `TrainConfig(batch_size=4, max_candidate_batch=8)` 造一组候选数 1~6 的假样本，手动演算 `_pack_candidate_chunks` 会切成几块，再用代码验证。
3. 阅读题：`train_on_batch` 在 AMP 路径里，为什么裁剪发生在 `unscale_` 之后、`scaler.step` 之前？（提示：裁剪阈值是针对真实梯度尺度设的，不能作用在放大了 6.5 万倍的梯度上。）
4. **亲眼看三阶段**（4.10）：用 06 章冒烟参数跑 `--games 20 --epochs 3 --enable-policy-eval --eval-games 2 --eval-sims 10 --ckpt checkpoints/dynamics.pt`，每轮记录 policy loss 与熵的数值——对照 4.10 的三阶段，判断你的 3 轮里处于哪一阶段。（评测规模压到最小只求跑得快；`--eval-games` 不能设 0，benchmark 会对局数除零。）
