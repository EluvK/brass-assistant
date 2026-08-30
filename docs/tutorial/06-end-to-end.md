# 06 · 端到端跑一遍：`bootstrap_imitation.py`

> **本章目标**：把前六章串成一条可操作的命令——每个参数在调什么、磁盘上会留下什么、终端输出的每行指标怎么读、中断了怎么恢复。
> 主读代码：`python/bootstrap_imitation.py`（232 行，一个 `main()` 走天下）。
> 前置：全部前六章（本章到处引用它们）。

---

## 6.1 这条命令干哪八件事

```bash
./.venv/Scripts/python.exe python/bootstrap_imitation.py
```

`main()` 的执行顺序（对照代码读）：

```text
① 检测设备(cuda/cpu),固定随机种子 torch.manual_seed(0)
② 准备数据目录:默认 checkpoints/bootstrap.pt.imitation/
     目录里已有 imitation-*.pkl shard → 直接复用(不重新下棋)
     否则 → 并行生成 1000 局启发式对局,边生成边落 shard(01 章 1.6)
③ 建网络 PolicyValueNet() + Trainer(含 AdamW/调度器/scaler)(04 章 4.1)
④ --resume 时从 checkpoint 恢复全套训练状态(04 章 4.9)
⑤ 训练:每个逻辑 epoch = 把所有 shard 各过一遍
     每个 shard:加载 pickle → train_one_epoch(物化池 + 装箱 + GPU)(04 章)
     全部 shard 完 → scheduler.step() 一次 → 原子写 checkpoint
⑥ (可选 --enable-policy-eval)对全部 shard 跑 evaluate_policy 打指标行(04 章 4.8)
⑦ benchmark:网络 MCTS(c_puct=2.5, 60 模拟) vs 启发式,20 局轮换座位(05 章 5.5)
⑧ 收尾:关物化池;按参数决定 shard 目录去留
```

注意 TrainConfig 里 `epochs=1` 的含义：`train_one_epoch` 每次 = "把一个 shard 过一遍"；`--epochs 10` 在外层循环 10 次，每次扫完全部 shard 才算一个**逻辑 epoch**、才踩一次学习率调度——学习率周期对齐的是"数据全集过了几遍"，不是"某个 shard 过了几遍"。

---

## 6.2 参数手册

按用途分组（默认值来自 `argparse` 定义）：

### 数据规模

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `--games` | 1000 | 生成多少局启发式对局（有质量过滤时指**被接受的**局数） |
| `--epochs` | 10 | 全部数据过几遍 |
| `--batch` | 256 | 训练批大小 |
| `--max-candidate-batch` | 131072 | 单块候选行预算（04 章 4.5；比 Trainer 默认 65536 更宽） |
| `--lr` | 1e-3 | 初始学习率 |

### 并行与显存

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `--workers` | min(4, CPU 核数) | 生成对局的进程数（01 章 1.7 的 spawn 池） |
| `--materialize-workers` | min(8, CPU 核数) | 训练时恢复 snapshot 的进程数（01 章 1.5） |

### 质量过滤（可选）

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `--min-avg-vp` | 无 | 只保留平均 VP 更高的对局 |
| `--min-vp` | 无 | 只保留**最低分玩家** VP 也达标的对局（排除畸形局） |
| `--max-attempts` | games×10 | 过滤开启时的尝试上限，防止死循环 |

### 断点与复用

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `--ckpt` | checkpoints/bootstrap.pt | checkpoint 路径（shard 目录名由它派生） |
| `--resume` | 关 | 从 `--ckpt` 恢复模型+优化器+调度器+scaler |
| `--sample-dir` | 无 | 显式指定/复用别的 shard 目录 |
| `--delete-samples-on-success` | 关 | 成功后删除 shard 目录（省磁盘，但之后无法 resume 续训） |

### 模式开关

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `--shortlist-candidates` | 关（=full-legal） | 开启后改用教师短列表训练（v4 默认全合法集，01 章） |
| `--mcts-full-legal` | 关 | benchmark 的搜索用全合法候选而非短列表（慢很多） |
| `--enable-policy-eval` | 关 | 训练后跑一遍 top-k 指标评估 |

### 评测规模

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `--eval-games` | 20 | benchmark 对局数（05 章：统计粒度 5%，别用 2 局下结论） |
| `--eval-sims` | 60 | 每步 MCTS 模拟数（"思考时间"） |

---

## 6.3 磁盘上会留下什么

```text
checkpoints/
├── bootstrap.pt                      # 主 checkpoint(模型+优化器+调度器+scaler+schema 门禁,04 章 4.9)
└── bootstrap.pt.imitation/           # shard 目录(名字 = ckpt 路径 + ".imitation")
    ├── imitation-000000.pkl          # 每个 = pickle 的 list[Sample],≤32768 个样本(01 章)
    ├── imitation-000001.pkl
    └── ...
```

两条保留规则（`finally` 块）：

- 训练**成功**：shard 目录默认保留（供 `--resume` 续训增删 epoch）；加 `--delete-samples-on-success` 才删；
- 训练**失败/中断**：shard 目录一定保留，并打印提示——重跑 `--resume` 直接复用已生成的对局数据，不用再下 1000 局棋。

checkpoint 属于本地资产（`checkpoints/` 已被 gitignore），不会进 git。

---

## 6.4 终端输出逐行解读

一次典型运行的关键行（数字为示意）：

```text
device: cuda                                    # ① GPU 可用;显示 cpu 说明走 CPU(慢一个量级以上)
generated imitation shards for 1000 heuristic games (952s)
                                                # ② 生成阶段耗时;复用旧 shard 时是
                                                #    "reusing N imitation shards from: ..."
epoch 1 (this run 1/10) ...
[train e1 s3/10] 13300/31744 elapsed:  35s |ETA:  48s
                                                # ⑤ 进度行(progress.py,单行覆盖刷新):
                                                #    第 3/10 个 shard,已训 13300 样本,ETA 48 秒
checkpoint updated after epoch 1: checkpoints/bootstrap.pt  # 每个逻辑 epoch 落盘一次
...
trained 281344 samples (10453s): policy=1.243 rank=0.051 winner=0.987
                                                # ⑤ 结束行:总样本数(1000 局约 28 万个决策点,
                                                #    只统计本次运行第一个 epoch,06 章示例非实测)、
                                                #    训练总耗时、三项主损失的平均值(04 章 4.2)
```

**损失行怎么读**（训练是否健康的第一信号）：

- `policy`：初始约为 `ln(平均候选数)`（网络均匀乱猜时 CE 的期望值，全合法集下可达 5~6）。健康下降到 1~2 区间表示"排序大体学会了"；若几乎不降，回 04 章查学习率/数据；
- `rank`：从 ~0.5（乱猜名次）往下走，0.05 左右意味着名次预测平均偏差 ~0.2 个名次；
- `winner`：从 ~1.39（ln4，四选一乱猜）往下；注意它是 CE 量纲，别和 top1 命中率混淆。

**（`--enable-policy-eval` 时）指标行**：

```text
policy eval: top1=38.2% top3=66.0% top5=78.1% winner_top1=31.5% entropy=1.85 candidates=412.3/p95=1930
```

- `top1/top3/top5`：网络的排序与老师的吻合度（04 章 4.8 表）。全合法集平均几百个候选里 top1 能到 30%+ 已经远超均匀基线（<1%）；
- `entropy`：预测分布熵。观察它**随 epoch 的走势**——持续暴跌要警惕过早自信（04 章）；
- `candidates=412.3/p95=1930`：候选数分母背景——平均 412 个可选动作，95 分位局面近 2000 个候选，top 指标是在这么大的选项空间里取得的。

**benchmark 行**（05 章 5.5）：

```text
MCTS(bootstrap net) vs heuristic: win_rate=45% (mcts_vp=41.3 vs heuristic_vp=35.8)
```

- 4 人局随机基线是 25%；45% 表示"第一名次数接近随机的两倍，且对手是启发式"；
- `mcts_vp − heuristic_vp` 的差比胜率更连续、对小样本更稳；
- 20 局的粒度是 5%，**两次训练之间 win_rate 相差 <10 个百分点不能下"更强"的结论**。

---

## 6.5 中断与恢复：推荐的实操节奏

`--resume` + 原子写（04 章 4.9）让训练可以随时掐断。推荐的节奏（逐 epoch 的停止时机判断依据见 04 章 4.10）：

**第一步：冒烟测试**（分钟级，验证管线通）：

```bash
./.venv/Scripts/python.exe python/bootstrap_imitation.py \
    --games 20 --epochs 1 --eval-games 4 --eval-sims 20 \
    --ckpt checkpoints/smoke.pt
```

跑通全链路（生成 → 训练 → 评测 → benchmark），确认 device 是 cuda、没有 OOM/shape 报错。checkpoint 用独立名字，不污染正式 run。

**第二步：正式训练**（小时级）：

```bash
./.venv/Scripts/python.exe python/bootstrap_imitation.py --epochs 1
./.venv/Scripts/python.exe python/bootstrap_imitation.py --resume --epochs 1
# ……按需重复,每次一个逻辑 epoch,随时可停
```

一次只训一个 epoch 的好处：每个 epoch 结束都有完整 checkpoint 落盘，实验中途想评估/改参数，随时停且无损。时间消耗大头在**生成阶段**（只有第一次运行有）和每个 epoch 的训练阶段——进度条自带 ETA（`progress.py` 的单行进度），跑起来看一眼就有预期。

**第三步：评估取舍**：`--eval-games` 每加一局都是一整局 60 模拟的对局时间；`--enable-policy-eval` 要再扫一遍全部 shard，会显著拉长收尾。冒烟/调试阶段都关掉，正式 run 再开。

---

## 6.6 第一次运行的常见坑

| 症状 | 原因与处理 |
| --- | --- |
| `device: cpu` | 没装 CUDA 版 torch 或没 NVIDIA 卡。能跑但慢一个量级以上；冒烟测试请把 `--games/--epochs` 再调小 |
| Windows 下 worker 报 import 错误 | 必须以脚本方式从仓库根目录运行（worker 是 spawn 重新 import 本脚本的，01/06 章的 Windows 约定）；不要在交互式 REPL 里 import 它跑 main |
| GPU OOM | 调小 `--batch` 或 `--max-candidate-batch`（候选矩阵是显存大头，04 章 4.5） |
| `only accepted N/M imitation games ...` | 质量过滤太严：放宽 `--min-avg-vp/--min-vp` 或提高 `--max-attempts` |
| resume 报 "missing trainer state" | 目标 checkpoint 是旧格式（没存 optimizer 等），删掉它重新训一个"可恢复"的 checkpoint |
| 加载 checkpoint 报 schema 版本错误 | 特征编码升级过，旧权重作废（02 章 schema 门禁——这是特性不是 bug） |
| 内存吃满 | 调小 `--workers` / `--materialize-workers`（每个进程都要载入 numpy/torch，01 章 1.7 的 BLAS 线程与内存注释） |

---

## 6.7 学完之后：往哪里走

bootstrap 产出的是一个"会玩 Brass"的暖启动网络（`checkpoints/bootstrap.pt`）。教程覆盖的主线到此为止，仓库里还有三块"下一阶段"的代码，现在你应该有能力自己读懂它们：

- **`train.py` 的 `LoopConfig` / `run_loop`**：最简单的 self-play → train 循环参考实现（05 章闭环的最小代码化；当前无顶层入口调用，是学习/实验用骨架）；
- **`mp_selfplay.py`**：多进程 self-play worker 池（`SelfPlayPool`），广播权重、并行下棋、支持历史模型 matchmaking——把闭环的数据生成端工业化；
- **`replay_worker.py`**：把 checkpoint 变成"决策服务"——stdin/stdout JSON 协议应答 Rust 的 `choose` 请求（协议见 [replay-design.md](../replay-design.md)），是通往 TTS 实时辅助（项目阶段 4/5）的桥。

这三块零件怎么组装成持续自我提升的系统、以及之后所有演进方向的盘点，见 [08 章（self-play 闭环）](08-selfplay-loop.md) 和 [09 章（演进方向全景）](09-evolution.md)。

## 练习

1. 跑 6.5 的冒烟命令，对照 6.4 逐行解释你屏幕上的每一行输出。
2. 把冒烟 checkpoint 的 benchmark 与 `--eval-sims 60` 的对比一次（同 `--eval-games 4`），验证"模拟次数 = 思考时间"。
3. 打开 `checkpoints/bootstrap.pt` 看看里面有什么（04 章 4.9 的键列表）：
   ```python
   import torch
   ckpt = torch.load("checkpoints/bootstrap.pt", map_location="cpu")
   print(ckpt.keys(), ckpt["epoch"], ckpt["action_feature_dim"])
   ```
4. （选做）给冒烟 run 加 `--enable-policy-eval`，记录 top1/top3 数值，作为你自己模型的第一个基线——以后每次改动都与它比较。
