****# Python AI 架构与操作手册

本文描述 `src/ai/` 当前可用的 Python AI 训练、搜索和评测工具。它面向开发与实验，不是 TTS 实时服务的部署文档。

## 范围与权威边界

Python 侧不实现游戏规则。Rust 扩展模块 `brass_engine` 是以下能力的唯一权威来源：`GameState` 状态、规则动作与回合推进、合法动作及固定策略槽映射、特征编码、对手手牌确定化、网络引导 ISMCTS 树，以及内置启发式教师策略。

Python 只负责组织数据、定义 PyTorch 网络、实现 Rust 搜索所需的批量推理回调，以及编排自博弈、训练和评测。规则或策略槽变化时，应先更新 Rust bridge，再同步更新 Python 契约、测试和本文。引擎整体分层见 [engine-tools.md](engine-tools.md) 与 [architecture.md](architecture.md)。

## 目录与职责

```text
src/ai/
|- brass_ai/
|  |- build_input.py  Rust NumPy 特征 -> PyTorch batch
|  |- net.py          PolicyValueNet：策略、价值、经济辅助头
|  |- rust_mcts.py    Rust `GameState.search_net` 的回调适配器
|  |- selfplay.py     自博弈、启发式模仿数据和 Sample 标签
|  |- mp_selfplay.py  常驻 spawn worker 池与样本序列化
|  |- dataset.py      replay NPZ 分片读写
|  |- train.py        损失、Trainer、优化器和学习率调度
|  |- evaluate.py     网络 MCTS 与 Rust 启发式的固定种子对局
|  `- progress.py     控制台进度与 ETA
|- train_mp.py        正式多进程训练入口
|- bootstrap_imitation.py
|                     启发式行为克隆预训练入口
|- bench_mp.py        单进程/多进程自博弈吞吐比较
|- experiments/       基准、回放、诊断和一次性实验脚本
`- tests/             bridge、网络、训练、自博弈、分片的回归测试
```

## 运行架构

```text
PolicyValueNet 权重
        |
        v
SelfPlayPool (多个 CPU worker) -- Rust GameState.search_net --> Rust ISMCTS
        |                                                    |
        |                 batched NumPy 特征                | Python 回调
        +----------------------------------------------------+
        |
        v
Sample 列表 -> replay/iter-xxxxx.npz -> 有界 replay buffer
                                             |
                                             v
                                      Trainer (主进程 GPU/CPU)
                                             |
                                             v
                         checkpoint + metrics + 固定种子 benchmark/gate
```

`RustISMCTS.search()` 返回 `SearchResult`：`best` 为 canonical 动作字符串，`visits` 为根节点 `slot -> visit count`，`canon_by_slot` 为可执行动作。自博弈用访问次数构造策略监督目标，并按温度采样动作；评测关闭根节点噪声并使用最佳动作。

## Rust-Python 契约

当前正式训练路径只支持四人局。虽然 Rust `GameState` 可创建 2--4 人局，但网络的价值头固定为 4 维、对手手牌编码固定为 3 组，不能把 `SelfPlayConfig.players` 当作通用人数配置使用。

| 项目 | 当前契约 |
| --- | --- |
| board | `float32 (17, 49)` |
| links | `float32 (6, 39)` |
| global | `float32 (50,)` |
| own_hand | `float32 (35,)` |
| opp_hands | `float32 (105,)`，三名对手各 35 |
| 策略空间 | Rust 定义的 1316 个槽；运行时以 `brass_engine.policy_table_size` 为准 |
| 策略 logit | `type_logits[:, slot_type(slot)] + goal_logits[:, slot]` |
| 价值标签 | 四名玩家终局 VP 的 `z = (vp - mean) / std`，平局时全零 |
| 经济辅助标签 | `(income_level, money)`；运河样本取运河结束时值，铁路样本取终局值 |

训练与推理均只能在 `state.legal_mask()` 返回的槽上归一化或选取动作。不要在 Python 复制动作到槽的映射，也不要依据网络输出直接拼装 canonical 动作；应由 Rust `legal_moves()` 或 `legal_moves_slots()` 提供可执行动作。

## 网络与损失

`PolicyValueNet` 将棋盘和连接分别逐格线性编码、做 mean/max pooling，再与全局及手牌特征拼接进入两层 MLP trunk。它有四个输出头：

- `type_head (7)`：build、network、develop、sell、loan、scout、pass。
- `goal_head (P)`：每个策略槽的目标 logit。
- `value_head (4)`：四位玩家的标准化终局 VP 预测。
- `econ_head (2)`：当前视角的收入和现金预测，仅作训练辅助，不参与 Rust 搜索。

训练总损失为掩码策略交叉熵、四维 value MSE、经济辅助损失与显式 L2 的和。`Trainer` 持有 AdamW、CosineAnnealingLR 和 epoch 计数；不能在每轮重新创建 Trainer，否则优化器动量和学习率进度会丢失。

## 环境准备

所有命令在仓库根目录运行，以下示例采用 PowerShell。

```powershell
# 激活引擎虚拟环境，并让 Python 能导入 src/ai 下的包
.\src\engine\.venv\Scripts\Activate.ps1
$env:PYTHONPATH = "src/ai"

# Rust bridge 或其导出契约变更后重新安装扩展
Push-Location src/engine
python -m maturin develop --release
Pop-Location

# 回归测试
python -m pytest src/ai/tests -q
```

若机器没有 CUDA，所有会调用网络的命令显式传入 `--device cpu`；`benchmark.py`、`diagnose.py`、`replay_net.py` 的默认值是 `cuda`。正式多进程训练通常使用“主进程 GPU、worker CPU”：`--worker_device cpu`。

## 常用流程

### 1. 启发式模仿预训练

此路径让 Rust 内置启发式在完整对局中行动，记录 one-hot 策略目标，再训练初始网络。质量筛选是严格大于阈值；`--games` 指接受的对局数，筛选开启时必须给出足够的 `--max-attempts`。

```powershell
python src/ai/bootstrap_imitation.py `
  --games 200 `
  --epochs 3 `
  --workers 4 `
  --min-avg-vp 80 `
  --min-vp 58 `
  --max-attempts 2000 `
  --eval-games 8 `
  --eval-sims 40 `
  --ckpt checkpoints/bootstrap-smoke.pt
```

正式批量示例：

```powershell
python src/ai/bootstrap_imitation.py `
  --games 2000 `
  --epochs 10 `
  --workers 8 `
  --min-avg-vp 80 `
  --min-vp 58 `
  --max-attempts 20000 `
  --ckpt checkpoints/bootstrap.pt
```

该脚本写出的 checkpoint 只有模型 `state_dict`；它可作为 `train_mp.py --ckpt` 输入，但不能作为 `--resume` 的完整 Trainer 恢复文件。

### 2. 多进程自博弈训练

`train_mp.py` 是当前正式入口。每轮 worker 使用当前网络生成完整自博弈，主进程保存分片、截取 replay buffer、执行固定数目的梯度步，再按频率执行基准。`--gate` 会用当前进程中已评测的最佳网络对候选网络做接受/回滚，已知恢复语义限制见 [ai-problems.md](ai-problems.md)。

```powershell
python src/ai/train_mp.py `
  --ckpt checkpoints/bootstrap.pt `
  --run_dir runs/main `
  --iters 8 `
  --workers 8 `
  --games_per_worker 2 `
  --worker_device cpu `
  --sims 200 `
  --train_steps 60 `
  --replay_size 6000 `
  --bench_sims 400 `
  --bench_games 20 `
  --gate
```

中断后恢复同一 run：

```powershell
python src/ai/train_mp.py `
  --run_dir runs/main `
  --out runs/main/checkpoints/latest.pt `
  --resume `
  --start_iter 8 `
  --iters 16 `
  --worker_device cpu
```

`--resume` 会恢复 `latest.pt` 中的模型、优化器、scheduler，并读取已有 replay 分片后按 `--replay_size` 截取。传入的参数不会由旧 `manifest.json` 自动恢复，恢复前需人工确认本次 CLI 参数与 manifest 中的契约一致。

### 3. 决策级基准

固定 seeds `0..games-1`，网络 MCTS 座位轮换，其余三席为 Rust 启发式。输出 MCTS 胜率、VP 分布和各 seed 的 VP，适合比较两个 checkpoint。

```powershell
python src/ai/experiments/benchmark.py `
  --ckpt runs/main/checkpoints/latest.pt `
  --sims 600 `
  --games 40 `
  --device cpu
```

### 4. 诊断与回放

| 入口 | 用途 | 示例 |
| --- | --- | --- |
| `bench_mp.py` | 单进程与 worker 池的自博弈吞吐对比 | `python src/ai/bench_mp.py --sims 60 --workers 8` |
| `experiments/net_all_vs_all.py` | 同一网络四席互弈；`--sims 0` 为纯贪心 | `python src/ai/experiments/net_all_vs_all.py --ckpt checkpoints/bootstrap.pt --start 0 --end 19 --sims 0 --device cpu` |
| `experiments/diagnose.py` | 汇总网络席的动作、时代分布和低分 seed | `python src/ai/experiments/diagnose.py --ckpt checkpoints/bootstrap.pt --sims 200 --seeds 0-19 --device cpu` |
| `experiments/replay_net.py` | 逐动作中文回放，输出到 `src/engine/logs/` 或 `--out` | 参数帮助当前有已知错误，见问题清单；直接运行可传 `--ckpt ... --seed 7 --device cpu` |
| `experiments/bc_baseline.py` | 旧的独立行为克隆实验，非正式训练路径 | 仅用于对照实验 |

实验目录中的脚本可能采用不同的回合驱动与默认参数，不能将其结果直接混入正式 run 的 `metrics.jsonl`。对外比较优先使用 `experiments/benchmark.py`。

## 训练产物与检查

`--run_dir runs/main` 下的文件含义如下：

```text
runs/main/
|- manifest.json                 本次启动时记录的 bridge 尺寸与 CLI 参数
|- replay/iter-00000.npz         每轮完整对局产生的压缩 Sample 分片
|- checkpoints/latest.pt          模型 + optimizer + scheduler + epoch
`- metrics.jsonl                 每轮一条 JSON，含损失、耗时、样本数及可选 benchmark
```

分片包含 `pid`、`era`、五组特征、`policy`、`value`、`econ` 和 `legal`。只接受正常结束的完整对局；自博弈超过 `max_moves` 会丢弃整局并向 worker 报错，避免把中间局面错误标作终局价值。

训练前后至少检查：

- `manifest.json` 的策略空间和特征尺寸是否仍与当前 Rust 扩展一致。
- `metrics.jsonl` 的 samples 是否为预期数量，loss 是否为有限值。
- 用相同的 `--sims`、`--games` 和设备运行 benchmark，对比胜率、均值和中位数。
- 改动 bridge、样本格式或训练损失后，运行 `python -m pytest src/ai/tests -q`。

## 常见故障

`ModuleNotFoundError: brass_ai`：确认在仓库根目录，并设置 `$env:PYTHONPATH = "src/ai"`。

`ModuleNotFoundError: brass_engine` 或接口属性缺失：激活 `src/engine/.venv` 后，在 `src/engine` 执行 `python -m maturin develop --release`。

`Torch not compiled with CUDA enabled` 或 CUDA 初始化失败：为相应脚本加 `--device cpu`；多 worker 时保持 `--worker_device cpu`。

worker 报 `samples discarded`：某局超过 `max_moves=600` 或产生不可执行动作，当前池会中止本轮。保留 seed、参数和完整错误信息后再定位；不要把截断局强制写入 replay。

更多已确认或需决策的实现风险见 [ai-problems.md](ai-problems.md)。
