# Python AI 当前架构与操作手册

本文只描述当前保留在 `python/` 的代码。

## 职责边界

Rust 扩展 `brass_ai._engine` 是游戏规则与动作语义的唯一权威，负责：

- `GameState`、回合推进、合法动作与 canonical action；
- 状态 tensor 和候选动作特征编码；
- heuristic teacher；
- candidate-policy-guided ISMCTS 搜索树。

Python 不实现规则或解析动作。它负责网络前向、训练样本组织、模仿训练、Rust 搜索的网络回调，以及自对弈支持代码。

## 当前数据流

```text
Rust GameState
  -> legal_candidates() / heuristic_candidates()
  -> Python PolicyValueNet(state tensors, candidate features)
  -> candidate logits + 4-player value + econ prediction
  -> Rust GameState.search_net() through Python callback

Rust heuristic self-play
  -> imitation Sample
  -> Trainer
  -> checkpoint
  -> Rust MCTS vs heuristic benchmark
```

候选集中的动作数 `N` 随局面变化。Python 只给 Rust 提供的候选动作打分，不能自行判断合法性。

## Rust-Python 契约

当前训练路径固定为四人局。`GameState` 虽可支持 2--4 人，网络的 value head 和对手手牌编码均固定按四人局设计。

| 数据 | 形状/含义 |
| --- | --- |
| board | `float32 (24, 49)` |
| links | `float32 (7, 39)` |
| global | `float32 (168,)` |
| own_hand | `float32 (35,)` |
| opp_hands | `float32 (105,)`，三名对手手牌 |
| candidate features | `float32 (N, 301)` |
| policy target | 对这 `N` 个候选归一化的分布 |
| rank target | 每座位终局名次 / n（VP → 收入 → 现金确定性破平局） `(4,)` |
| winner target | 唯一冠军（破平局后第一名）的 one-hot `(4,)` |
| econ target | `(income_level, money)`，按时代拆分的辅助监督 |

动作特征 schema 当前为 `ACTION_FEATURE_SCHEMA_VERSION = 4`，状态特征 schema 为 `STATE_FEATURE_SCHEMA_VERSION = 4`。Python adapter 和 checkpoint 会拒绝未知 schema；Rust 修改编码时必须同步更新这些位置和测试。动作特征 301 维的具体布局见 [ai-action-encoding.md](./ai-action-encoding.md)。

## 当前目录

```text
python/
|- bootstrap_imitation.py     heuristic imitation warm-start 入口
|- brass_ai/
|  |- hierarchical_policy.py  Rust 候选动作和 teacher adapter
|  |- net.py                  PolicyValueNet
|  |- rust_mcts.py            Rust 搜索的 Python 网络回调
|  |- selfplay.py             Sample、imitation 与 MCTS self-play
|  |- train.py                Trainer、loss、训练指标
|  |- evaluate.py             MCTS 对 heuristic 的评测
|  |- mp_selfplay.py          多进程 self-play worker pool
|  |- replay_worker.py        replay-web 网络座位子进程（stdin/stdout JSON 协议）
|  |- progress.py             长任务进度输出
|  `- __init__.py             包定义
`- tests/                     当前回归测试
```

## 环境与回归

虚拟环境支持 Python 3.9–3.12（pyproject `requires-python = ">=3.9,<3.13"`），推荐使用 3.12。初次安装虚拟环境命令如下：

```powershell
uv venv --python 3.12 .venv
source .venv/Scripts/activate
uv pip install -e ".[dev]"
```

后续使用虚拟环境时，只需激活虚拟环境即可：

```powershell
source .venv/Scripts/activate
```

安装 Rust 扩展和运行回归测试

```powershell
# Rust bridge 有修改时需要重新安装扩展
maturin develop --release

# Python 当前回归测试
python -m pytest python/tests -q
```

## 当前可运行入口

### Heuristic imitation bootstrap

`bootstrap_imitation.py` 是保留的训练入口。Rust heuristic 进行完整对局，Python 使用每步 teacher 候选集训练候选评分、value 和经济辅助头，再让网络引导 Rust MCTS 与 heuristic 对战。

```powershell
python python/bootstrap_imitation.py `
  --games 20 --epochs 1 --workers 1 `
  --eval-games 2 --eval-sims 10 `
  --ckpt checkpoints/bootstrap-smoke.pt
```

这是一条小规模 smoke 命令。正式训练前应先确认 Rust 扩展已由当前源码构建，且 Python 测试通过。

| 参数 | 用途 |
| --- | --- |
| `--ckpt` | checkpoint 输出路径（默认 `checkpoints/bootstrap.pt`） |
| `--games` | heuristic 对局数（默认 1000） |
| `--epochs` | 每个 replay shard 的训练轮数 |
| `--workers` / `--materialize-workers` | imitation 生成进程数 / 训练时 snapshot 候选物化进程数；`1` 为串行 |
| `--batch` / `--max-candidate-batch` | 每次读取的样本数 / 一个训练 micro-batch 的候选行预算（限制 padding 造成的显存峰值） |
| `--lr` | AdamW 学习率 |
| `--eval-games` / `--eval-sims` | 结尾 benchmark 的对局数（默认 20）/ 每步模拟数（默认 60） |
| `--min-avg-vp` / `--min-vp` / `--max-attempts` | 样本质量门槛：整局平均 VP / 最差座位 VP 下限，及启用门槛后的对局尝试上限 |
| `--shortlist-candidates` | 改用 heuristic shortlist 候选训练；**默认 full-legal**（完整合法候选 + one-hot teacher，样本存 Rust snapshot，训练前实时物化候选集），会显著增加 CPU/内存压力 |
| `--resume` | 从 `--ckpt` 恢复完整 Trainer 状态（模型/optimizer/scheduler/scaler） |
| `--sample-dir` | 复用已有 `imitation-*.pkl`，跳过重新生成 |
| `--delete-samples-on-success` | 成功结束后删除默认的 `<ckpt>.imitation` 样本目录 |
| `--enable-policy-eval` | 训练后统计全部 shard 上的 top-k policy 指标 |
| `--mcts-full-legal` | 仅让结尾 benchmark 使用全合法候选 |

当前源码中 `--max-candidate-batch` 的默认值是 `131072`。GPU 显存有限时应显式设置更小值，例如 `16384`，并从小规模运行开始。

### 自对弈模块的状态

`selfplay.py`、`mp_selfplay.py` 和 `train.py` 仍保留网络 MCTS self-play 所需能力，但当前没有保留的顶层长期 self-play 训练脚本。它们是后续重新设计自对弈训练入口时可复用的基础模块，不应把它们当作已完成的端到端训练工作流。

## 样本与 checkpoint

`Sample` 代表一个决策点。默认 full-legal 模式下，样本只保存 Rust state snapshot + teacher canonical action（不保存状态张量），候选集与状态张量在训练前从 snapshot 实时物化；`--shortlist-candidates` 模式则直接保存状态张量与 shortlist 候选特征。候选上的监督包括 policy 分布、rank/winner 终局目标与 econ 目标。

Trainer checkpoint 包含：

```text
model
optimizer
scheduler
scaler
epoch
action_feature_dim
action_feature_schema_version
state_feature_schema_version
state_feature_shapes
```

因此 `--resume` 只能使用 Trainer 生成的完整 checkpoint，不能使用只含模型参数的文件。

## 修改后的最低验证标准

- 修改 Rust bridge、候选特征或网络输入：重新构建 Rust 扩展，并运行 `python -m pytest python/tests -q`。
- 修改 loss、`Sample` 或训练 batch：运行全部 Python 测试，并用小规模 bootstrap smoke 验证。
- 修改 Rust 搜索 callback：至少确认 `test_mcts_selfplay.py` 通过，且 smoke benchmark 能完成。
