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
| board | `float32 (17, 49)` |
| links | `float32 (6, 39)` |
| global | `float32 (50,)` |
| own_hand | `float32 (35,)` |
| opp_hands | `float32 (105,)`，三名对手手牌 |
| candidate features | `float32 (N, 235)` |
| policy target | 对这 `N` 个候选归一化的分布 |
| value target | 四名玩家终局 VP 标准化向量 `(vp - mean) / std`；平局为全零 |
| econ target | `(income_level, money)`，辅助监督 |

动作特征 schema 当前为 `ACTION_FEATURE_SCHEMA_VERSION = 2`。Python adapter 和 checkpoint 会拒绝未知 schema；Rust 修改动作编码时必须同步更新这些位置和测试。

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
|  |- progress.py             长任务进度输出
|  `- __init__.py             包定义
`- tests/                     当前回归测试
```

## 环境与回归

虚拟环境需要 Python 3.12，初次安装虚拟环境命令如下：

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
| `--games` | heuristic 对局数 |
| `--epochs` | 每个 replay shard 的训练轮数 |
| `--workers` | imitation 生成进程数；`1` 为串行 |
| `--batch` | 每次读取的样本数 |
| `--max-candidate-batch` | 一个训练 micro-batch 的候选行预算，限制 padding 造成的显存峰值 |
| `--full-legal-candidates` | 用完整合法候选和 one-hot teacher target；会显著增加 CPU/内存压力 |
| `--resume` | 从 `--ckpt` 恢复模型、optimizer 和 scheduler |
| `--sample-dir` | 复用已有 `imitation-*.pkl`，跳过重新生成 |
| `--mcts-full-legal` | 仅让结尾 benchmark 使用全合法候选 |

当前源码中 `--max-candidate-batch` 的默认值是 `131072`。GPU 显存有限时应显式设置更小值，例如 `16384`，并从小规模运行开始。

### 自对弈模块的状态

`selfplay.py`、`mp_selfplay.py` 和 `train.py` 仍保留网络 MCTS self-play 所需能力，但当前没有保留的顶层长期 self-play 训练脚本。它们是后续重新设计自对弈训练入口时可复用的基础模块，不应把它们当作已完成的端到端训练工作流。

## 样本与 checkpoint

`Sample` 代表一个决策点，可直接保存 state tensor、候选特征与 policy；完整合法候选模仿模式还可保存 Rust state snapshot 和 teacher canonical action，在训练前实时恢复候选集。

Trainer checkpoint 包含：

```text
model
optimizer
scheduler
epoch
action_feature_dim
action_feature_schema_version
```

因此 `--resume` 只能使用 Trainer 生成的完整 checkpoint，不能使用只含模型参数的文件。

## 修改后的最低验证标准

- 修改 Rust bridge、候选特征或网络输入：重新构建 Rust 扩展，并运行 `python -m pytest python/tests -q`。
- 修改 loss、`Sample` 或训练 batch：运行全部 Python 测试，并用小规模 bootstrap smoke 验证。
- 修改 Rust 搜索 callback：至少确认 `test_mcts_selfplay.py` 通过，且 smoke benchmark 能完成。
