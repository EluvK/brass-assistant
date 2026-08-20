# Python AI 架构与操作手册

本文是 `src/ai/` 当前候选动作 Policy 架构的操作手册。它只描述已经迁移到 candidate policy 的主链路，并明确区分已验证和待验证流程。

## 权威边界

Rust 扩展 `brass_engine` 是规则和动作语义的唯一权威，负责：

- `GameState`、规则、回合推进和对手手牌确定化；
- 完整 legal concrete action 的生成；
- state tensor 编码；
- action feature 编码；
- Rust heuristic teacher；
- candidate-policy-guided ISMCTS。

Python 不实现规则或 canonical action 解析。它负责候选评分网络、样本编排、训练、checkpoint 和评测。

## 当前架构

```text
Rust GameState
  -> legal concrete candidates / action features
  -> PolicyValueNet(state, candidates) -> candidate logits + value
  -> Rust ISMCTS over full legal candidates

Rust heuristic teacher
  -> bounded scored shortlist (at most 14 candidates)
  -> imitation replay / soft policy target
```

MCTS 推理展开完整合法动作集合。为避免 replay 随动作组合空间爆炸，imitation 训练只持久化 teacher shortlist：top-4 Build、top-4 Network、其他每类最佳动作，必要时再加入 heuristic 的 2-ply 最终选择。当前每个 imitation 样本最多 14 个候选，而不是所有 legal actions。

## Rust-Python 契约

正式训练路径当前只支持四人局。虽然 `GameState` 支持 2--4 人，但网络 value head 固定为 4 维，手牌编码也固定为三名对手。

| 数据 | 契约 |
| --- | --- |
| board | `float32 (17, 49)` |
| links | `float32 (6, 39)` |
| global | `float32 (50,)` |
| own_hand | `float32 (35,)` |
| opp_hands | `float32 (105,)` |
| action feature | `float32 (N, 235)`，`N` 随状态变化 |
| feature schema | `ACTION_FEATURE_SCHEMA_VERSION = 2` |
| policy | 对同一状态的候选集合归一化，而非固定全局动作表 |
| value target | 四名玩家终局 VP 的标准化向量 `(vp - mean) / std`；平局为全零 |
| econ target | `(income_level, money)`；仅作辅助监督 |

Python 必须拒绝未知 action feature schema version。修改 Rust action encoding 时，应同时升级 schema version、Python adapter、replay 格式和相关测试。

## 目录

```text
src/ai/
|- brass_ai/
|  |- build_input.py          Rust state tensor -> PyTorch batch
|  |- hierarchical_policy.py  candidate batch 与 teacher bridge adapter
|  |- net.py                  candidate scorer + value/econ heads
|  |- rust_mcts.py            Rust MCTS candidate-logit callback
|  |- selfplay.py             self-play、teacher imitation、Sample
|  |- mp_selfplay.py          worker pool 与候选样本打包
|  |- dataset.py              candidate replay NPZ shard
|  `- train.py                Trainer、loss、candidate metrics
|- bootstrap_imitation.py     已验证的 imitation warm-start 入口
|- train_mp.py                多进程 self-play 训练入口，待验证
|- experiments/               历史实验脚本，多数待迁移
`- tests/                     当前主链路回归测试
```

## 环境准备

所有命令从仓库根目录运行，PowerShell 示例：

```powershell
# 使用项目虚拟环境，并使 Python 能发现 AI 包
.\src\engine\.venv\Scripts\Activate.ps1
$env:PYTHONPATH = "src/ai"

# Rust bridge 改动后重新构建和安装
Push-Location src/engine
python -m maturin develop --release
Pop-Location

# 当前 AI 主链路回归测试
python -m pytest src/ai/tests -q
```

上述安装和测试流程已验证。当前测试集为 15 项；它覆盖 candidate feature、teacher shortlist、candidate policy、replay、训练和 Rust MCTS adapter。

## 已验证流程

### 1. Heuristic Imitation Bootstrap

状态：已验证。

Rust heuristic 生成完整对局；每步返回有界 scored shortlist。Python 将 teacher scores softmax 为 policy target，训练 candidate scorer、value head 和 econ head。

```powershell
python src/ai/bootstrap_imitation.py --games 200 --epochs 3 --workers 4 --min-avg-vp 80 --min-vp 58 --max-attempts 20000 --ckpt checkpoints/bootstrap-smoke.pt
```

当前输出包含：

```text
policy / value
top1 / top3 / top5
type_top1
entropy
candidates mean / p95
MCTS vs heuristic benchmark
```

解释：top-k 是训练后的 teacher target 命中率；当前先在训练 replay 上计算，不能作为泛化结果。`candidates` 是 teacher shortlist 大小，不是 MCTS 的完整合法动作数。checkpoint 是模型 `state_dict`，不是完整 Trainer 恢复状态。

### 2. Rust 和 Python 回归

状态：已验证。

```powershell
Push-Location src/engine
cargo fmt --check
cargo check
cargo test --lib
Pop-Location

python -m pytest src/ai/tests -q
```

bridge、feature schema、candidate batch 或训练 loss 发生修改后，必须重跑这些检查。

## 待验证流程

以下入口在旧 flat policy 架构下存在过，但尚未完成 candidate-policy 端到端验证。
它们不应作为长期训练基线或结果依据。

| 入口 | 当前状态 | 需要先验证的内容 |
| --- | --- | --- |
| `train_mp.py` | 待验证 | worker pool、candidate replay shard、resume、manifest、GPU 主进程训练 |
| `bench_mp.py` | 待验证 | full-candidate MCTS 的 worker 吞吐、内存和 batch 行为 |
| `experiments/benchmark.py` | 待验证 | checkpoint 加载、candidate MCTS、固定 seed 对局结果 |
| `experiments/diagnose.py` | 待迁移 | 仍可能依赖旧 slot API |
| `experiments/replay_net.py` | 待迁移 | 仍可能依赖旧 slot API |
| `experiments/net_all_vs_all.py` | 待迁移 | 仍可能依赖旧 slot API |
| `experiments/bc_baseline.py` | 历史脚本 | 不属于当前训练主链路 |

验证顺序应为：先用 `train_mp.py` 跑 `1 worker / 1 game / 1 iteration / CPU`，确认 sample、shard、checkpoint 和 resume；再逐步增加 worker、game 数、simulation 数和 GPU 训练。不要直接启动旧文档中的长时间 self-play run。

## 训练产物

candidate replay shard 的每条样本包含：

```text
pid, era
board, links, global_vec, own_hand, opp_hands
candidates: (N, 235)
policy: (N,)
value: (4,)
econ: (2,)
```

`N` 是样本自己的候选数。落盘时为压缩 NPZ 的 padded array 加 mask；加载后恢复为变长 candidate 数组。旧的 `legal` mask 和固定 1316 维 policy 不属于当前格式。

训练中 `evaluate_policy()` 必须分 batch 计算。禁止把整个 replay 按全局最大候选数一次性 padding 后送入 GPU，否则候选 tensor 会造成极高峰值内存。

## 当前风险与下一步

- action feature v2 已将被消耗卡片编码为稳定语义；资源来源和 Sell 结构仍保留在 concrete action 特征中，后续再做 collision 统计。
- imitation shortlist 与 MCTS full legal candidates 存在分布差异；下一步应加入有限 数量的 hard negatives，而不是将完整候选集合写入 replay。
- policy top-k 当前是在训练数据上评估；需要建立固定 held-out teacher validation。
- value 是 final normalized VP 预测，不是校准后的 win probability。
- `train_mp.py` 与实验工具尚未在新架构下验证，使用前需先完成最小 smoke run。
