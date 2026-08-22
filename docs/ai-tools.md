# Python AI 架构与操作手册

本文是 `src/ai/` 当前 concrete-candidate 动作架构的操作手册。

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
  -> Rust ISMCTS over configured candidate set

Rust heuristic teacher
  -> bounded scored shortlist (at most 14 candidates)
  -> imitation replay / soft policy target
```

默认 MCTS 使用与 teacher 一致的 bounded shortlist（`candidate_k=4`）：top-4 Build、top-4
Network、其他每类最佳动作，必要时再加入 heuristic 的 2-ply 最终选择。设置
`candidate_k=0` 时恢复为完整 `legal_moves()`。imitation 默认持久化 teacher shortlist，每个样本最多
14 个候选；实验开关 `--full-legal-candidates` 会改为保存该状态的全部合法候选，并在 full candidate
集合上使用 heuristic 选择动作作为 one-hot target。

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
|- train_mp.py                多进程 self-play 训练入口
|- experiments/               可复用的评测、回放和诊断工具
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

上述安装和测试流程已验证。测试覆盖 candidate feature、teacher shortlist、candidate policy、replay、训练和 Rust MCTS adapter。

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

候选集对齐实验：

```powershell
# full legal candidates 训练 + full legal MCTS benchmark
python src/ai/bootstrap_imitation.py --games 100 --epochs 1 --workers 8 `
  --full-legal-candidates --mcts-full-legal `
  --ckpt checkpoints/bootstrap-full-legal-test.pt
```

`--full-legal-candidates` 会显著增加临时 replay shard 体积，建议先用 100 局检查 candidate count、
生成速度和磁盘占用。`--mcts-full-legal` 仅影响 bootstrap 末尾 benchmark；不加时 benchmark 使用默认
`candidate_k=4` shortlist。Rust extension 改动后必须重新运行 `maturin develop`。

full-legal 持久化实验结果（2026-08-22）：2 个 worker 可以运行，但 40 局已经产生约 2 GB
replay 数据；8 个 worker 在把完整 `Sample` 列表通过 multiprocessing queue pickle 回主进程时触发
`MemoryError`。原因是每个状态平均约 500 个合法候选，每个候选包含 235 维 `float32` 特征，单局
约 125 个状态，多个 worker 的结果会同时驻留内存。当前不应继续用该模式做正式 bootstrap。

为保留 full-legal 过渡实验能力，replay/进程传输现在对完整候选特征使用无损定点压缩：动作特征值均为
0.25 的整数倍，保存时乘 4 后转为 `uint8`，训练组 batch 时再还原为 `float32`。这只压缩存储格式，
不改变网络输入或 policy target；候选特征部分理论上减少约 75% 体积，并降低 worker 返回结果的峰值
内存。该格式只用于验证和短期实验，不是最终 replay 结构。

2026-08-22 进一步实验：1000 局 full-legal 样本生成成功（约 1325 秒），但训练第一 epoch 使用固定
`batch=256` 时在 8GB GPU 上 OOM。原因是 batch padding 到该批次最大候选数，候选矩阵和 action encoder
激活随 `batch_size * max_candidates * 235` 增长。训练器现已按候选数自动拆分 batch（默认限制候选行预算
为 16384），评估路径同样受限；bootstrap 失败时会保留临时 shard 并打印目录，不再被 finally 静默删除。

已有 shard 可用 `--sample-dir` 跳过生成并重复训练；该参数指向的目录永不由脚本删除。例如：

```powershell
python src/ai/bootstrap_imitation.py --sample-dir checkpoints/bootstrap-imitation-XXXXXX `
  --epochs 10 --mcts-full-legal --ckpt checkpoints/bootstrap-0822-full.pt
```

目录必须包含 `imitation-*.pkl`。传入 `--sample-dir` 后，`--games`、`--workers`、质量筛选和
`--full-legal-candidates` 都不参与生成；它们不会改变已落盘样本。

训练日志会显示 `train e<epoch>/<total> s<shard>/<total>`、当前 shard 的样本数和文件名。full-legal
实验应以该 shard 编号判断进度，不要将单个 shard 内部的 progress 重置误认为整轮重新开始。训练进度
按实际 GPU micro-batch 更新并以 2 秒限流刷新；候选预算导致的 batch 拆分会反映在进度和 ETA 中。

`--max-candidate-batch` 是 GPU batch 内的候选行预算，不是字节数：实际 batch 满足
`sample_count * max_candidates <= budget`。默认 `16384`；对于一个最大候选数约 600 的 batch，至多约
27 个 sample。8GB GPU 在默认值稳定后可按 `16384 -> 24576 -> 32768` 逐次提高，每次先跑一个 epoch；
一旦 OOM 就退回上一个值。GPU 利用率 100% 代表正在算，显存有余量时提高该值才可能减少 batch 拆分、提高吞吐。

bootstrap 默认跳过全 replay 的 policy 指标统计；这一步不更新模型，也不影响 checkpoint 或 MCTS
benchmark，但 full-legal 时会额外读取全部 shard 并执行一次完整前向。需要 top-k、entropy、候选数
等指标时显式传入 `--enable-policy-eval`。

同一轮排查还修复了两个 canonical 规范化问题：heuristic Sell 返回的 tile keys 已统一按升序排列，
Scout 返回的 card indices 已统一按升序排列；两者现在与 `legal_moves()` 的 concrete canonical
顺序一致。hard negatives 不作为正式演进方向：少量负样本无法覆盖全部未见合法动作，不能保证
full-legal MCTS 的分布对齐。最终应改为训练时动态生成候选，而不是把完整候选特征直接持久化到 replay。

动态 full-legal replay 实现状态（2026-08-22）：Rust bridge 已提供 `GameState.snapshot()` 与
`GameState.from_snapshot(bytes)`。snapshot 是版本化的不透明事件流，包含初始 seed、人数和成功执行的
engine-level 操作；恢复时由 Rust 从初始状态重放，因此完整保留隐藏手牌、牌堆、市场和所有规则相关状态，
不将 Python tensor 当作状态来源。`--full-legal-candidates` 的 imitation shard 现保存该 snapshot、teacher
canonical action 和 value/econ target；`bootstrap_imitation.py` 加载 shard 后调用 `materialize_samples()`，
恢复状态并实时请求 `legal_candidates()`，再以 teacher action 生成 full-legal one-hot policy。短名单旧 shard
保持原格式和训练路径，便于已有 baseline 复现。

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

## 工具入口

实验脚本均使用当前 concrete-candidate 接口：

| 入口 | 用途 |
| --- | --- |
| `experiments/benchmark.py` | 固定 seed 的 MCTS 对 heuristic 基准 |
| `experiments/replay_net.py` | 单局或多局中文逐步回放，可选 greedy/MCTS |
| `experiments/net_all_vs_all.py` | 四席位 network-vs-network 统计 |
| `experiments/diagnose.py` | 逐动作诊断时代得分、翻面和动作分布 |
| `experiments/bc_baseline.py` | 可选的 heuristic imitation 训练基线 |

实验脚本没有独立的稳定性承诺。修改 Rust bridge、动作特征或网络结构后，应先运行 `src/ai/tests`，再用小规模参数运行对应脚本；不要把单次实验结果当作训练 gate。

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

上述格式仍用于 shortlist replay 与旧 full-legal 实验 shard。新的 full-legal imitation replay 保存可恢复的
Rust `GameState` 快照、teacher canonical action、value/econ target；`candidates` 和 `policy` 的候选维度
数据在训练加载阶段由 Rust 从快照重新生成，不再作为长期持久化字段。

训练中 `evaluate_policy()` 必须分 batch 计算。禁止把整个 replay 按全局最大候选数一次性 padding 后送入 GPU，否则候选 tensor 会造成极高峰值内存。

## 当前风险与下一步

- action feature v2 已将被消耗卡片编码为稳定语义；资源来源和 Sell 结构仍保留在 concrete action 特征中，后续再做 collision 统计。
- 当前同时支持 shortlist 对齐实验和 full-legal candidate 实验。旧 full-legal shard 直接保存所有候选，
  已验证会造成 replay shard 体积和 worker IPC 内存不可接受；`uint8` 版本仅用于旧 shard 兼容。
- full-legal imitation 已改为可恢复 Rust `GameState` 快照：replay 只保存状态快照、teacher canonical
  action 和 value/econ target；训练时恢复状态并动态生成全部 legal candidates，推理继续使用 full-legal
  MCTS。仍需用流式 worker/micro-batch 控制训练时的 CPU 和显存开销，并完成 held-out/full-legal benchmark。
- 在动态候选训练完成并通过 held-out validation、full-legal benchmark 前，当前正式 bootstrap 仍使用
  bounded shortlist，不启动大规模 self-play。
- policy top-k 当前是在训练数据上评估；需要建立固定 held-out teacher validation。
- value 是 final normalized VP 预测，不是校准后的 win probability。
- `train_mp.py` 和实验工具依赖已安装的 Rust extension；使用前应完成最小 CPU smoke run。
