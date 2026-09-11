# Python AI 当前架构与操作手册

本文只描述当前保留在 `python/` 的代码。

## 职责边界

Rust 扩展 `brass_ai._engine` 是游戏规则与动作语义的唯一权威，负责：

- `GameState`、回合推进、合法动作与 canonical action；
- 状态 token 与动作引用的编码；
- heuristic teacher；
- candidate-policy-guided ISMCTS 搜索树。

Python 不实现规则或解析动作。它负责网络前向、训练样本组织、模仿训练、Rust 搜索的网络回调，以及自对弈支持代码。

## 当前数据流

```text
Rust GameState
  -> legal_candidates() / heuristic_candidates()
  -> Python PolicyValueNet(state tokens, action references)
  -> candidate logits + Q(s,a) + 4-player value + winner + econ
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

逐字段的定义与形状见 [ai-action-encoding.md](./ai-action-encoding.md)，那里是唯一契约：

| 数据 | 形状/含义 |
| --- | --- |
| 状态 token | cells `(49,F_CELL)`、links `(39,F_LINK)`、merchants `(9,F_MERCHANT)`、seats `(4,F_SEAT)`、global `(F_GLOBAL,)`，按行动方视角旋转 |
| 动作引用 | `float32 (N, ACTION_FEATURE_DIM)`，动作类型 + 实体引用 + 少量标量 |
| policy target | 对这 `N` 个候选归一化的分布 |
| value target | 每座位终局效用 `(vp - 桌均 VP) / VP_SCALE` `(4,)`，相对座位序 |
| winner target | 唯一冠军（VP→收入→现金破平局后第一名）的 one-hot `(4,)`，相对座位序 |
| econ target | `(income_level, money)`，按时代拆分的辅助监督 |

Python adapter 与 checkpoint 会拒绝未知 schema（`ACTION_SCHEMA_VERSION`、
`STATE_TOKEN_SCHEMA_VERSION`）；Rust 修改编码时必须同步更新这些位置与测试。

## 当前目录

```text
python/
|- bootstrap_imitation.py     heuristic imitation warm-start 入口
|- selfplay_train.py          长期 self-play 训练入口（阶段 3）
|- bench_value_ranking.py     Q(s,a) 与 V(s) 动作/价值排序体检探针
|- inspect_ckpt.py            模型权重、Schema 兼容性与元数据检查工具
|- brass_ai/
|  |- hierarchical_policy.py  Rust 候选动作和 teacher adapter
|  |- net.py                  PolicyValueNet
|  |- rust_mcts.py            Rust 搜索的 Python 网络回调
|  |- selfplay.py             Sample、imitation 与 MCTS self-play
|  |- selfplay_loop.py        自对弈循环：replay window、对手池、arena、指标
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
# Rust 侧（规则、搜索、bridge）有修改时先重装扩展
& .\.venv\Scripts\python.exe -m maturin develop --release --features python

# Python 当前回归测试
& .\.venv\Scripts\python.exe -m pytest python/tests -q

# Rust 回归测试（nn_mcts 在 python feature 下，必须带 --features python；
# pyo3 需要知道解释器位置，否则报 "no Python 3.x interpreter found"）
$env:PYO3_PYTHON = "$PWD\.venv\Scripts\python.exe"
cargo test --features python
```

三层验证的推荐顺序：先跑上面两个测试套件，再跑一次
[self-play 冒烟](#self-play-训练入口阶段-3)（单进程、`--sims 8`、2 轮），
最后才放正式规模的训练。冒烟能覆盖扩展加载、自对弈采样、snapshot 物化、
训练一步、arena 与 checkpoint 写出这条完整链路。

## 当前可运行入口

### Heuristic imitation bootstrap

`bootstrap_imitation.py` 是保留的训练入口。Rust heuristic 进行完整对局，Python 用每一步的 teacher 候选集训练候选评分、value / winner / Q 与经济辅助头，再让网络引导 Rust MCTS 与 heuristic 对战。

```powershell
& .\.venv\Scripts\python.exe python/bootstrap_imitation.py `
  --ckpt checkpoints/bootstrap-v6.pt --games 2000 --epochs 10 --batch 256 `
  --workers 12 --materialize-workers 8 --eval-games 0
```

训练预算按**优化器步数**算，不是按 epoch：`games × 每局决策点 / batch` 才是步数。
200 局约 2.5 万样本、只有约 194 步，基本等于没训；2000 局约 25 万样本，每个 epoch
约 970 步、单卡约 5 分钟。先跑 `--games 200 --epochs 2` 确认链路，再上正式规模。

| 参数 | 用途 |
| --- | --- |
| `--ckpt` | checkpoint 输出路径（默认 `checkpoints/bootstrap.pt`） |
| `--games` | heuristic 对局数（默认 1000） |
| `--epochs` | 每个 replay shard 的训练轮数 |
| `--workers` / `--materialize-workers` | imitation 生成进程数 / 训练时 snapshot 候选物化进程数（默认 8，受系统可用内存约束，内存紧张时调低）；`1` 为串行 |
| `--batch` / `--max-candidate-batch` | 每次读取的样本数 / 一个训练 micro-batch 的候选行预算（限制 padding 造成的显存峰值） |
| `--lr` | AdamW 学习率 |
| `--eval-games` / `--eval-sims` | 结尾 benchmark 的对局数（默认 20）/ 每步模拟数（默认 60） |
| `--min-avg-vp` / `--min-vp` / `--max-attempts` | 样本质量门槛：整局平均 VP / 最差座位 VP 下限，及启用门槛后的对局尝试上限 |
| `--resume` | 从 `--ckpt` 恢复完整 Trainer 状态（模型/optimizer/scheduler/scaler） |
| `--sample-dir` | 复用已有 `imitation-*.pkl`，跳过重新生成 |
| `--delete-samples-on-success` | 成功结束后删除默认的 `<ckpt>.imitation` 样本目录 |
| `--enable-policy-eval` | 训练后统计全部 shard 上的 top-k policy 指标 |
| `--mcts-shortlist` | 仅让结尾 benchmark 使用 heuristic shortlist；默认 full-legal |

`--max-candidate-batch` 是**填充后的候选行数**上限，默认 `65536`。显存占用与它近似
线性（约 1.8 GB @ 65536，含 AMP），放不下时先降它（`32768` / `16384`）。
`--materialize-workers` 影响的是主机内存：Windows 下每个 worker 都是独立进程，
各自 import torch，8 个大约 2~3 GB RSS。

### Self-play 训练入口（阶段 3）

`selfplay_train.py` 是长期自对弈训练入口，`brass_ai/selfplay_loop.py` 是它的循环实现。
它从 imitation checkpoint 热启动：先用 heuristic 教师给出可用先验，再用 MCTS visit
分布作为 policy 目标逐步替代 teacher。

```powershell
# 冒烟：单进程、极小搜索，几分钟内跑完并写出 checkpoint
./.venv/Scripts/python.exe python/selfplay_train.py `
  --ckpt-dir checkpoints/selfplay-smoke --init-from checkpoints/bootstrap-v6.pt `
  --iterations 2 --games-per-iter 2 --workers 1 --sims 8 --train-samples 512 `
  --eval-every 1 --eval-games 2 --eval-sims 8 --heuristic-eval-games 2 --heuristic-eval-sims 8

# 正式起点（16 核 + 1 GPU 量级）
./.venv/Scripts/python.exe python/selfplay_train.py `
  --ckpt-dir checkpoints/selfplay --init-from checkpoints/bootstrap-v6.pt `
  --iterations 200 --games-per-iter 16 --workers 8 --sims 128 `
  --train-samples 40000 --batch 256 --buffer-samples 400000 --buffer-iterations 20 `
  --eval-every 5 --eval-games 40 --eval-sims 128

# 中断后续训（参数需与首次运行一致或显式重传）
./.venv/Scripts/python.exe python/selfplay_train.py --ckpt-dir checkpoints/selfplay --resume ...
```

每轮写出的产物：

```text
<ckpt-dir>/latest.pt      完整 Trainer 状态（model/optimizer/scheduler/scaler/schema）
<ckpt-dir>/latest.json    最近完成轮次的编号与摘要，--resume 用它定位轮次
<ckpt-dir>/best.pt        arena 达标时刷新的参考模型
<ckpt-dir>/metrics.jsonl  每轮一行的 IterationStats（loss、耗时、胜率、搜索自检计数）
```

`latest.pt` 与 `best.pt` 都是 Trainer checkpoint，含 `model` 字段，可直接作为
`python -m brass_ai.replay_worker --ckpt <path>` 的网络座位权重使用。
循环每一轮做什么、各模块如何衔接见 [architecture.md](architecture.md) 的「训练循环现状」。

| 参数 | 用途 |
| --- | --- |
| `--init-from` / `--resume` | 从 imitation checkpoint 热启动 / 从 `<ckpt-dir>/latest.pt` 续训 |
| `--iterations` / `--games-per-iter` / `--sims` | 训练轮数 / 每轮对局数 / 每步模拟数 |
| `--workers` | actor 进程数；`1` 为单进程，调试与冒烟用 |
| `--mm-prob` / `--pool-size` | 对手池 matchmaking 概率 / 保留的历史 checkpoint 数量 |
| `--buffer-samples` / `--buffer-iterations` | replay window 的双重上限（样本数 / 轮数） |
| `--recent-fraction` / `--recent-iterations` | 每轮采样中新数据的占比 / "新"的轮数窗口 |
| `--train-samples` | 每轮从 replay window 抽多少样本训练（训练预算的主旋钮） |
| `--eval-every` / `--eval-games` / `--eval-sims` | arena 与 heuristic benchmark 的间隔与规模；`0` 关闭评估 |
| `--heuristic-eval-games` / `--heuristic-eval-sims` | ending benchmark 对 heuristic 的规模 |
| `--promote-winrate` | 刷新 `best.pt` 所需的 arena 胜率阈值（默认 0.55） |
| `--prior-top-k` / `--c-puct` / `--no-fpu` / `--no-q-init` | 搜索分支控制，见下 |
| `--max-depth` / `--mcts-batch` / `--candidate-k` | Rust ISMCTS 参数；`--candidate-k 0` 为 full-legal |

### 价值头兄弟排序基准

```powershell
& .\.venv\Scripts\python.exe python/bench_value_ranking.py --ckpt checkpoints/bootstrap-v6.pt --positions 36
```

以终局 rollout 为参照，测 `V(s)` / `Q(s,a)` / 先验 / winner 概率对候选动作的排序能力
（within-position Spearman）。这是把 Q 接进搜索的 go/no-go：Q 的相关性必须显著高于 V，
否则"用 Q 初始化未访问孩子"没有意义。参照量是行动方自己的终局效用，与网络同一尺度。

搜索配置的默认值与理由：

- `--q-init`（默认开启，用 `--no-q-init` 关闭）：用网络给出的边价值 `Q(s,a)` 估计从未
  访问过的孩子。本作单状态有 114–600 个合法动作，只按先验排序时 P ≈ 1/350，PUCT 的
  探索项量级压不过价值差，搜索会退化成先验的弱锐化器；用 Q 初始化之后兄弟招是按模型
  排序的。**这是全合法展开能否可搜索的关键**，也因此 `--prior-top-k` 不再是必需品。
- `--no-q-init` 之后才会退回 FPU：`--no-fpu` 关闭 FPU（默认开启）时未访问孩子按 0 处理。
  终局效用是零均值的 VP 差，取 0 已是中性假设；FPU 则把它初始化为父节点价值。
- `--prior-top-k 16` / `--c-puct 1.0`：仍然先把全部合法动作打分、只保留先验最高的 K 个
  孩子，用于压缩搜索宽度。这两个参数耦合，K 越小先验越尖、c 就该越小；`--prior-top-k 0`
  退回全合法展开。**当前默认值是旧尺度下定的，还没有在新价值尺度上重新标定**，
  见 [roadmap.md](roadmap.md) 的「近期」第 3 条。

已知边界：

- replay window 不写盘。`--resume` 只恢复模型/优化器与 best 参考，buffer 从空开始重新积累。
- `metrics.jsonl` 是追加写的，resume 不会截断它；按 `iteration` 去重即可。
- 自对弈对局超过 `--max-moves` 会被丢弃（没有合法终局就没有 value 目标），
  单局丢弃不会中止整轮；丢弃数反映在样本数与本轮对局数的差值上。

入口必须保留的几条既有约定：

- **观测与推理一致**：`SelfPlayConfig.determinize_observation`（默认开启）在采集样本时先对真实状态
  做一次 determinization 再编码，对手手牌来自合法隐藏牌池而不是模拟器的真实手牌；否则训练输入会
  包含推理时看不到的信息。己方手牌、公共盘面、市场与行动历史不受影响。
- **每局唯一种子**：`play_batch` 在 `SelfPlayConfig.seed` 给定时按 `seed + game_id` 派生每局种子；
  `selfplay_loop` 进一步按 `seed + iteration * seed_stride` 错开每一轮，避免整批对局复用同一发牌。
- **搜索自检**：`play_game_with_roles(..., stats=...)` 会回填 `failed_applies` / `rewritten_applies` /
  `moves`；`SelfPlayPool.last_diagnostics` 汇总 worker 侧同名计数。它们度量搜索树跨 determinization
  复用的代价：节点里的 `ResolvedMove` 保存的是手牌下标，而每次 simulation 都会重采样手牌，
  所以下标可能指向另一张牌。引擎按**语义卡牌**重新绑定下标（`rebind_cards`），因此这一手永远
  支付它被枚举时选的那张牌——`rewritten_applies` 计数的是发生了重绑定的 simulation
  （衡量复用规模，不是错误），`failed_applies` 计数的是当前 determinization 下根本执行不了、
  被剪枝的分支（例如手牌里已无等价卡，或资源/连通性变了）。
- **样本形态**：`SelfPlayConfig.store_snapshots`（默认开启）让自对弈样本只保存
  determinize 后的 snapshot 与稀疏 `canonical -> visit`，训练前由 `materialize_sample`
  还原成密集张量。密集形态每个决策点约 `N*55` 个 float32（N=300 时约 66 KB），
  不压缩成 snapshot 就无法让 replay window 跨轮存在。设为 `False` 则退回密集样本，
  仅在对照实验与单元测试中使用。

## 样本与 checkpoint

`Sample` 代表一个决策点。imitation 训练统一使用 full-legal：样本只保存 Rust state snapshot + teacher canonical action（不保存状态张量），候选集与状态张量在训练前从 snapshot 实时物化。候选上的监督包括 policy 分布、value/winner 终局目标与 econ 目标。

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
