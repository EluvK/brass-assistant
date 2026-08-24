# Rust 引擎实验工具

本文件描述 `engine/src/bin/` 中保留的开发与实验入口。它们服务于规则回归、启发式策略调优和 MCTS 性能/参数分析，不是生产服务接口。

所有命令从仓库根目录运行。建议在需要稳定结果时使用 `--release`；`seed` 相同且参数相同的命令可复现随机基线对局。

## 保留入口

```
engine/src/bin/
├─ replay.rs        # 单局中文回放、摘要诊断或 heuristic 决策追踪
├─ sweep_scores.rs  # 批量 seed 扫描，输出 CSV
└─ mcts_lab.rs      # MCTS 基准、局面检查、参数扫描
```

## `replay`：单局诊断

```sh
cargo run --release -p brass-engine --bin replay -- <seed> <players> [policy] [sims] [canal-only] [full|summary] [trace] [candidate-k] [max-moves]
```

常用命令：

```sh
# 逐动作的完整回放
cargo run --release -p brass-engine --bin replay -- 7 4 heuristic

# 仅输出每个时代的结算、动作汇总和终局，替代旧 stat_game
cargo run --release -p brass-engine --bin replay -- 7 4 heuristic 300 false summary

# 只运行到运河时代结算前，用于检查运河局面
cargo run --release -p brass-engine --bin replay -- 7 4 heuristic 300 canal summary

# 输出 heuristic 决策的候选评分；仅追踪前 20 步
cargo run --release -p brass-engine --bin replay -- 42 4 heuristic 300 false summary trace 10 20
```

`policy` 可为 `heuristic`（默认）、`mcts`、`random`、`mcts-vs-random` 或 `mcts-vs-heur`。混合策略中 MCTS 座位由 `seed % players` 决定。`sims` 仅用于含 MCTS 的策略。

最后一个参数为 `summary` 时，隐藏逐动作的盘面、手牌和商家日志，但保留时代结算、各玩家动作统计、翻面板块和终局排名。

传入 `trace` 会在每次启发式决策前输出当前手牌、按分数降序排列的 `candidate_actions_k` 结果和 2-ply 最终选择；使用 `full` 模式时还会输出实际执行结果。`candidate-k` 默认为 `30`，并沿用 `candidate_actions_k` 的语义：每类 Build/Network 动作最多保留该数量，因此总候选数可能更大。`max-moves` 默认为 `200000`。对于 `mcts-vs-heur`，trace 只输出启发式座位的决策。

## `sweep_scores`：批量评测

```sh
cargo run --release -p brass-engine --bin sweep_scores -- <start_seed> <end_seed> <policy> [sims] [full|canal] > out.csv
```

```sh
# 评测 0..499 共 500 局完整对局；CSV 写入文件，摘要写入 stderr
cargo run --release -p brass-engine --bin sweep_scores -- 0 500 heuristic 200 full > scores.csv

# 只评测运河时代，替代旧 sweep_canal
cargo run --release -p brass-engine --bin sweep_scores -- 0 500 heuristic 200 canal > canal.csv
```

`full` 是默认范围。CSV 包含终局 VP、终局收入/现金、运河结束时收入和每局 `elapsed_us`。`canal` 会在运河时代清理后停止，CSV 改为输出运河 VP、各类行动次数、翻面建筑数、链接数和每局耗时。stderr 摘要输出 winner/player/game 的均值与方差、唯一赢家率、每局耗时均值与方差，以及 illegal/stuck 计数。

该工具固定为四人局，并使用 Rayon 并行扫描。批量结果用于策略回归比较时，应固定 seed 区间、策略、simulation 数及范围。

## `mcts_lab`：MCTS 实验台

```sh
cargo run --release -p brass-engine --bin mcts_lab -- <bench|inspect|sweep> [参数]
```

```sh
# 不同 simulation 预算下的单次决策耗时和选择
cargo run --release -p brass-engine --bin mcts_lab -- bench 7 4 5000 10000

# 检查一个中局局面、MCTS/heuristic 的选择和根节点先验候选
cargo run --release -p brass-engine --bin mcts_lab -- inspect 7 4 2000 60

# 扫描既有的 depth / c_puct / 叶评估模式参数组合
cargo run --release -p brass-engine --bin mcts_lab -- sweep 7 4 2000
```

三个子命令都先以启发式策略推进到指定中局。`bench` 固定推进 60 步；`inspect` 与 `sweep` 的默认值分别为 60 步和 60 步。设置环境变量 `BRASS_MCTS_TWO_PLY=1` 可让 `inspect` 使用实验性的 `RootTwoPly` 叶节点估值；其余配置使用 `MctsConfig::default()`，仅由子命令覆盖需要扫描的参数。
