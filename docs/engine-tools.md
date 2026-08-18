# Rust 引擎实验工具

本文件描述 `src/engine/src/bin/` 中保留的开发与实验入口。它们服务于规则回归、启发式策略调优和 MCTS 性能/参数分析，不是生产服务接口。

所有命令从 `src/engine` 目录运行。建议在需要稳定结果时使用 `--release`；`seed` 相同且参数相同的命令可复现随机基线对局。

## 保留入口

```
src/engine/src/bin/
├─ replay.rs        # 单局中文回放或摘要诊断
├─ sweep_scores.rs  # 批量 seed 扫描，输出 CSV
└─ mcts_lab.rs      # MCTS 基准、局面检查、参数扫描
```

## `replay`：单局诊断

```sh
cargo run --release --bin replay -- <seed> <players> [policy] [sims] [canal-only] [full|summary]
```

常用命令：

```sh
# 逐动作的完整回放
cargo run --release --bin replay -- 7 4 heuristic

# 仅输出每个时代的结算、动作汇总和终局，替代旧 stat_game
cargo run --release --bin replay -- 7 4 heuristic 300 false summary

# 只运行到运河时代结算前，用于检查运河局面
cargo run --release --bin replay -- 7 4 heuristic 300 canal summary
```

`policy` 可为 `heuristic`（默认）、`mcts`、`random`、`mcts-vs-random` 或 `mcts-vs-heur`。混合策略中 MCTS 座位由 `seed % players` 决定。`sims` 仅用于含 MCTS 的策略。

最后一个参数为 `summary` 时，隐藏逐动作的盘面、手牌和商家日志，但保留时代结算、各玩家动作统计、翻面板块和终局排名。

## `sweep_scores`：批量评测

```sh
cargo run --release --bin sweep_scores -- <start_seed> <end_seed> <policy> [sims] [full|canal] > out.csv
```

```sh
# 评测 0..499 共 500 局完整对局；CSV 写入文件，摘要写入 stderr
cargo run --release --bin sweep_scores -- 0 500 heuristic 200 full > scores.csv

# 只评测运河时代，替代旧 sweep_canal
cargo run --release --bin sweep_scores -- 0 500 heuristic 200 canal > canal.csv
```

`full` 是默认范围。CSV 包含终局 VP、终局收入/现金和运河结束时收入。`canal` 会在运河时代清理后停止，CSV 改为输出运河 VP、各类行动次数、翻面建筑数和链接数。

该工具固定为四人局，并使用 Rayon 并行扫描。批量结果用于策略回归比较时，应固定 seed 区间、策略、simulation 数及范围。

## `mcts_lab`：MCTS 实验台

```sh
cargo run --release --bin mcts_lab -- <bench|inspect|sweep> [参数]
```

```sh
# 不同 simulation 预算下的单次决策耗时和选择
cargo run --release --bin mcts_lab -- bench 7 4 5000 10000

# 检查一个中局局面、MCTS/heuristic 的选择和根节点先验候选
cargo run --release --bin mcts_lab -- inspect 7 4 2000 60

# 扫描既有的 depth / c_puct / LeafEval 参数组合
cargo run --release --bin mcts_lab -- sweep 7 4 2000
```

三个子命令都先以启发式策略推进到指定中局。`bench` 固定推进 60 步；`inspect` 与 `sweep` 的默认值分别为 60 步和 60 步。设置环境变量 `BRASS_MCTS_TWO_PLY=1` 可让 `inspect` 使用 `TwoPly` 叶节点估值；其余配置使用 `MctsConfig::default()`，仅由子命令覆盖需要扫描的参数。
