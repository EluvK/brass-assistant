# Rust 引擎实验工具

本文件描述 `engine/src/bin/` 中保留的开发与实验入口。它们服务于规则回归、启发式策略调优和 MCTS 性能/参数分析，不是生产服务接口。

所有命令从仓库根目录运行。建议在需要稳定结果时使用 `--release`；`seed` 相同且参数相同的命令可复现随机基线对局。

## 保留入口

```
engine/src/bin/
├─ replay.rs        # 单局中文回放、摘要诊断或 heuristic 决策追踪
├─ replay_web.rs    # 内存回放与浏览器决策诊断
├─ sweep_scores.rs  # 批量 seed 扫描，输出 CSV
└─ mcts_lab.rs      # MCTS 基准、局面检查、参数扫描
```

## `replay`：单局诊断

```sh
cargo run --release -p brass-engine --bin replay -- <seed> <players> [policy] [sims] [canal-only] [full|summary] [trace] [candidate-k] [max-moves]
```

常用命令：

```sh
# 完整回放（trace 默认关闭）
cargo run --release -p brass-engine --bin replay -- 7 4 heuristic

# 仅输出每个时代的结算、动作汇总和终局
cargo run --release -p brass-engine --bin replay -- 7 4 heuristic 300 false summary

# 开启 trace，输出每个启发式决策的候选评分
cargo run --release -p brass-engine --bin replay -- 7 4 heuristic 300 false summary trace

# 只运行到运河时代结算前，用于检查运河局面
cargo run --release -p brass-engine --bin replay -- 7 4 heuristic 300 canal summary

# 输出 heuristic 决策的候选评分；仅追踪前 20 步
cargo run --release -p brass-engine --bin replay -- 42 4 heuristic 300 false summary trace 10 20
```

`policy` 可为 `heuristic`（默认）、`mcts`、`random`、`mcts-vs-random` 或 `mcts-vs-heur`。混合策略中 MCTS 座位由 `seed % players` 决定。`sims` 仅用于含 MCTS 的策略（默认 `300`）。

最后一个参数为 `summary` 时，隐藏逐动作的盘面、手牌和商家日志，但保留时代结算、各玩家动作统计、翻面板块和终局排名。

`trace` 位默认关闭，传 `trace`（或 `1`/`true`）开启：每次启发式决策前输出按分数降序排列的 `candidate_actions_k` 结果、2-ply 最终选择和卡牌保留分；使用 `full` 模式时还会输出实际执行结果。`candidate-k` 默认为 `30`，并沿用 `candidate_actions_k` 的语义：每类 Build/Network 动作最多保留该数量，因此总候选数可能更大。`max-moves` 默认为 `200000`。对于 `mcts-vs-heur`，trace 只输出启发式座位的决策。

## `replay-web`：浏览器回放与诊断

```sh
cargo run --release -p brass-engine --bin replay_web -- --seed 7 --players 4 \
  --player heuristic --player heuristic --player mcts --player random
```

命令启动后访问终端打印的 `http://127.0.0.1:8787/`。页面初始暂停，可单步、连续运行、暂停刷新和跳转已生成的时间线步骤。会话完全在 CLI 进程内存中，关闭浏览器或按 `Ctrl+C` 后不会留下回放文件。

不传 `--player` 时，所有座位默认使用 heuristic；`--player` 数量少于 `--players` 时，剩余座位同样补 heuristic。`--sims` 控制 MCTS 每步 simulation 数（默认 `500`）；`--port` 改变 loopback 端口。座位策略可为 `heuristic`、`random`、`mcts` 或 `python:<worker-config>`（网络 checkpoint 座位）。

网络座位示例（先确保 `.venv` 中已安装 `brass_ai._engine` 扩展）：

```sh
cargo run --release -p brass-engine --bin replay_web -- --seed 7 --player "python:--ckpt checkpoints/<name>.pt --sims 200 --device cpu"
```

`python:` 之后的参数原样传给 `python -m brass_ai.replay_worker`：`--ckpt` 为训练 checkpoint（必填），`--mode mcts|policy`（默认 `mcts`，前者为 Rust ISMCTS + 网络引导并按根访问数 argmax，后者为网络对全部合法候选一次前向后直接 argmax），`--sims`（默认 `128`）、`--device`（默认 cuda 可用则 cuda）。worker-config 按空白切分，含空格的参数（如 checkpoint 路径）可用单/双引号包裹。会话启动时即加载 checkpoint 并等待 worker 握手，加载失败会直接报错退出；每步决策受 `--worker-timeout`（默认 `300` 秒）约束，`--python-bin` 可指定解释器（默认 `python`）。网络座位的动作表会展示根访问次数、策略概率与当前玩家价值估计（`net-mcts` 证据），网络座位不做确定性承诺。

## `sweep_scores`：批量评测

```sh
cargo run --release -p brass-engine --bin sweep_scores -- <start_seed> <end_seed> <policy> [sims] [full|canal] > out.csv
```

```sh
# 评测 0..499 共 500 局完整对局；CSV 写入文件，摘要写入 stderr
cargo run --release -p brass-engine --bin sweep_scores -- 0 500 heuristic 200 full > scores.csv

# 只评测运河时代
cargo run --release -p brass-engine --bin sweep_scores -- 0 500 heuristic 200 canal > canal.csv
```

`full` 是默认范围。CSV 包含各座位终局 VP、整局平均 VP、终局收入/现金、运河结束时收入和每局 `elapsed_us`。`canal` 会在运河时代清理后停止，CSV 改为输出运河 VP、胜者座位、各类行动次数、翻面建筑数、链接数和每局耗时。stderr 摘要输出 winner/player/game 的均值与方差、唯一赢家率、每局耗时均值与方差，以及 illegal/stuck 计数。

注意：`policy` 仅支持 `heuristic` 与 `mcts`，其他取值会报错退出。

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

三个子命令都先以启发式策略推进到指定中局：`bench` 固定推进 60 步；`inspect` 的第 4 个参数为推进步数（默认 60）；`sweep` 固定 60 步。设置环境变量 `BRASS_MCTS_TWO_PLY`（任意非空值）可让 `inspect` 使用实验性的 `RootTwoPly` 叶节点估值（不影响 `bench`/`sweep`）；其余配置使用 `MctsConfig::default()`，仅由子命令覆盖需要扫描的参数。
