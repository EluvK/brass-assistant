# Rust 引擎实验工具

本文件描述 `engine/src/bin/` 中保留的开发与实验入口。它们服务于规则回归、启发式策略调优和 MCTS 性能/参数分析，不是生产服务接口。

所有命令从仓库根目录运行。建议在需要稳定结果时使用 `--release`；`seed` 相同且参数相同的命令可复现随机基线对局。

## 构建边界

默认构建是纯 Rust 引擎：规则、启发式策略、原生 MCTS、单局文本回放和批量实验都不解析或链接 Python 相关依赖。需要时加上 `--features python` 启用 Python 相关功能。

`train_bench` 是该 feature 的专属二进制，运行时须带上 `--features python`。`replay_web` 默认可运行 heuristic/random/mcts；只有座位配置含 `python:` 时才须带上 `--features python`。普通 `replay` 同样不依赖 Python。

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
cargo run --profile fast-release -p brass-engine --bin replay -- <seed> <players> [policy] [sims] [canal-only] [full|summary] [trace] [candidate-k] [max-moves]
```

> planned as deprecated 文字回放模式，用于单局对局的详细诊断。建议使用下面的 web 回放模式。

## `replay-web`：浏览器回放与诊断

```sh
cargo run --profile fast-release --bin replay_web -- --seed 7 --players 4
```

命令启动后访问终端打印的 `http://127.0.0.1:8787/`。页面初始暂停，可单步、连续运行、暂停刷新和跳转已生成的时间线步骤。会话完全在 CLI 进程内存中，关闭浏览器或按 `Ctrl+C` 后不会留下回放文件。

不传 `--player` 时，所有座位默认使用 heuristic；`--player` 数量少于 `--players` 时，剩余座位同样补 heuristic。`--sims` 控制 MCTS 每步 simulation 数（默认 `500`）；`--port` 改变 loopback 端口。座位策略可为 `heuristic`、`random`、`mcts` 或 `python:<worker-config>`（网络 checkpoint 座位）。

网络座位示例（先确保 `.venv` 中已安装 `brass_ai._engine` 扩展）：

```sh
cargo run --profile fast-release --features python --bin replay_web -- --seed 7 --player "python:--ckpt checkpoints/bootstrap-0831-20000.pt --sims 1000"
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

## `train_bench`：训练数据热路径基准

```sh
cargo run --release -p brass-engine --features python --bin train_bench -- [positions] [seed]
```

以启发式对局收集 `positions` 个中局快照，逐项输出训练管线使用的引擎操作耗时：合法动作枚举（`legal_resolved_moves`）、候选特征编码（`encode_move` × 全部候选）、状态张量编码（`state_to_tensor`）、快照序列化/恢复、determinize、整状态 clone、教师打分（`candidate_actions_k(4)`）与 2-ply `choose_action`。用于评估引擎侧改动对训练数据生成（imitation 生成 / snapshot 物化 / NN-MCTS 展开）的影响。

对应的 Python 侧跨界基准是 `python/bench_train_paths.py`（`legal_candidates` numpy 化、`materialize_snapshot` 单次调用端到端、`coalesce_equivalent_policy`），运行方式：`.venv/Scripts/python.exe python/bench_train_paths.py [n_positions]`。
