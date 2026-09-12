# Rust 引擎实验工具

本文件描述 `engine/src/bin/` 中保留的开发与实验入口。它们服务于规则回归、启发式策略调优以及引擎热路径 / NN-MCTS 数据链路性能分析，不是生产服务接口。

所有命令从仓库根目录运行。建议在需要稳定结果时使用 `--release`；`seed` 相同且参数相同的命令可复现随机基线对局。

## 构建边界

默认构建是纯 Rust 引擎：规则、启发式策略、单局文本回放和批量实验都不解析或链接 Python 相关依赖。需要时加上 `--features python` 启用 Python 相关功能。

`train_bench` 是该 feature 的专属二进制，运行时须带上 `--features python`。`replay_web` 默认可运行 heuristic/random 座位；只有座位配置含 `python:` 时才须带上 `--features python`。普通 `replay` 同样不依赖 Python。

跑引擎相关调试时建议使用 `--profile fast-release` 可以加速编译过程中的链接速度，仅在生成环境（使用python训练）时才建议使用 `--release`。

## 保留入口

```
engine/src/bin/
├─ replay.rs        # 单局中文回放、摘要诊断或 heuristic 决策追踪
├─ replay_web.rs    # 内存回放与浏览器决策诊断
├─ sweep_scores.rs  # 批量 heuristic seed 扫描，输出 CSV
└─ gen_imitation.rs # 纯 Rust 高并发百万级模仿学习分片生成器
```

`engine/src/bin/` 不再包含独立的 MCTS 实验台；NN-MCTS 的胜率/性能基准在
Python 端提供（见下文「NN-MCTS 基准」）。

## `replay`：单局诊断

```sh
cargo run --profile fast-release -p brass-engine --bin replay -- <seed> <players> [policy] [canal-only] [full|summary] [trace] [candidate-k] [max-moves]
```

> planned as deprecated 文字回放模式，用于单局对局的详细诊断。建议使用下面的 web 回放模式。

## `replay-web`：浏览器回放与诊断

```sh
cargo run --profile fast-release --bin replay_web -- --seed 7 --players 4
```

命令启动后访问终端打印的 `http://127.0.0.1:8787/`。页面初始暂停，可单步、连续运行、暂停刷新和跳转已生成的时间线步骤。会话完全在 CLI 进程内存中，关闭浏览器或按 `Ctrl+C` 后不会留下回放文件。

不传 `--player` 时，所有座位默认使用 heuristic；`--player` 数量少于 `--players` 时，剩余座位同样补 heuristic。`--port` 改变 loopback 端口。座位策略可为 `heuristic`、`random` 或 `python:<worker-config>`（网络 checkpoint 座位）；`--sims` 只存在于 `python:` 座位配置内部，控制 NN-MCTS 每步 simulation 数。

网络座位示例（先确保 `.venv` 中已安装 `brass_ai._engine` 扩展）：

```sh
cargo run --profile fast-release --features python --bin replay_web -- --seed 7 --player "python:--ckpt checkpoints/bootstrap-v6.pt --sims 1000"
```

checkpoint 必须是与当前 schema 匹配的训练产物（用 `python/bootstrap_imitation.py`
或 `selfplay_train.py` 生成）；旧 checkpoint 会因 schema 不符直接拒绝加载。

`python:` 之后的参数原样传给 `python -m brass_ai.replay_worker`：`--ckpt` 为训练 checkpoint（必填），`--mode mcts|policy`（默认 `mcts`，前者为 Rust ISMCTS + 网络引导并按根访问数 argmax，后者为网络对全部合法候选一次前向后直接 argmax），`--sims`（默认 `128`）、`--device`（默认 cuda 可用则 cuda）。worker-config 按空白切分，含空格的参数（如 checkpoint 路径）可用单/双引号包裹。会话启动时即加载 checkpoint 并等待 worker 握手，加载失败会直接报错退出；每步决策受 `--worker-timeout`（默认 `300` 秒）约束，`--python-bin` 可指定解释器（默认 `python`）。网络座位的动作表会展示根访问次数、策略概率与当前玩家价值估计（`net-mcts` 证据），网络座位不做确定性承诺。

## `sweep_scores`：批量评测

```sh
cargo run --release -p brass-engine --bin sweep_scores -- <start_seed> <end_seed> [policy] [full|canal] > out.csv
```

```sh
# 评测 0..499 共 500 局完整对局；CSV 写入文件，摘要写入 stderr
cargo run --release -p brass-engine --bin sweep_scores -- 0 500 heuristic full > scores.csv

# 只评测运河时代
cargo run --release -p brass-engine --bin sweep_scores -- 0 500 heuristic canal > canal.csv
```

`full` 是默认范围。CSV 包含各座位终局 VP、整局平均 VP、终局收入/现金、运河结束时收入和每局 `elapsed_us`。`canal` 会在运河时代清理后停止，CSV 改为输出运河 VP、胜者座位、各类行动次数、翻面建筑数、链接数和每局耗时。stderr 摘要输出 winner/player/game 的均值与方差、唯一赢家率、每局耗时均值与方差，以及 illegal/stuck 计数。

注意：`policy` 仅支持 `heuristic`，其他取值会报错退出。

该工具固定为四人局，并使用 Rayon 并行扫描。批量结果用于策略回归比较时，应固定 seed 区间、策略及范围。

## NN-MCTS 基准

Rust 侧不再有独立的 MCTS 实验台。网络引导 NN-MCTS 的决策基准在 Python 端：

- `python/brass_ai/evaluate.py`：`benchmark_mcts_vs_heuristic(mcts, sims, games)`
  或 `benchmark_net_vs_heuristic(net, sims, games)` 以轮换座位跑
  NN-MCTS vs heuristic 对抗并输出胜率/VP。
- `bootstrap_imitation.py` 的结尾 benchmark（`--eval-games` / `--eval-sims`）
  对训练出的 checkpoint 执行同一对抗。

引擎侧热路径（含共享 `determinize`）耗时用 [`train_bench`](#train_bench训练数据热路径基准)；
批量 heuristic 对局回归用 `sweep_scores`。

## `train_bench`：训练数据热路径基准

```sh
cargo run --release -p brass-engine --features python --bin train_bench -- [positions] [seed]
```

以启发式对局收集 `positions` 个中局快照，逐项输出训练管线使用的引擎操作耗时：合法动作枚举（`legal_resolved_moves`）、动作引用编码（`encode_move` × 全部候选）、状态 token 编码（`state_tokens`）、快照序列化/恢复、determinize、整状态 clone、教师打分（`candidate_actions_k(4)`）与 2-ply/末位四联动 `choose_action`。用于评估引擎侧改动对训练数据生成（imitation 生成 / snapshot 物化 / NN-MCTS 展开）的影响。

对应的 Python 侧跨界基准是 `python/bench_train_paths.py`（`legal_candidates` numpy 化、`materialize_snapshot` 单次调用端到端、`coalesce_equivalent_policy`），运行方式：`.venv/Scripts/python.exe python/bench_train_paths.py [n_positions]`。

## `gen_imitation`：高并发百万级专家数据生成器

纯 Rust 原生多线程生成工具，彻底替代过去基于 Python 多进程的缓慢导出。直接基于 Rayon 多核并行调度启发式 AI 批量对战，应用质量门禁，并以紧凑二进制形式（bincode 序列化）流式写入 `.bin` 分片。

```sh
cargo run --release -p brass-engine --bin gen_imitation -- [OPTIONS]
```

### 参数说明
- `--games, -n <N>`：目标接收的有效对局数（默认 `1000`）。
- `--out-dir, -o <DIR>`：分片产物输出目录（默认 `data/imitation_shards`）。
- `--min-vp <VP>`：局内单人最低 VP 质量门禁（默认 `0.0`，推荐设为 `30.0` 过滤破产局）。
- `--min-avg-vp <VP>`：全局平均 VP 质量门禁（默认 `0.0`）。
- `--shard-size <STEPS>`：单分片步数容量（默认 `32768` 步/分片，约 74 MB）。
- `--seed <SEED>`：起始随机种子（默认 `1000000`）。

### 性能表现与产物
- **生成吞吐**：单机实测达 **~70 局/秒**（生成 10,000 局仅耗时约 2 分 24 秒，产出 124 万步标准训练样本）。
- **生成产物**：切分为 `imitation-000000.bin` ~ `imitation-*.bin`，Python 端可通过 `brass_ai.selfplay.load_bin_shard(path)` 在 0.04 秒内瞬间反序列化完成。
- **目标标签**：自动打上相对终局价值 `value`（四人零和）、绝对标准化得分 `abs_vp`（$(VP - 100) / 50$）、终局冠军 `winner` 与时代经济指标 `econ`。

