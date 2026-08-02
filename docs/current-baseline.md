# 当前可信基线（2026-08-02）

> 本文档只保留可复现、仍有效的基线命令与结论。
> 历史调参细节统一归档到 `docs/handoff/0801.md`（仅历史参考）。

## 1. 适用范围

- 引擎目录：`src/engine`
- 玩家数：默认 4 人局
- 现状策略：已启用多条 heuristic 约束（如禁造 1 级酒、禁研发 2+ 级铁）

## 2. 正确性状态

- `cargo test --release` 通过
- 4 人局铁路时代为完整 8 轮
- `replay` / `stat_game` 可正常复盘

当前测试规模：
- unit tests: `3`
- integration tests: `27`

## 3. 当前强度基线

### 3.1 Heuristic（500 局）

命令：

```bash
cargo run --release --bin brass-engine -- 500 4 heuristic
```

结果：

| 指标 | 数值 |
| --- | --- |
| Avg final VP per player | `[40.222, 39.442, 40.938, 38.944]` |
| Avg VP/人 | `~39.9` |
| built / 局 | `17.3` |
| flipped / 局 | `10.4` |
| links / 局 | `16.8` |
| build / 局 | `40.0` |
| network / 局 | `37.4` |
| develop / 局 | `11.9` |
| sell / 局 | `8.7` |
| loan / 局 | `15.9` |
| pass / 局 | `11.3` |
| Avg final income / 人 | `~1.66` |

### 3.2 2-Ply（500 局）

命令：

```bash
cargo run --release --bin brass-engine -- 500 4 2ply
```

结果：

| 指标 | 数值 |
| --- | --- |
| Avg final VP per player | `[48.972, 49.092, 48.766, 46.802]` |
| Avg VP/人 | `~48.4` |
| built / 局 | `21.5` |
| flipped / 局 | `13.3` |
| links / 局 | `16.3` |
| build / 局 | `44.4` |
| network / 局 | `36.0` |
| develop / 局 | `10.7` |
| sell / 局 | `10.5` |
| loan / 局 | `17.4` |
| pass / 局 | `6.4` |
| Avg final income / 人 | `~3.17` |

## 4. 当前结论

- 在当前约束配置下，`2ply` 仍明显强于 `heuristic`
- `heuristic` 仍存在运营效率问题（收入与翻面不足）
- 现阶段不建议把本轮分数作为深度学习起始教师上限

## 5. 推荐复跑命令

### 正确性

```bash
cargo test --release
cargo run --release --bin replay -- 7 4 heuristic
cargo run --release --bin replay -- 7 4 2ply
cargo run --release --bin stat_game -- 7 4
```

### 强度

```bash
cargo run --release --bin brass-engine -- 500 4 heuristic
cargo run --release --bin brass-engine -- 500 4 2ply
```

### MCTS 诊断（可选）

```bash
cargo run --release --bin bench_mcts -- 7 4 5000 10000
cargo run --release --bin sweep_mcts -- 7 4 2000
cargo run --release --bin debug_mcts -- 7 4 2000 60
```
