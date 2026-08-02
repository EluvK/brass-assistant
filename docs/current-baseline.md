# 当前可信基线（2026-08-02，已复跑）

> 本文档只保留可复现、仍有效的基线命令与结论。
> 历史调参细节统一归档到 `docs/handoff/0801.md`（仅历史参考）。

## 1. 适用范围

- 引擎目录：`src/engine`
- 玩家数：默认 4 人局
- 现状策略：heuristic 已加入阶段权重、潜在连边估值、铁路煤稀缺优先与铁路建网显式计煤成本

## 2. 正确性状态

- `cargo test --release` 通过
- 4 人局铁路时代为完整 8 轮
- `cargo run --release --bin replay -- 7 4 heuristic` 通过
- `cargo run --release --bin replay -- 7 4 2ply` 通过
- `cargo run --release --bin stat_game -- 7 4` 通过

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
| Avg final VP per player | `[48.192, 46.326, 46.084, 45.950]` |
| Avg VP/人 | `~46.6` |
| built / 局 | `18.0` |
| flipped / 局 | `13.4` |
| links / 局 | `21.4` |
| build / 局 | `37.2` |
| network / 局 | `40.4` |
| develop / 局 | `10.1` |
| sell / 局 | `8.7` |
| loan / 局 | `15.9` |
| pass / 局 | `12.7` |
| Avg final income / 人 | `~3.11` |

### 3.2 2-Ply（500 局）

命令：

```bash
cargo run --release --bin brass-engine -- 500 4 2ply
```

结果：

| 指标 | 数值 |
| --- | --- |
| Avg final VP per player | `[59.372, 58.302, 59.076, 57.782]` |
| Avg VP/人 | `~58.6` |
| built / 局 | `22.7` |
| flipped / 局 | `17.4` |
| links / 局 | `22.2` |
| build / 局 | `42.5` |
| network / 局 | `37.9` |
| develop / 局 | `9.5` |
| sell / 局 | `9.6` |
| loan / 局 | `17.8` |
| pass / 局 | `7.7` |
| Avg final income / 人 | `~4.61` |

## 4. 当前结论

- `2ply` 仍显著强于 `heuristic`（约 `+12.0 VP/人`）
- `heuristic` 相比旧基线有明显提升（`~39.9 -> ~46.6 VP/人`，翻面与收入均提升）
- 但与 `2ply` 仍有稳定差距，现阶段不建议作为教师策略上限

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
