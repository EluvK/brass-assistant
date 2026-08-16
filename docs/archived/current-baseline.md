# 当前可信基线（2026-08-02，调优后复跑）

> 本文档只保留可复现、仍有效的基线命令与结论。
> 调参方法论与判断准则见 `docs/heuristic-tuning-playbook.md`（推荐先读）。
> 历史调参细节归档到 `docs/handoff/0801.md`（仅历史参考）。

## 1. 适用范围

- 引擎目录：`src/engine`
- 玩家数：默认 4 人局
- 现状策略：heuristic 已加入阶段权重、潜在连边估值、铁路煤稀缺优先与显式计煤成本；
  2ply 已加入回合末流动性惩罚、末期 ALPHA 衰减；
  两者共用翻面率分级运河乘数、啤酒饱和度、建网防滥铺、研发行动护栏

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
| Avg final VP per player | `[60.104, 62.248, 61.056, 59.772]` |
| Avg VP/人 | `~60.8` |
| built / 局 | `20.6` |
| flipped / 局 | `17.4` |
| links / 局 | `26.5` |
| build / 局 | `35.7` |
| network / 局 | `45.1` |
| develop / 局 | `12.3` |
| sell / 局 | `9.3` |
| loan / 局 | `14.4` |
| pass / 局 | `8.4` |
| Avg final income / 人 | `~6.3` |

### 3.2 2-Ply（500 局）

命令：

```bash
cargo run --release --bin brass-engine -- 500 4 2ply
```

结果：

| 指标 | 数值 |
| --- | --- |
| Avg final VP per player | `[73.250, 73.712, 73.552, 73.218]` |
| Avg VP/人 | `~73.4` |
| built / 局 | `25.6` |
| flipped / 局 | `21.7` |
| links / 局 | `26.3` |
| build / 局 | `40.7` |
| network / 局 | `43.4` |
| develop / 局 | `11.1` |
| sell / 局 | `11.3` |
| loan / 局 | `16.3` |
| pass / 局 | `2.5` |
| Avg final income / 人 | `~8.2` |

## 4. 当前结论

- `2ply` 显著强于 `heuristic`（约 `+12.6 VP/人`）
- 本轮调优后两者均大幅提升：heuristic `~46.6 -> ~60.8`，2ply `~58.6 -> ~73.4`
- 2ply 分布健康：pass 降到 `2.5/局`，income `~8.2`；`seed_scores_2ply_0_499.csv` 中
  bottom-20 均值 `~40.5`，仍有约 90/500 局存在个别 <40 分选手（多为早期不可逆崩盘）
- 调参方法论见 `docs/heuristic-tuning-playbook.md`

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
