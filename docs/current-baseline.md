# 当前可信基线（修复后）

> 本文档汇总 **当前可相信** 的 Brass 引擎基线结果。
> 这些数据建立在以下修复之后：
> - 铁路时代不再二次埋牌
> - 非法动作失败后不再推进回合
> - `Develop` 合法动作生成与可支付铁成本对齐
> - `Sell` 合法性判定与执行逻辑对齐
> - `MCTS` 的 `Sell` 动作身份包含 `use_merchant_beer`

## 1. 适用范围

- 时间：2026-08 本轮修复后
- 引擎目录：`src/engine`
- 玩家数：默认 4 人局
- 这些数字可用于：
  - 当前 AI 强度对比
  - 文档更新参考
  - 后续调参前后的回归比较

## 2. 已知失效的旧结论

- `docs/handoff/0801.md` 中的旧基线数字仅保留历史参考价值
- `docs/reference-notes/2ply-search.md` 与 `mcts-stage3.md` 里修复前的强度结论已失效
- 任何建立在错误“铁路时代二次埋牌”或“非法动作仍推进回合”之上的统计均不应再使用

## 3. 正确性状态

已确认：
- `cargo test --release` 通过
- 4 人局铁路时代为完整 `8` 轮
- `heuristic` / `2ply` / `mcts` / mixed 模式均可完整跑完当前复测集
- `random` / `mcts-vs-random` 不再出现“0 次时代切换”的假死现象

当前测试规模：
- unit tests: `3`
- integration tests: `16`

## 4. 当前强度基线

### 4.1 Heuristic（500 局）

命令：

```bash
cargo run --release --bin brass-engine -- 500 4 heuristic
```

结果：

| 指标 | 数值 |
| --- | --- |
| Avg final VP per player | `[60.146, 58.138, 58.02, 58.984]` |
| Avg VP/人 | `~58.8` |
| built / 局 | `17.8` |
| flipped / 局 | `12.6` |
| links / 局 | `21.4` |
| build / 局 | `40.9` |
| network / 局 | `39.4` |
| develop / 局 | `9.7` |
| sell / 局 | `10.1` |
| loan / 局 | `15.3` |
| pass / 局 | `7.8` |
| Avg final income / 人 | `~3.80` |

### 4.2 2-Ply（500 局）

命令：

```bash
cargo run --release --bin brass-engine -- 500 4 2ply
```

结果：

| 指标 | 数值 |
| --- | --- |
| Avg final VP per player | `[66.042, 64.37, 64.578, 63.616]` |
| Avg VP/人 | `~64.7` |
| built / 局 | `20.4` |
| flipped / 局 | `14.4` |
| links / 局 | `20.4` |
| build / 局 | `45.8` |
| network / 局 | `37.3` |
| develop / 局 | `7.5` |
| sell / 局 | `11.7` |
| loan / 局 | `16.8` |
| pass / 局 | `4.4` |
| Avg final income / 人 | `~4.56` |

结论：
- `2ply` 明显强于 `heuristic`
- 主要表现为：更多建厂、更多卖货、更少过牌、更高终局收入

### 4.3 MCTS vs Heuristic（80 局，16 线程，600 sims）

命令：

```bash
cargo run --release --bin brass-engine -- 80 4 mcts-vs-heur 16 600
```

结果：

| 指标 | 数值 |
| --- | --- |
| MCTS seat 胜率 | `27.5% (22/80)` |
| MCTS avg VP | `56.8` |
| heuristic avg VP | `57.5` |

结论：
- 当前配置下，`MCTS` 没有强过 `heuristic`

### 4.4 MCTS vs 2-Ply（80 局，16 线程，600 sims）

命令：

```bash
cargo run --release --bin brass-engine -- 80 4 mcts-vs-2ply 16 600
```

结果：

| 指标 | 数值 |
| --- | --- |
| MCTS seat 胜率 | `20.0% (16/80)` |
| MCTS avg VP | `57.6` |
| 2-ply avg VP | `63.3` |

结论：
- 当前配置下，`MCTS` 明显弱于 `2ply`

## 5. 当前总判断

当前 4 人局强度排序：

1. `2ply`
2. `heuristic` 与 `mcts` 接近，但当前 `mcts` 没占优
3. `random` 仅作弱基线/烟雾测试

因此：
- `2ply` 可以继续作为当前最强可用 baseline
- `MCTS` 目前更像“可运行研究原型”，不应再当作现阶段最强策略

## 6. 推荐复跑命令

### 正确性

```bash
cargo test --release
cargo run --release --bin replay -- 7 4 heuristic
cargo run --release --bin replay -- 7 4 2ply
cargo run --release --bin replay -- 7 4 mcts
cargo run --release --bin stat_game -- 7 4
```

### 强度

```bash
cargo run --release --bin brass-engine -- 500 4 heuristic
cargo run --release --bin brass-engine -- 500 4 2ply
cargo run --release --bin brass-engine -- 80 4 mcts-vs-heur 16 600
cargo run --release --bin brass-engine -- 80 4 mcts-vs-2ply 16 600
```

### MCTS 单决策诊断

```bash
cargo run --release --bin bench_mcts -- 7 4 5000 10000
cargo run --release --bin sweep_mcts -- 7 4 2000
cargo run --release --bin debug_mcts -- 7 4 2000 60
```

## 7. 后续建议

1. 若目标是继续提高当前最强 baseline，优先继续调 `2ply` 或直接调 `heuristic` 评分
2. 若目标是推进搜索路线，先解决 `MCTS` 强度落后于 `2ply` 的原因，再谈扩 sims
3. 后续所有文档中的新基线，优先引用本文件
