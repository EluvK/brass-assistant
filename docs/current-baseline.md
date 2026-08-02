# 当前可信基线（2026-08-02）

> 本文档汇总 **当前可相信** 的 Brass 引擎基线结果。
> 这些数据建立在以下修复之后：
> - 铁路时代不再二次埋牌
> - 非法动作失败后不再推进回合
> - `Develop` 合法动作生成与可支付铁成本对齐
> - `Sell` 合法性判定与执行逻辑对齐
> - 连接得分不再错误计入未翻面建筑
> - `replay` 时代结算明细与真实积分口径对齐
> - heuristic 多卖货计划不再重复占用同一桶商家酒 / 啤酒资源
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
- `score_era()` 的连接得分只统计已翻面建筑
- `replay` 中连接得分来源明细只显示已翻面建筑
- heuristic 生成的 `Sell` 动作不会再超用同一桶商家酒

当前测试规模：
- unit tests: `3`
- integration tests: `25`

## 4. 当前强度基线

### 4.1 Heuristic（500 局）

命令：

```bash
cargo run --release --bin brass-engine -- 500 4 heuristic
```

结果：

| 指标 | 数值 |
| --- | --- |
| Avg final VP per player | `[49.258, 48.524, 47.092, 47.768]` |
| Avg VP/人 | `~48.2` |
| built / 局 | `16.6` |
| flipped / 局 | `11.1` |
| links / 局 | `20.4` |
| build / 局 | `39.9` |
| network / 局 | `38.3` |
| develop / 局 | `11.0` |
| sell / 局 | `9.8` |
| loan / 局 | `15.5` |
| pass / 局 | `10.1` |
| Avg final income / 人 | `~2.45` |

### 4.2 2-Ply（500 局）

命令：

```bash
cargo run --release --bin brass-engine -- 500 4 2ply
```

结果：

| 指标 | 数值 |
| --- | --- |
| Avg final VP per player | `[56.084, 53.44, 53.228, 52.894]` |
| Avg VP/人 | `~53.9` |
| built / 局 | `19.5` |
| flipped / 局 | `13.0` |
| links / 局 | `19.7` |
| build / 局 | `45.3` |
| network / 局 | `36.7` |
| develop / 局 | `9.0` |
| sell / 局 | `11.1` |
| loan / 局 | `17.1` |
| pass / 局 | `5.9` |
| Avg final income / 人 | `~3.21` |

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
| MCTS avg VP | `50.8` |
| heuristic avg VP | `47.8` |

结论：
- 当前配置下，`MCTS` 的平均 VP 已高于 `heuristic`
- 但 80 局样本下 seat 胜率只略高于 4 人局均线 `25%`，暂不把这个优势描述为“显著领先” 

### 4.4 MCTS vs 2-Ply（80 局，16 线程，600 sims）

命令：

```bash
cargo run --release --bin brass-engine -- 80 4 mcts-vs-2ply 16 600
```

结果：

| 指标 | 数值 |
| --- | --- |
| MCTS seat 胜率 | `21.2% (17/80)` |
| MCTS avg VP | `47.9` |
| 2-ply avg VP | `52.5` |

结论：
- 当前配置下，`MCTS` 明显弱于 `2ply`

## 5. 当前总判断

当前 4 人局强度排序：

1. `2ply`
2. `mcts`（当前样本下平均 VP 已超过 `heuristic`，但优势不够稳）
3. `heuristic`
4. `random` 仅作弱基线/烟雾测试

因此：
- `2ply` 可以继续作为当前最强可用 baseline
- `MCTS` 目前已不再明显弱于 `heuristic`，但距离 `2ply` 仍有清楚差距
- `MCTS` 仍更像“可运行研究原型”，不应当作现阶段最强策略

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

当前 `bench_mcts` 参考：
- `5000 sims` 约 `12.73s`
- `10000 sims` 约 `18.33s`

## 7. 后续建议

1. 若目标是继续提高当前最强 baseline，优先继续调 `2ply` 或直接调 `heuristic` 评分
2. 若目标是推进搜索路线，当前应重点解释 `MCTS` 为何能压过 `heuristic` 的 avg VP，却仍明显落后 `2ply`
3. 后续所有文档中的新基线，优先引用本文件
