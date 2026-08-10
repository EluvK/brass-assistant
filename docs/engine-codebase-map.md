# Engine 代码结构地图（Rust 引擎现状梳理）

> 本文档是 2026-08 对 `src/engine/` 的静态梳理结果：每个文件的当前职责、整体层次结构与依赖方向、以及整改候选问题清单。
> 文档由 AI agent 通读全部 `src/engine/src/**/*.rs` 后生成，供后续逐项重构时对照。

## 1. 整体层次结构

```
lib.rs（模块根，仅声明 mod）
│
├─ 静态数据层（世界模型，无逻辑）
│   ├─ data.rs   产业/时代/动作/卡牌类型 枚举 + 板块定义 TileDef/industry_tiles()
│   └─ map.rs    地图：27 地点、城市槽位、商家与奖励、39 连接、牌堆构成、市场/收入/货币常量
│
├─ 动态状态层
│   ├─ state.rs   GameState + Player + Card/BoardTile/Link/MerchantTile
│   │             + 状态变更原语（置板块/翻面/消耗资源/市场结算/收入） + 发牌洗牌 + 牌池重建
│   │             + 免费煤/铁缓存与连通分量缓存维护（指纹惰性重建，自愈）
│   ├─ graph.rs   图连通性查询（读 state 缓存，无 BFS）+ 资源源查询与成本函数
│   │             （煤/铁/酒 find_*_sources + coal_purchase_cost/iron_purchase_cost）+ is_resource_depleted
│   └─ income.rs  收入格(0-99) ⇄ 收入等级(-10..30) 换算
│
├─ 规则执行层
│   ├─ rules.rs    Move 枚举(9 变体) + 资源源选择/校验（免费优先）
│   │              + 合法动作生成（`generate_moves` 单一生成器，`MoveExpansion` All/OnePerSlot）
│   │              + 7 种行动执行 + apply_move 分发
│   ├─ engine.rs   回合/轮次推进、收入阶段、短差偿付、时代切换 + `handle_turn_result`
│   ├─ game_loop.rs 共享全局驱动：`play(state, max_moves, hooks, choose)` 统一游戏循环
│   └─ scoring.rs  时代计分（连接 VP + 板块 VP）+ 终局排名
│
├─ AI 决策层
│   ├─ heuristic_ai.rs  1-ply 启发式（时代策略档/生产计划/各类行动评分/局面估值，2366 行）
│   ├─ search_ai.rs     2-ply 确定性同回合 combo 前瞻
│   ├─ mcts_ai.rs       启发式引导 ISMCTS（每 sim 确定性 + PUCT 启发式先验 + MaxN 值向量）
│   ├─ nn_mcts.rs       网络引导 ISMCTS（批量推理、分叉 policy 合并、4 玩家 value，依赖 Python）
│   └─ random_ai.rs     随机基线
│
├─ 桥接 / 序列化层（Python/NN 相关）
│   ├─ policy.rs     固定动作表(1316 槽) + move→slot 映射 + legal_mask + 槽位类型
│   ├─ move_codec.rs Move ⇄ canonical 字符串（无损，含资源源/卡牌选择）
│   ├─ encode.rs     状态 → 张量特征编码（board/links/global/hands）
│   ├─ replay_fmt.rs 中文回放格式化（纯只读）
│   └─ pymod.rs      PyO3 绑定 brass_engine
│
└─ 可执行文件
    ├─ main.rs            批量对局统计 / MCTS-vs-X 混战 / 动作频率
    └─ bin/ replay, stat_game, sweep_scores, sweep_canal,
              bench_mcts, debug_mcts, sweep_mcts
```

依赖方向严格单向：`data/map`（无依赖）→ `state` → `graph/rules` → `engine` → `AI` → `policy/move_codec/encode/replay_fmt/pymod`（胶水层依赖全部）。

## 2. 逐文件职责

| 文件 | 行数 | 职责 |
| --- | --- | --- |
| `data.rs` | 219 | `IndustryType`/`Era`/`Action`/`CardType` 枚举、`TileDef` 与 `industry_tiles()`（六大产业板块参数） |
| `map.rs` | 575 | `Loc`(27)、`city_slots()`、商家定义与牌组构成（`merchant_bonus_at` 唯一查找）、`connections()`(39)、牌堆/市场/货币/连接常量 |
| `state.rs` | 934 | `GameState` 定义 + 全部状态变更原语；免费煤/铁缓存 + 连通分量缓存（`connected_mask` 指纹惰性重建、`rebuild_free_sources`、`sync_free_source`、`assert_caches_consistent`）；初始化发牌/洗牌/埋牌；`deck_composition` 牌池重建 |
| `graph.rs` | 380 | 连通性查询（`connected_locations` 读缓存，无 BFS）、`is_in_network`/`player_has_presence`、煤/铁/酒源查询（`find_coal_sources`/`find_iron_sources` 读缓存+市场逐桶+补给）、成本函数（`coal_purchase_cost`/`iron_purchase_cost`）、`is_resource_depleted` |
| `income.rs` | 95 | 收入格⇄等级换算（含贷款下限判定） |
| `rules.rs` | 2303 | 规则核心：Move 定义、资源源选择（免费优先枚举/校验）、合法动作生成（`generate_moves` 单一生成器 + `MoveExpansion` All/OnePerSlot）、7 行动执行器、双铁路事务（`RailTx` 守卫，含 dry-run）与回滚、多卖货计划 |
| `engine.rs` | 292 | `advance_turn`/`end_round`/`end_canal_era`/`end_game`/`resolve_shortfall`/`step` + `handle_turn_result`（TurnResult 分发的唯一实现） |
| `game_loop.rs` | 152 | 共享全局驱动 `play`：循环骨架（guard/game_over/apply/advance/时代切换）+ 可选 hooks（before_move/after_move/on_era/after_era）+ `AfterEra` 停止控制 + `finish_game`；全部游戏二进制统一走此驱动 |
| `scoring.rs` | 104 | `score_era`（连接+板块 VP）、`final_ranking`（VP→收入→现金平手规则） |
| `heuristic_ai.rs` | 2358 | 1-ply 启发式 AI：时代分档策略、生产计划、build/network/develop/sell/loan/scout 评分、`evaluate_position` 叶估值、debug 辅助 |
| `search_ai.rs` | 132 | 2-ply 同回合 combo 前瞻 + 回合末流动性惩罚 |
| `mcts_ai.rs` | 490 | 启发式引导 ISMCTS：确定性采样、MoveKey 归一化、PUCT + MaxN 向量、OnePly/TwoPly 叶评估 |
| `nn_mcts.rs` | 564 | 网络引导 ISMCTS：slot 树、批量 Python 推理、分叉先验合并、Dirichlet 噪声 |
| `random_ai.rs` | 36 | 随机基线 |
| `policy.rs` | 262 | 固定动作槽表（Build 482 + Network 39 + 双铁路 703 + Develop 42 + Sell 47 + 3 尾槽 = 1316）、`move_slots`/`legal_mask`/`slot_type` |
| `move_codec.rs` | 491 | Move ⇄ canonical 字符串（含回环测试） |
| `encode.rs` | 205 | `state_to_tensor`：board(17,49)/links(6,39)/global(50)/hands(35×4) |
| `replay_fmt.rs` | 515 | 中文回放日志格式化（只读，供 replay 二进制与 Python 驱动共用） |
| `pymod.rs` | 464 | PyO3 绑定：`GameState` 类、`apply_move`/`determinize`/`search_net`/`state_to_tensor`/回放 stepwise 接口 |
| `main.rs` | 334 | 批量对局统计 + MCTS 混战，rayon 并行 |
| `bin/replay.rs` | 251 | 单局中文详细回放（含运河结算明细、canal-only 模式） |
| `bin/stat_game.rs` | 187 | 单局分时代动作/翻面诊断 |
| `bin/sweep_scores.rs` | 150 | seed 区间批量 CSV（终局/运河收入等经济列） |
| `bin/sweep_canal.rs` | 182 | 仅运河时代批量扫描 |
| `bin/bench_mcts.rs` | 73 | MCTS 单决策计时 |
| `bin/debug_mcts.rs` | 60 | MCTS 根树/候选集诊断 |
| `bin/sweep_mcts.rs` | 47 | MCTS (depth, c_puct, leaf) 参数扫描 |
| `tests/engine_tests.rs` | 1201 | 规则/计分/确定性/MCTS/卖货/双铁路集成测试 |

## 3. 候选问题清单（按依赖自底向上排列）

1. ✅ **热路径重复分配（已完成 2026-08）**：`player_industry_stack()`、`city_slot_offsets()`、`total_city_slots()` 改为 `OnceLock` 静态表；`loc_from_key` 收敛为 `state::loc_from_key(key) -> Option<(Loc, usize)>`（O(1) 静态查表），删除 rules.rs / replay_fmt.rs / stat_game.rs 三份拷贝。验收：build + 34 tests + seed233/7 replay 正常跑完、无非法动作/guard。
2. ✅ **`GameState<R: Rng>` 泛型解耦（已完成 2026-08）**：去掉泛型，`GameState` 固定内部成员 `rng: StdRng`（私有，仅 `new()` 初始化洗牌用；引擎逻辑完全确定性）。全仓 ~150 处 `GameState<impl Rng>`/`GameState<R>` 收敛为 `GameState`；MCTS/random_ai 改用独立 `StdRng::from_entropy()`（`state.rng` 不再外泄）。验收：build 零警告 + 34+7 tests + replay/main/stat_game 正常跑完。
3. ✅ **资源源"幽灵桶"与资源成本实现（已完成 2026-08）**：两件事一起解决。(a) 修复多桶市场采购 flat-pricing bug（原按最便宜格价 × N 计费）：`find_coal_sources`/`find_iron_sources` 现按**各自槽位价**逐桶列出，`source_options` 缺口补位取最便宜 N 桶，`execute_develop` 按所选源 `src.price` 收费。(b) **废除资源搜索 BFS**：`GameState` 新增 `free_coal_mines`/`free_iron_works` 缓存（`place_tile`/`remove_tile`/`consume_*`/`auto_sell_to_market`/时代末重建维护）+ `component_cache` 连通分量位掩码缓存（39 链接存在性指纹惰性重建，对直接写 `links` 自动自愈，`RwLock` 保证 PyO3 Send+Sync）。`connected_locations`/`find_beer_sources` 一并改为读缓存，引擎内资源相关 BFS 归零。新增成本 API `graph::coal_purchase_cost`/`iron_purchase_cost`（免费先抵扣→市场逐桶→General Supply 空市价）。魔法数字 6 → `map::GENERAL_SUPPLY_CAP = 4`（单行动 max 2 / 双动 max 4）。验收：build 零警告 + 42 tests + seed7/233 replay 与基线字节一致。

4. ✅ **重复实现（已完成 2026-08）**：(a) `handle_turn_result` 并入 `engine.rs`（`TurnResult` 分发的唯一实现），mcts_ai.rs / search_ai.rs 删本地副本，nn_mcts.rs / main.rs / pymod.rs:apply_move 的内联 match 一并改走公共函数（`loc_from_key` 三份已并入 1）。(b) 商家 bonus 查找收敛为 `map::merchant_bonus_at(loc)`（静态数据层唯一查找），rules.rs 删 `merchant_bonus_for`，heuristic_ai.rs `merchant_bonus_value` 改匹配返回的 `MerchantBonus`（AI 侧加权逻辑保留）。验收：build 零警告 + 42 tests + seed7/233 replay 字节一致 + bench_mcts ~4350 sims/s 无劣化。注：bin/* 各回放/扫描二进制内「advance_turn + 时代分支」的内联循环仍存在，属第 6 项（游戏循环统一）范畴，且部分需在时代清理前打印结算明细，不并入本次。
5. ✅ **`rules.rs` 合法生成依赖 `&mut state` + dry-run 回滚（已完成 2026-08）**：(a) **双铁路事务化**：新增 `RailTx` 守卫（记录 link 放置 + 煤消耗 undo，`Drop` 保证任何返回路径/panic 都回滚），`get_second_rail_options` 与 `execute_network_double` 全部改走事务，删除 4 处手工回滚块（原先任意新增校验若漏回滚即污染棋盘）。(b) **raw/slot 双轨合并为单一生成器**：`generate_moves(state, MoveExpansion)`（`All` = 全枚举 raw 空间，`OnePerSlot` = 每槽一个代表），`legal_moves`/`legal_slot_moves` 只是薄包装，覆盖不可能再漂移。All 模式迭代结构与旧 `legal_moves` 逐字节一致。验收：build 零警告 + 42 tests（debug+release）+ seed7/233 replay 字节一致 + bench_mcts ~4300 sims/s 无劣化 + Python 绑定测试与基线一致（13 过 1 既有失败 `test_apply_move_valid_and_invalid`，改动前已存在，与本次无关）。
6. ✅ **游戏循环去重（已完成 2026-08）**：新增 `game_loop.rs` 共享驱动 `play(state, max_moves, hooks, choose)`，统一「guard / game_over / apply_move / advance_turn / 时代切换」循环骨架。`choose` 返回 `Option<Move>`（None = 空手推进回合）；hooks 为 `before_move`（行动前统计/盘面快照）、`after_move`（行动后打印）、`on_era`（时代清理前，返回 `AfterEra::{Continue, StopBeforeCleanup, StopAfterCleanup}`，canal-only 用 StopBeforeCleanup 保留清理前棋盘）、`after_era`（清理后快照）；`finish_game` 统一收尾 `if !game_over end_game`。main.rs、replay.rs、stat_game.rs、sweep_scores.rs、sweep_canal.rs、bench_mcts.rs、debug_mcts.rs、sweep_mcts.rs 全部改走驱动；`pymod.rs:apply_move` 已并入 `handle_turn_result`（第 4 项）。多 hook 共享统计用 `RefCell`（replay 的 canal/rail_stats、sweep_canal 的 CanalResult）避免闭包 `&mut` 冲突。验收：build 零警告 + 42 tests + seed7/233 replay 字节一致 + bench_mcts ~4400 sims/s 无劣化 + 各二进制（stat_game/sweep_*/canal-only replay/main 混战）正常产出。
7. **多卖货/启发式干跑整状态 clone**：`build_multi_sell_plans`、`sell_plan_executes_all`、`score_sell_plan`、`best_same_turn_after_loan` 等 `state.clone()` 昂贵。
8. **`heuristic_ai.rs` 单文件 2366 行**：魔法权重密集、硬编码 BAN 开关（当前优先级最低，暂缓）。
9. 🔜 **每玩家路网连通性缓存（后续方向，2026-08 规划）**：为加速"判断当前玩家能做什么操作"（建厂/建网目标合法性、启发式网络评分、MCTS 树内重复可达性查询），在现有 `component_cache`（任意链接全局连通分量，本次已落地）之上，新增**每玩家网络掩码** `network_mask: [u32; 4]`（27 位：自有板块位置 ∪ 自有链接端点）。
   - **语义前提（关键）**：`is_in_network(pid, loc)` 现为"自有板块/自有链接**直接在场**判定"（graph.rs:46-75），与 npow `isInNetwork`（gameState.js:311-345）一致；又因建链必须邻接自有网络，该判定 ≡ "玩家自有链接连通分量"。所以掩码维护 = 置板块/建链时置位即可，**无需做每玩家分量 BFS**。
   - **待裁决**：规则书是否允许"经对手链接延伸自己的网络"？若允许则 `is_in_network` 语义应改为"任意链接分量 ∩ 玩家触达"（从 `component_cache` 派生 `player_touch[pid][comp]`），属于**规则修正**而非纯优化，需与 npow 对比后裁决。
   - 维护点：`place_tile`（置位）、`execute_network`/`execute_network_double` 建链（置两端 + via 农场位）、双铁路 dry-run 回滚（沿用 `ResourceUndo` 机制，新增 `NetworkMaskUndo`）、`end_canal_era`（运河链清除后重算）、`resolve_shortfall` 卖板块（无链接残留时清位）。
   - **自愈策略**：测试/双铁路 dry-run 直接写 `links` 仍会绕过维护 → 建议沿用本次的"指纹惰性重建"：每玩家指纹 = 自有链接存在位掩码 + 自有板块指纹（塌缩成 u64），失配时从 `links`+`city_tiles` 重算。可放进 `component_cache` 同一结构或独立 `RwLock`。
   - 衍生能力：`network_locations(pid)`（O(1) 掩码迭代）、`player_has_presence`（= 掩码非零）、"网络扩张潜力"评分（掩码周边未建连接计数，启发式/MCTS 可直接用）、`legal_moves` 里"产业牌可建位置"一次求值（`network_mask` 预筛，替代逐候选 `is_in_network`）。
   - 验收：build 零警告 + tests 全绿 + seed7/233 replay 字节一致（若语义不变）或按裁决合理回归 + bench_mcts 不劣化。

## 4. 约定

- 优先级：1→7 为引擎底层结构/性能问题，逐项修复；8（heuristic_ai 重构）暂缓。
- 每项修复验收：`cargo build --release` + `cargo test` + 基线 replay 输出无回归（对局日志能正常产出）。
