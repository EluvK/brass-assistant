# Engine 代码结构地图（Rust 引擎现状梳理 · 重写版）

> 本文档是对 `src/engine/` 的静态通读结果（2026-08 复核）：每个文件的当前职责、整体层次结构与依赖方向、以及**尚未落地**的整改候选清单。
> 此前已完成并验收的优化（热路径静态表 / 泛型解耦 / 资源源缓存与逐桶定价 / 重复实现收敛 / 双铁路事务与单一生成器 / 游戏循环去重）已生效，**不再列入**本清单；本文只保留仍需改进的部分。
> 行数以本次 `wc -l` 为准（与旧版有出入处已修正）。

## 1. 整体层次结构

```
lib.rs（模块根，仅声明 mod；无 src/engine/baselines/ 残留内容）
│
├─ 静态数据层（世界模型，无逻辑）
│   ├─ data.rs   产业/时代/动作/卡牌类型 枚举 + 板块定义 TileDef/industry_tiles()
│   └─ map.rs    地图：27 地点、城市槽位、商家与奖励、39 连接+邻接表、牌堆构成、市场/收入/货币常量
│
├─ 动态状态层
│   ├─ state.rs   GameState + Player + Card/BoardTile/Link/MerchantTile
│   │             + 状态变更原语（置板块/翻面/消耗资源/市场结算/收入） + 发牌洗牌 + 牌池重建
│   │             + 免费煤/铁/酒缓存 + 连通分量缓存（指纹惰性重建，自愈）+ assert_caches_consistent
│   ├─ graph.rs   图连通性查询（读 state 缓存，无 BFS）+ 资源源查询与成本函数
│   │             （煤/铁/酒 find_*_sources + coal_purchase_cost/iron_purchase_cost）+ is_resource_depleted
│   └─ income.rs  收入格(0-99) ⇄ 收入等级(-10..30) 换算
│
├─ 规则执行层
│   ├─ rules.rs    Move 枚举(9 变体) + 资源源选择/校验（免费优先）
│   │              + 合法动作生成（`generate_moves` 单一生成器，`MoveExpansion` All/OnePerSlot）
│   │              + 7 种行动执行 + apply_move 分发 + RailTx 双铁路事务（dry-run/回滚）+ 多卖货计划
│   ├─ engine.rs   回合/轮次推进、收入阶段、短差偿付、时代切换 + `handle_turn_result`
│   ├─ game_loop.rs 共享全局驱动：`play(state, max_moves, hooks, choose)` 统一游戏循环
│   └─ scoring.rs  时代计分（连接 VP + 板块 VP）+ 终局排名
│
├─ AI 决策层
│   ├─ heuristic_ai.rs  1-ply 启发式（时代分档/生产计划/各类行动评分/局面估值，2358 行，含硬编码护栏）
│   ├─ search_ai.rs     2-ply 确定性同回合 combo 前瞻（当前默认"教师"）
│   ├─ mcts_ai.rs       启发式引导 ISMCTS（determinize + PUCT + MaxN 值向量）
│   ├─ nn_mcts.rs       网络引导 ISMCTS（slot 树、批量 Python 推理、分叉 policy 合并、4 玩家 value）
│   └─ random_ai.rs     随机基线
│
└─ 桥接 / 序列化层（Python/NN 相关；依赖全部上层）
    ├─ policy.rs     固定动作表(1316 槽) + move→slot 映射 + legal_mask + 槽位类型
    ├─ move_codec.rs Move ⇄ canonical 字符串（无损，含资源源/卡牌选择）
    ├─ encode.rs     状态 → 张量特征编码（board/links/global/hands）
    ├─ replay_fmt.rs 中文回放格式化（纯只读，供 replay 二进制与 Python 驱动共用）
    └─ pymod.rs      PyO3 绑定 brass_engine（GameState 类 + search_net + stepwise replay）
```

依赖方向大体单向：`data/map`（无依赖）→ `state` → `graph/rules` → `engine/game_loop` → AI → 桥接层。
唯一反向边：`encode.rs`（桥接层）依赖 `heuristic_ai::estimate_rounds_remaining`（AI 层）——属"胶水层依赖全部"的既有设计，非漂移。

## 2. 逐文件职责

| 文件 | 行数 | 职责 |
| --- | --- | --- |
| `data.rs` | 219 | `IndustryType`/`Era`/`Action`/`CardType` 枚举、`TileDef` 与 `industry_tiles()`（六大产业板块参数） |
| `map.rs` | 575 | `Loc`(27)、`city_slots()`、商家定义与牌组构成（`merchant_bonus_at` 唯一查找）、`connections()`(39) 与 `adjacency()` 邻接表、牌堆/市场/货币/连接常量、`GENERAL_SUPPLY_CAP` |
| `state.rs` | 1068 | `GameState` 定义 + 全部状态变更原语；免费煤/铁/酒缓存（`rebuild_free_sources`/`sync_free_source`/`drop_free_source` + `sync_farm_beer`/`drop_farm_beer` 覆盖啤酒农场）+ 连通分量缓存（`connected_mask` 指纹惰性重建、`compute_component_masks`、自愈）+ `assert_caches_consistent`；初始化发牌/洗牌/埋牌；`deck_composition` 牌池重建 |
| `graph.rs` | 428 | 连通性查询（`connected_locations` 读缓存，无 BFS）、`is_in_network`/`player_has_presence`、煤/铁/酒源查询（`find_coal_sources`/`find_iron_sources` 读 free 缓存；`find_beer_sources` 读 `free_beer_cubes` 缓存，自家桶不过滤连通、对手桶按掩码过滤，输出顺序与旧全扫一致）、成本函数（`coal_purchase_cost`/`iron_purchase_cost`）、`is_resource_depleted`、`cheapest_coal_for_connection` |
| `income.rs` | 95 | 收入格⇄等级换算（含贷款下限判定），3 个单元测试 |
| `rules.rs` | 2303 | 规则核心：Move 定义、资源源选择（`source_options` 免费优先枚举/`validate_source_choice` 校验）、合法动作生成（`generate_moves` 单一生成器 + `MoveExpansion` All/OnePerSlot + `legal_moves`/`legal_slot_moves` 薄包装）、7 行动执行器、`apply_move` 分发（debug 下 `assert_caches_consistent`）、双铁路事务（`RailTx`，含 dry-run 与回滚）、多卖货计划（`build_multi_sell_plans` 上限 24） |
| `engine.rs` | 292 | `advance_turn`/`end_round`/`end_canal_era`/`end_game`/`resolve_shortfall`/`step` + `handle_turn_result`（TurnResult 分发唯一实现） |
| `game_loop.rs` | 152 | 共享全局驱动 `play`：循环骨架（guard/game_over/apply/advance/时代切换）+ 可选 hooks（before_move/after_move/on_era/after_era）+ `AfterEra` 停止控制 + `finish_game`；全部游戏二进制统一走此驱动 |
| `scoring.rs` | 103 | `score_era`（连接+板块 VP）、`final_ranking`（VP→收入→现金平手规则） |
| `heuristic_ai.rs` | 2358 | 1-ply 启发式：时代分档 `EraProfile`/`Phase`、生产计划 `Plan`、build/network/double-rail/develop/sell/loan/scout/pass 评分、`evaluate_position` 叶估值、`candidate_actions_k`（MCTS 先验来源）、硬编码护栏（`BAN_BUILD_LV1_BREWERY`/`BAN_DEVELOP_IRON_LV2_PLUS`）、临时 debug 助手（`debug_flip`/`debug_market_adjust`） |
| `search_ai.rs` | 132 | 2-ply 同回合 combo 前瞻（`choose_action_2ply`，`heuristic_ai::choose_action` 现委托给它）+ 回合末流动性惩罚 |
| `mcts_ai.rs` | 490 | 启发式引导 ISMCTS：`determinize`（对手手牌采样）、`MoveKey` 归一化、PUCT + MaxN 向量、OnePly/TwoPly 叶评估、归一化除数魔法常数 |
| `nn_mcts.rs` | 564 | 网络引导 ISMCTS：slot 树、批量 Python 推理（`flush_net` 每请求 1 行，返回 (type(7), goal(P), value(4))）、分叉先验合并、Dirichlet 噪声、virtual-loss 式近似（visits 先于 value 计数）、first-parker world 近似 |
| `random_ai.rs` | 36 | 随机基线 |
| `policy.rs` | 261 | 固定动作槽表（Build 482 + Network 39 + 双铁路 703 + Develop 42 + Sell 47 + 3 尾槽 = 1316）、`move_slots`/`legal_mask`/`slot_type`/`describe_slot`、`double_rail_pairs` 静态表 |
| `move_codec.rs` | 491 | Move ⇄ canonical 字符串（无损，含资源源/卡牌选择），4 个单元测试（回环/槽位稳定/畸形输入/槽位区分） |
| `encode.rs` | 204 | `state_to_tensor`：board(17,49)/links(6,39)/global(50)/hands(35×4)；依赖 `heuristic_ai::estimate_rounds_remaining` |
| `replay_fmt.rs` | 505 | 中文回放日志格式化（只读，供 replay 二进制与 Python 驱动共用） |
| `pymod.rs` | 464 | PyO3 绑定 `brass_engine`：`GameState` 类、`apply_move`/`apply_move_raw`/`determinize`/`search_net`/`state_to_tensor`/`legal_moves(_slots)`/`legal_mask` + stepwise replay 接口 |
| `main.rs` | 334 | 批量对局统计 + MCTS 混战，rayon 并行多局 |
| `bin/replay.rs` | 251 | 单局中文详细回放（含运河结算明细、canal-only 模式） |
| `bin/stat_game.rs` | 187 | 单局分时代动作/翻面诊断 |
| `bin/sweep_scores.rs` | 150 | seed 区间批量 CSV（终局/运河收入等经济列） |
| `bin/sweep_canal.rs` | 182 | 仅运河时代批量扫描（`StopAfterCleanup`） |
| `bin/bench_mcts.rs` | 73 | MCTS 单决策计时（先行 60 步启发式局面） |
| `bin/debug_mcts.rs` | 60 | MCTS 根树/候选集诊断 |
| `bin/sweep_mcts.rs` | 47 | MCTS (depth, c_puct, leaf) 参数扫描 |
| `tests/engine_tests.rs` | 1407 | 42 个集成测试：规则/计分/确定性/MCTS/卖货/双铁路/资源成本/缓存一致性 |

## 3. 候选改进清单（按依赖自底向上，只列未落地项）

### A. 状态 / 查询层

1. 🔜 **每玩家路网连通性掩码 `network_mask: [u32; 4]`**：为加速"当前玩家能做什么"的反复查询（`is_in_network` 现为逐槽/逐连接扫描，被 `get_valid_build_targets`（每候选×每卡）、`get_valid_network_targets`、`get_second_rail_options`（每候选）、heuristic 大量评分反复调用）。
   - **语义前提已闭环**：`graph::is_in_network`（state.rs/graph.rs）与 npow `isInNetwork`（gameState.js:311-345）逐条一致——"自有板块在场 或 自有链接触及（含 via 农场）"；建链又必须邻接自有网络，故判定 ≡ 玩家自有链接连通分量。**无需规则裁决，纯优化**，掩码维护 = 置板块/建链时置位即可，无需每玩家分量 BFS。
   - 维护点：`place_tile`（置位）、`execute_network`/`execute_network_double` 建链（置两端 + via 农场位）、双铁路 dry-run 回滚（沿用 `ResourceUndo` 机制，新增 `NetworkMaskUndo`）、`end_canal_era`（运河链清除后重算）、`resolve_shortfall` 卖板块（无链接残留时清位）。
   - 自愈策略：沿用"指纹惰性重建"——每玩家指纹 = 自有链接存在位掩码 + 自有板块指纹（塌缩 u64），失配时从 `links`+`city_tiles` 重算；可并入 `component_cache` 同结构或独立 `RwLock`。
   - 衍生能力：`network_locations(pid)`（O(1) 掩码迭代，取代 `connected_locations` 的 27 位展开）、`player_has_presence`（掩码非零）、"网络扩张潜力"评分、`legal_moves` 里"产业牌可建位置"预筛。
   - 验收：build 零警告 + tests 全绿 + seed7/233 replay 字节一致 + bench_mcts 不劣化。

2. **`determinize` 每模拟重建整副牌堆**：`mcts_ai::determinize` 每 sim 调 `deck_composition(player_count)`（~60 张 Vec 重建）+ 整状态 `clone()`（含 deck/hand/merchants/rng/组件缓存），5000 sims = 5000 次。`deck_composition` 只依赖玩家数、完全确定性，可 `OnceLock` 静态缓存（clone 后只做"删已知手牌 + shuffle"）；顺带省 `init_deck` 的一次重建。注意：整状态 clone 仍是每 sim 主成本，此优化只去掉其中一小块，优先级低。

### B. 规则执行层

3. **`source_options` 在 OnePerSlot 模式仍全量枚举**：`generate_moves` 的 All/OnePerSlot 只对**已算出的组合**用 `exp.each` 截断；`coal_source_options`/`iron_source_options`/`coal_options_for_connection`（及其内 `source_options` 递归枚举）在 OnePerSlot（`legal_slot_moves` / policy mask / `nn_mcts` 每个树节点展开）下仍把全部免费源组合算完才取第一个。单次组合量不大（build ≤2 桶、develop ≤2 铁、rail 1 桶），但在 nn_mcts 展开每个节点时重复发生。建议给 `source_options` 加 expansion 参数或提供 first-only 短路。同样适用于 `get_second_rail_options` 里 `beer_sources_for_link` 的 full 枚举（OnePerSlot 只取 `beers[0]`）。

4. 🔜 **卖货 / 贷款干跑整状态 clone**：`rules::build_multi_sell_plans`（每个候选 plan 一个 `state.clone()`，上限 24）、`heuristic_ai::sell_plan_executes_all`（`score_sell_plan` 内按 route 逐个试）、`heuristic_ai::best_same_turn_after_loan`、`search_ai::choose_action_2ply`（每候选一次）都在干跑验证上整状态 clone。卖货干跑只需验证"计划能全部翻面"，建议仿 `RailTx` 做**定向 undo 日志**（翻转+收入+啤酒/商家酒消耗+弃牌+商家 bonus），避免复制 deck/手牌/merchants/rng/组件缓存。验收：build 零警告 + tests 全绿（卖货/双铁路相关回归）+ bench_mcts 不劣化。

5. **合法动作生成依赖 `&mut GameState` → Python 每查询 clone**：`generate_moves`/`legal_slot_moves`/`policy::legal_mask` 均 `&mut GameState`（仅为双铁路 dry-run 的事务回滚），导致 `pymod` 的 `legal_moves`/`legal_moves_slots`/`legal_mask` 每次都要 `self.state.clone()`。方案：(a) 把双铁路 dry-run 改造成生成器主体 `&GameState`、内部仅对 dry-run 部分 clone（代价是 Rust 侧 `nn_mcts` 展开也要多一次局部 clone）；或 (b) 给 pymod 提供不 clone 的借用路径。优先级低（Python 侧查询频率低于 Rust 热路径）。

### C. AI / 搜索层

6. 🔜 **MCTS 单线程是最大性能缺口**：`mcts_ai`/`nn_mcts` 都是单线程逐 sim 循环，rayon 只用于"多局并行"（main/sweep）。面向实时目标（10-15s 内数万次模拟做单步决策），下一步应是**单决策并行 ISMCTS**：线程池切分 SIMS、每线程独立 determinize + 子树、根节点原子合并统计（或 virtual loss）。`nn_mcts` 的批量推理已消除 Python/PyO3 每 sim 瓶颈，sim 并行是其自然延续。需同时复核 `nn_mcts` 现有的两个近似（"visits 先于 value 计数"的虚拟损失、"first-parker world" 共享求值状态）在并行合并下是否仍成立。

7. **`heuristic_ai.rs` 单文件 2358 行 + 硬编码护栏（暂缓重构）**：
   - `BAN_BUILD_LV1_BREWERY`/`BAN_DEVELOP_IRON_LV2_PLUS` 是"临时战略护栏"编译期常量，静默改变全部 AI 输出；建议提为配置/参数（或随参数扫描一并处理），并保留一处显式记录。
   - `debug_flip`/`debug_market_adjust` 是"临时 debug 助手"，混在热模块尾部；建议挪到 bin 或删除。
   - 全文件魔法权重密集 + 散落的 `if state.era`/`if era_phase` 分支（部分已由 `EraProfile`/`Phase` 收敛，未收敛的仍是"评分漂移"风险）。整体重构优先级保持最低。

8. **叶值归一化与搜索超参数魔法常数**：`mcts_ai` 的 LeafEval 归一化除数（12.0/60.0）、`MctsConfig` 的 `prior_temp`/`c_puct`/`k_candidates`、`nn_mcts` 的 `c_puct`/`dirichlet_*` 均为硬编码默认值；`sweep_mcts` 只扫了 (depth, c_puct, leaf)。建议随 #6 并行化时做一次参数表化 + 更系统的扫描（尤其 `c_puct` 与批量 batch_size 的相互作用）。

## 4. 约定

- 优先级：A 层 1 是"状态查询层补缓存"的确定性收益，建议先做；B 层 3/4 是规则层干跑/枚举开销，收益中等；#5 低；C 层 #6（并行 ISMCTS）是面向实时目标的最大缺口，但改动面大，建议在 A/B 层稳定后单独排期；#7/#8 暂缓。
- 每项验收：`cargo build --release` + `cargo test` + 基线 replay 输出无回归（seed7/233 顺利运行完）+ bench_mcts 不劣化；涉及缓存项还须 `debug_assertions` 下跑 `assert_caches_consistent`。
- 参考实现只读：需要改动逻辑时从 `reference/` 复制到 `src/` 再改。
****