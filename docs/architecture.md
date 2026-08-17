# 系统架构设计

待补充

## 总体架构

```
+-------------------------------------------------------------------+
|                        1. Tabletop Simulator                      |
|  [TTS 客户端] --(隐蔽挂载小棋子 Lua 脚本)--+                      |
+--------------------------------------------|----------------------+
                                             | HTTP / WebRequest (JSON)
                                             v (localhost)
+-------------------------------------------------------------------+
|                       2. AI Service (Python)                      |
|  +---------------------------+     +---------------------------+  |
|  |  FastAPI / gRPC Web Server |     | PySide6 桌面透明悬浮窗    |  |
|  +-------------+-------------+     +-------------+-------------+  |
|                |                                 ^                |
|                v                                 | (Top-3 推荐操作)|
|  +-----------------------------------------------+-------------+  |
|  | ISMCTS 搜索决策器 + PyTorch (Policy-Value Network)          |  |
|  +-----------------------------+-------------------------------+  |
+--------------------------------|----------------------------------+
                                 | Pybind11 / CFFI 接口调用
                                 v
+-------------------------------------------------------------------+
|                   3. Game Engine (Rust)                           |
|  * 极速游戏状态 (State) 更新                                      |
|  * 合法动作生成器 (Action Generator)                              |
|  * 图论连通性分析 (Graph Connectivity & Resource Flow)            |
+-------------------------------------------------------------------+
```

## 模块职责

### 模块划分（`src/engine/src/`）

```
lib.rs（模块根，声明职责层并为既有调用方再导出平铺模块路径）
│
├─ model/ 静态数据层（世界模型，无规则执行）
│   ├─ data.rs   产业/时代/动作/卡牌类型 枚举 + 板块定义 TileDef/industry_tiles()
│   └─ map.rs    地图：27 地点、城市槽位、商家与奖励、39 连接+邻接表、牌堆构成、市场/收入/货币常量
│   └─ move.rs   跨层规范动作 `Move`（供 gameplay、AI、bridge 共用）
│
├─ game_state/ 动态状态层
│   ├─ state.rs   GameState + Player + Card/BoardTile/Link/MerchantTile
│   │             + 状态变更原语（置板块/翻面/消耗资源/市场结算/收入） + 发牌洗牌 + 牌池重建
│   │             + 免费煤/铁/酒缓存 + 连通分量缓存（指纹惰性重建，自愈）+ 每玩家网络掩码缓存（指纹自愈）
│   │             + deck_composition 静态缓存 + assert_caches_consistent
│   ├─ graph.rs   图连通性查询（读 state 缓存，无 BFS）+ 每玩家网络查询（O(1) 掩码）+ 资源源查询与成本函数
│   │             （煤/铁/酒 find_*_sources + coal_purchase_cost/iron_purchase_cost）+ is_resource_depleted
│   └─ income.rs  收入格(0-99) ⇄ 收入等级(-10..30) 换算
│
├─ gameplay/ 规则执行层
│   ├─ rules.rs    兼容门面：重导出既有规则 API；`apply_move` 原子分发与完整状态回滚
│   ├─ legal_moves.rs 单一合法动作生成器（原始动作 / 每类一个代表动作）；不依赖策略槽编码
│   ├─ actions/    按行动领域组织的规则校验与状态转换
│   │   ├─ common.rs 资源选择/校验（免费优先）与卡牌辅助
│   │   ├─ build.rs   BUILD 合法目标、成本与执行
│   │   ├─ network.rs NETWORK（含 RailTx 双铁路 dry-run/回滚）
│   │   ├─ develop.rs DEVELOP 与免费发展奖励结算
│   │   ├─ sell.rs    SELL 路径、商人奖励与多卖货执行
│   │   └─ basic.rs   LOAN / SCOUT / PASS
│   ├─ engine.rs   回合/轮次推进、收入阶段、短差偿付、时代切换 + `handle_turn_result`
│   ├─ game_loop.rs 共享全局驱动：`play(state, max_moves, hooks, choose)` 统一游戏循环
│   └─ scoring.rs  时代计分（连接 VP + 板块 VP）+ 终局排名
│
├─ ai/ AI 决策层
│   ├─ heuristic_ai.rs  1-ply 启发式（时代分档/生产计划/各类行动评分/局面估值，2358 行，含硬编码护栏）
│   ├─ search_ai.rs     2-ply 确定性同回合 combo 前瞻（当前默认"教师"）
│   ├─ mcts_ai.rs       启发式引导 ISMCTS（determinize + PUCT + MaxN 值向量）
│   ├─ nn_mcts.rs       网络引导 ISMCTS（slot 树、批量 Python 推理、分叉 policy 合并、4 玩家 value）
│   └─ random_ai.rs     随机基线
│
└─ bridge/ 桥接 / 序列化层（Python/NN 相关；依赖全部上层）
    ├─ policy.rs     固定动作表(1316 槽) + move→slot 映射 + `legal_slot_moves` + legal_mask + 槽位类型
    ├─ move_codec.rs Move ⇄ canonical 字符串（无损，含资源源/卡牌选择）
    ├─ encode.rs     状态 → 张量特征编码（board/links/global/hands）
    ├─ replay_fmt.rs 中文回放格式化（纯只读，供 replay 二进制与 Python 驱动共用）
    └─ pymod.rs      PyO3 绑定 brass_engine（GameState 类 + search_net + stepwise replay）
```

依赖方向大体单向：`model` → `game_state` → `gameplay` → AI / bridge。`gameplay` 只生成规则意义上的
合法动作，策略槽编码属于 `bridge::policy`，因此规则层不依赖 bridge。
唯一反向边：`bridge/encode.rs`（桥接层）依赖 `ai/heuristic_ai::estimate_rounds_remaining`（AI 层）——属"胶水层依赖全部"的既有设计，非漂移。

为保持 Rust 调用方、二进制工具及 PyO3 绑定的兼容性，`lib.rs` 仍公开再导出
`brass_engine::rules`、`brass_engine::state` 等原有平铺路径；新代码应使用对应的职责层路径。

### Python AI 训练

待补充
