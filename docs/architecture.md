# 系统架构设计

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

### 模块划分（`engine/src/`）

```
lib.rs（模块根，声明职责层并为既有调用方再导出平铺模块路径）
│
├─ model/ 静态数据层（世界模型，无规则执行）
│   ├─ data.rs   产业/时代/动作/卡牌类型 枚举 + 板块定义 TileDef/industry_tiles()
│   └─ map.rs    地图：27 地点、城市槽位、商家与奖励、39 连接+邻接表、牌堆构成、市场/收入/货币常量
│   └─ move.rs   分层动作：结构 `Move`（操作 + 候选卡牌/保留价值）与可执行 `ResolvedMove`
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
│   ├─ legal_moves.rs 结构合法动作生成器；`legal_resolved_moves` 仅供执行适配层生成完整动作
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
│   ├─ heuristic_ai/    启发式候选生成与各类行动评分
│   │   ├─ mod.rs       对外 API 门面（Decision / candidate_actions_k 等）+ 候选编排
│   │   ├─ config.rs    HeuristicConfig：跨行动共享的权重/阈值/开关
│   │   ├─ context.rs   EvalContext：每候选批一次的评估上下文（阶段/轮次/换算便捷函数）
│   │   ├─ value.rs     只读 VP 估算（镜像 scoring.rs）+ 市场模型
│   │   ├─ board.rs     公共盘面查询（merchant 可达 / 啤酒可用 / 自由资源比例等）
│   │   ├─ probability.rs 唯一一套翻转概率模型（build 视角与 plan 视角共用）
│   │   ├─ cards.rs     卡牌保留价值（独立卡牌选择头）
│   │   ├─ lookahead.rs 确定性 2-ply 同回合 combo 前瞻
│   │   ├─ plan.rs      时代分档与生产计划选择
│   │   ├─ build.rs     Build 评分与候选生成
│   │   ├─ network.rs   Network / Double-Rail 评分与候选生成
│   │   ├─ develop.rs   Develop 评分与候选生成
│   │   ├─ sell.rs      Sell 评分与候选生成
│   │   ├─ loan.rs      Loan 评分
│   │   └─ scout_pass.rs Scout / Pass 评分
│   ├─ determinize.rs   隐藏信息 determinize（保己方手牌、重洗对手手牌；共享模块）
│   ├─ nn_mcts.rs       网络引导 ISMCTS（具体候选动作树、批量 Python 推理、4 玩家 value）
│   ├─ replay.rs        replay-web 内存会话：快照/完整合法集/DecisionTrace 记录 + StrategyAdapter 策略契约
│   ├─ python_worker.rs PythonWorkerStrategy：replay-web 网络座位（子进程 worker + stdin/stdout JSON 协议）
│   └─ random_ai.rs     随机基线
│
└─ bridge/ 桥接 / 序列化层（Python/NN 相关；依赖全部上层）
    ├─ action_features.rs ResolvedMove → 执行候选动作特征
    ├─ move_codec.rs ResolvedMove ⇄ canonical 字符串（无损，含资源源/已选卡牌）
    ├─ encode.rs     状态 → 张量特征编码（board/links/global/hands）
    ├─ replay_fmt.rs 中文回放格式化（纯只读，供 replay 二进制与 Python 驱动共用）
    └─ pymod.rs      PyO3 绑定 brass_ai._engine（GameState 类 + search_net + stepwise replay）
```

依赖方向大体单向：`model` → `game_state` → `gameplay` → AI / bridge。`gameplay` 只生成规则意义上的合法动作；候选动作特征编码属于 bridge，因此规则层不依赖 bridge。
历史反向边 `bridge/encode.rs` → `ai/heuristic_ai::estimate_rounds_remaining` 已消除：该函数现已并入 `GameState::rounds_remaining`，桥接层直接调用状态方法。

`main.rs` 为默认二进制入口，用于批量对局统计扫描；开发实验入口（`engine/src/bin/`）见 [engine-tools.md](./engine-tools.md)，replay-web 的会话模型与 Python worker 协议见 [replay-design.md](./replay-design.md)。

为保持 Rust 调用方、二进制工具及 PyO3 绑定的兼容性，`lib.rs` 仍公开再导出 `_engine::rules`、`_engine::state` 等原有平铺路径；新代码应使用对应的职责层路径。

### Python AI 训练

Python 侧位于 `python/`，负责训练编排与模型推理，不重复实现游戏规则、合法动作生成或搜索树。规则执行、信息集确定化、状态特征编码和网络引导 ISMCTS 均以 Rust `brass_ai._engine` 为唯一权威实现。

```
python/
├─ brass_ai/
│  ├─ hierarchical_policy.py  Rust 候选动作/teacher 适配、schema 校验与候选 batch padding
│  ├─ net.py          Policy-Value 网络：候选动作打分（FiLM+集合上下文）+ rank/winner/econ 头
│  ├─ rust_mcts.py    `GameState.search_net` 的 PyTorch 回调适配器
│  ├─ selfplay.py     Sample、imitation 与 MCTS self-play 生成
│  ├─ train.py        损失函数、Trainer、优化器和学习率调度器
│  ├─ evaluate.py     固定种子、轮换座位的对局评测
│  ├─ mp_selfplay.py  常驻 multiprocessing worker 池
│  ├─ replay_worker.py replay-web 网络座位子进程：加载 checkpoint，stdin/stdout JSON 协议应答决策
│  └─ progress.py     长任务进度与 ETA 输出
├─ bootstrap_imitation.py
│                     用 Rust 启发式教师生成行为克隆预训练数据（当前唯一训练入口）
└─ tests/             Rust bridge、搜索、自博弈、训练、replay 分片与 replay worker 测试
```

#### Rust-Python 契约

`brass_ai._engine.GameState` 是 Python 侧唯一的游戏状态对象。它提供：

- `search_net(...)`：Rust 中执行批量网络 ISMCTS；Python callback 输入为 `board`、`links`、`global`、`own_hand`、`opp_hands`、补齐后的 `candidates` 和 `candidate_mask`，返回 `(candidate_logits, values)`。Rust 负责合法动作枚举和 mask，Python 不应重新实现动作映射。
- `state_to_tensor()`：供训练样本采集使用的单状态特征；当前 state-feature schema v4 的维度固定为 board `(24, 49)`、links `(7, 39)`、global `(168,)`、own hand `(35,)`、 opponent hands `(105,)`。links 同时编码地图静态的水路/铁路可建性、动态建成状态与归属；global 包含每位玩家的手牌数、本回合花费、收入格和收入等级，以及每个商家的收货类型（5 种：Blank/Any/棉纺/制造厂/陶器）与啤酒状态；board 额外携带静态的槽位行业能力与槽位序号平面。Rust 还导出 board-cell/location 与 connection endpoint 拓扑，Python 网络据此做节点-边消息传递。
- `legal_candidates()`：Rust 返回完整可执行动作及其结构化特征；网络只对当前候选集合执行 softmax。

网络当前直接对每个具体候选动作输出 logit（FiLM 调制 + 候选集上下文）；候选动作特征由 Rust `bridge::action_features` 编码，合法动作枚举也完全由 Rust 完成。301 维动作特征的逐块布局、每类动作的实测编码示例，以及 policy/rank/winner/econ 网络头的设计见 [ai-action-encoding.md](./ai-action-encoding.md)。

#### 训练循环现状

当前唯一保留的训练入口是 `bootstrap_imitation.py`（heuristic imitation bootstrap）：

```
Rust heuristic 完整对局
  -> imitation 样本分片 <ckpt>.imitation/imitation-*.pkl
  -> Trainer (AdamW + CosineAnnealingLR)
  -> checkpoint（含 model/optimizer/scheduler/schema 状态）
  -> 网络引导 Rust MCTS vs heuristic benchmark
```

默认 full-legal 模式下，每个 `Sample` 保存当前视角状态、Rust state snapshot 与
teacher canonical action（候选集训练前实时物化），监督目标为候选上的 policy
分布、rank/winner 终局目标与经济辅助目标。只有正常到达 `game_over` 的完整
对局可以入库；达到 `max_moves` 的截断局会被丢弃，不能以当前盘面伪造终局价值。

面向网络的长期 self-play 训练循环尚未重建顶层入口。`selfplay.py`
（`play_game` / `play_batch`）、`mp_selfplay.py`（worker 池）与 `train.py`
（`run_loop`）保留了可复用的模块能力，重新设计自对弈入口时应基于它们构建。

搜索树以 Rust `RustISMCTS` 为唯一实现，Python 侧不做搜索、不实现规则。任何规则或特征变更必须同时更新 Rust bridge 契约、Python 测试和本节。state-feature schema 或 action-feature schema 升级会拒绝旧 checkpoint/样本，必须重新采样训练。
