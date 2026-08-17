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

依赖方向大体单向：`model` → `game_state` → `gameplay` → AI / bridge。`gameplay` 只生成规则意义上的合法动作，策略槽编码属于 `bridge::policy`，因此规则层不依赖 bridge。
唯一反向边：`bridge/encode.rs`（桥接层）依赖 `ai/heuristic_ai::estimate_rounds_remaining`（AI 层）——属"胶水层依赖全部"的既有设计，非漂移。

为保持 Rust 调用方、二进制工具及 PyO3 绑定的兼容性，`lib.rs` 仍公开再导出 `brass_engine::rules`、`brass_engine::state` 等原有平铺路径；新代码应使用对应的职责层路径。

### Python AI 训练

Python 侧位于 `src/ai/`，负责训练编排与模型推理，不重复实现游戏规则、合法动作生成或搜索树。规则执行、信息集确定化、状态特征编码和网络引导 ISMCTS 均以 Rust `brass_engine` 为唯一权威实现。

```
src/ai/
├─ brass_ai/
│  ├─ net.py          Policy-Value 网络：动作类型头 + 1316 槽位头 + 4 玩家价值头
│  ├─ rust_mcts.py    `GameState.search_net` 的 PyTorch 回调适配器
│  ├─ selfplay.py     完整自博弈、访问次数策略目标和终局价值目标采集
│  ├─ dataset.py      replay 样本的压缩 NPZ 分片读写
│  ├─ train.py        损失函数、Trainer、优化器和学习率调度器
│  ├─ mp_selfplay.py  常驻 multiprocessing worker 池
│  ├─ evaluate.py     固定种子、轮换座位的对局评测
│  └─ progress.py     长任务进度与 ETA 输出
├─ train_mp.py        正式多进程训练入口
├─ bootstrap_imitation.py
│                     用 Rust 启发式教师生成行为克隆预训练数据
├─ experiments/       benchmark、diagnose、replay_net 等诊断工具
└─ tests/             Rust bridge、搜索、自博弈、训练和 replay 分片测试
```

#### Rust-Python 契约

`brass_engine.GameState` 是 Python 侧唯一的游戏状态对象。它提供：

- `search_net(...)`：Rust 中执行批量网络 ISMCTS；Python callback 输入为  `board`、`links`、`global`、`own_hand`、`opp_hands` 的二维 `float32` 数组，返回 `(type_logits, goal_logits, values)`。
- `state_to_tensor()`：供训练样本采集使用的单状态特征；维度固定为 board `(17, 49)`、links `(6, 39)`、global `(50,)`、own hand `(35,)`、 opponent hands `(105,)`。
- `legal_mask()`：当前局面合法的策略槽。网络训练和推理均只在这些槽上执行 softmax 或 argmax。

策略空间由 Rust `bridge::policy` 定义，固定为 1316 个槽。网络的完整槽位 logit 为 `type_logits[slot_type(slot)] + goal_logits[slot]`；禁止在 Python 重新实现槽位映射或动作合法性判断。

#### 自博弈与训练循环

正式路径为：

```
当前模型权重
  -> RustISMCTS 自博弈（worker 进程）
  -> 完整对局 Sample
  -> replay/iter-xxxxx.npz
  -> 有界 replay buffer
  -> Trainer (AdamW + CosineAnnealingLR)
  -> checkpoint + metrics
  -> 固定种子 benchmark / gate
```

每个 `Sample` 包含当前视角状态、合法槽掩码、根节点访问次数形成的策略分布、四位玩家的标准化终局 VP 向量，以及经济辅助监督目标。只有正常到达 `game_over` 的完整对局可以入库；达到 `max_moves` 的截断局会被丢弃，不能以当前盘面伪造终局价值。

`train_mp.py` 的 `--run_dir` 是一次训练的持久化边界：

- `manifest.json`：Rust 特征/策略空间契约和本次参数快照；
- `replay/iter-xxxxx.npz`：逐轮压缩样本分片；
- `metrics.jsonl`：每轮样本数、损失、学习率和可选 benchmark；
- `checkpoints/latest.pt`：模型、AdamW、scheduler 和 epoch 状态。

传入 `--resume` 时，入口从 `checkpoints/latest.pt` 恢复训练器状态，并从 replay 分片重建受 `--replay_size` 限制的缓冲区。worker 默认使用 CPU；主训练进程会在可用时使用 CUDA，避免多个 worker 争用单张 GPU。

旧的纯 Python MCTS 已移除，不能作为训练或评测路径。任何新训练入口必须使用 `RustISMCTS`，任何规则或特征变更必须同时更新 Rust bridge 契约、Python 测试和本节。
