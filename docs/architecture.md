# 系统架构设计

> 总体架构与阶段规划详见 `startup.md`；本文档描述目标架构蓝图与模块边界。

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

### 1. Game Engine (Rust) — 阶段 1
- `state`：纯数据、可序列化的游戏状态
- `action_gen`：合法动作生成器（动作空间可能上百）
- `graph`：城市网络连通性（增量 BFS / 并查集）
- `rules`：7 种行动的资源结算与合法性校验
- `heuristic_ai`：启发式 AI（Baseline，参考 npow `aiPlayer.js`）
- `search_ai`：2-ply 确定性前瞻（同回合两动 Combo，`score(first) + α·score(best second)`）
- 数据对齐：`reference/npow-brass-birmingham/js/gameData.js`

### 2. TTS 隐蔽数据抽取 (Lua) — 阶段 2
- 个人 Saved Objects 小物件脚本，快捷键触发
- 读取公共盘面 + 己方手牌 → JSON → `localhost` HTTP POST
- 参考：`reference/ikegami-tts-brass/objs/` 的 API 用法

### 3. AI Service (Python) — 阶段 3
- Pybind11 / CFFI 桥接 Rust 引擎
- 状态向量化（Feature Encoding）
- Policy-Value 双头网络 + ISMCTS
- FastAPI 接收盘面 JSON，返回 Top-3 推荐

### 4. Self-Play 训练 — 阶段 4
- 自我对弈数据闭环，3060 Ti FP16 训练
- 目标：单次 10,000 次 MCTS 模拟 ≤ 10 秒

### 5. UI 悬浮窗 (PySide6) — 阶段 5
- 无边框透明置顶，实时展示推荐操作

## 关键决策约束

1. **性能热路径**（simulation / self-play）一律走 Rust，Python 只做胶水
2. **状态必须可序列化**：便于确定化采样（Determinization）、存档、调试
3. **数据单一事实来源**：地图/卡牌/数值以 reference 数据文件为准，避免各模块重复维护

## 一致性校验

- JS（npow）与 Rust 引擎对同一盘面跑随机对弈，比对动作合法性结果
- TTS Mod（ikegami）作为边界规则的二次交叉验证
