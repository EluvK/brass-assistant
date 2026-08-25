# 对局回放与决策诊断

`replay-web` 是 Rust 引擎的开发者调试入口。它由 CLI 创建一次内存会话并在 `127.0.0.1` 提供本地页面；不写入 `.brass-replay` 文件，不承诺跨代码版本可读。CLI 退出后所有状态、步骤和诊断证据均销毁。

## 会话模型

每个座位必须有一个策略。`ReplaySession` 在执行动作前保存完整 `GameState` 快照、规则生成的完整合法动作集和策略产生的 `DecisionTrace`；成功应用动作、推进回合并完成必要的时代清理后，再保存动作后快照。时间线读取已有快照，不会重新执行历史策略或历史动作。

默认是全知调试视图：所有玩家手牌、牌堆、弃牌和盘面均可见。动作包含 lossless canonical 标识和 Rust 生成的展示字段，前端不解析 canonical 字符串。

## 策略契约

`StrategyAdapter` 的输入是当前状态及完整规则合法集，输出实际动作和 `DecisionTrace`。原生 `heuristic`、`random` 和 `mcts` 均通过该契约接入。策略返回不在完整合法集中的 canonical 动作会终止会话并留下失败原因。

启发式 evidence 记录其 shortlist 分数和所有卡牌保留分；完整合法表中不在 shortlist 的动作会明确标为“未被此策略评分”。随机策略只记录实际选择。当前 MCTS evidence 显示用于根候选筛选的启发式先验分，完整的根访问次数/价值统计仍应作为下一步扩展从 `mcts_ai` 导出，而不能伪造为搜索统计。

`python:<worker-config>` 已在 CLI 解析和策略契约中保留，但本版本尚未启动 Python worker；该策略被选择时会明确中止会话。这防止没有定义稳定 stdin/stdout 协议时静默回退为其他策略。后续 worker 应采用逐行 JSON，返回 canonical 选择及网络/ISMCTS evidence，并对超时、进程退出和非法选择返回可诊断失败。

## HTTP 界面

页面固定为顶部运行控制、左侧时间线、中间盘面、右侧全局/玩家状态和底部动作/证据表。前端静态文件直接由 `engine/web/` 提供，Rust 只提供 API 和静态文件宿主，不内嵌页面内容。`Step` 将任务交给唯一的游戏 worker，`Run` 在上一步完成后再安排下一步，`Pause / Refresh` 停止本地循环并刷新。HTTP API 为：

- `GET /api/status`：会话元信息、当前状态、已生成步骤的轻量时间线摘要，以及 `busy`/`complete` 状态。
- `GET /api/steps/:index`：按需读取一手的盘面前后快照、完整合法动作与策略证据。
- `POST /api/step`：立即入队并返回 `202 Accepted`；若当前策略正在运行或会话已结束，返回 `409 Conflict`。结果通过状态轮询读取。

服务仅绑定 loopback。HTTP 连接各自在线程中处理；状态为独立读模型，策略/MCTS 执行不会持有它的锁，因此状态请求不会被长计算阻塞。当前使用轮询控制，而非 WebSocket；后续可将相同会话读模型推送到 WebSocket，不改变回放数据模型。

## 回归约束

- 每一步的 chosen canonical 必须存在于保存的完整合法集，且该集合中的任一动作都能在决策前快照执行。
- 快照恢复后所见盘面、全局信息和当前玩家必须与原始记录一致。
- 固定 seed、座位策略和参数时，原生策略及状态迁移应保持确定性。
