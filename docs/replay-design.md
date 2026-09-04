# 对局回放与决策诊断

`replay-web` 是 Rust 引擎的开发者调试入口。它由 CLI 创建一次内存会话并在 `127.0.0.1` 提供本地页面；不写入 `.brass-replay` 文件，不承诺跨代码版本可读。CLI 退出后所有状态、步骤和诊断证据均销毁。

## 会话模型

每个座位必须有一个策略。`ReplaySession` 在执行动作前保存完整 `GameState` 快照、规则生成的完整合法动作集和策略产生的 `DecisionTrace`；成功应用动作、推进回合并完成必要的时代清理后，再保存动作后快照。时间线读取已有快照，不会重新执行历史策略或历史动作。

默认是全知调试视图：所有玩家手牌、牌堆、弃牌和盘面均可见。动作包含 lossless canonical 标识和 Rust 生成的展示字段，前端不解析 canonical 字符串。

## 策略契约

`StrategyAdapter` 的输入是当前状态及完整规则合法集，输出实际动作和 `DecisionTrace`。原生 `heuristic`、`random` 与 `python:<worker-config>` 网络座位均通过该契约接入。策略返回不在完整合法集中的 canonical 动作会终止会话并留下失败原因。

启发式 evidence 记录其 shortlist 分数和所有卡牌保留分；完整合法表中不在 shortlist 的动作会明确标为“未被此策略评分”。随机策略不做评分，其完整合法表同样逐行列出并标注“未被此策略评分”。

`python:<worker-config>` 通过 `PythonWorkerStrategy` 实现：每个此类座位启动一个独立的 Python worker 子进程（`python -u -m brass_ai.replay_worker <worker-config>`，`PYTHONPATH` 指向仓库 `python/`；worker-config 按空白切分，含空格的参数可用单/双引号包裹），worker 持有一个网络 checkpoint 并通过 stdin/stdout 的逐行 JSON 协议应答：

- 启动握手：worker 加载 checkpoint 后输出 `{"type":"ready","name":...,"meta":{ckpt, mode, sims, device, action/state feature schema 版本}}`；Rust 在会话创建时等待握手，坏 checkpoint 或解释器缺失会在 HTTP 服务启动前报错退出。
- 决策请求：Rust 发送 `{"type":"choose","request_id":N,"snapshot":"<base64>","legal":[...]}`。snapshot 采用与 pyo3 `GameState.snapshot()` 完全相同的字节格式（magic + version + `snapshot_bytes`），worker 用 `GameState.from_snapshot` 无损还原完整局面（含 RNG 与全部手牌）。
- 决策响应：`{"type":"choice","request_id":N,"canonical":...,"evidence":{mode, policy, visits, root_value}}`；单次请求失败时返回 `{"type":"error",...}` 并保持存活。
- 超时（`--worker-timeout`，默认 300 秒）、进程退出和返回不在合法集中的 canonical 均产生可诊断的会话失败。

worker 支持 `--mode mcts`（默认，Rust ISMCTS + 网络引导，按根访问数 argmax 决策，等价 temperature=0）与 `--mode policy`（网络对全部合法候选一次前向后直接 argmax）。两种模式都会额外做一次全候选前向，返回覆盖全部合法动作的 policy 概率与当前玩家的 value 头估计。

`DecisionTrace` 对网络座位使用 `evidence_kind` 为 `net-mcts` / `net-policy` 的 evidence：完整合法表按结构操作逐行给出，`score` 列在 mcts 模式下为根访问次数（未被搜索展开的动作明确标注），policy 概率始终出现在 note 中；`root_value` 为当前玩家的网络价值估计。原生策略的 `root_value` 为 `None`。

网络座位不做确定性承诺（GPU/浮点非确定性）；"固定 seed 保持确定性"的回归约束仅覆盖原生策略。

## HTTP 界面

页面固定为顶部运行控制、左侧时间线、中间盘面、右侧全局/玩家状态和底部动作/证据表。前端静态文件直接由 `engine/web/` 提供，Rust 只提供 API 和静态文件宿主，不内嵌页面内容。`Step` 将任务交给唯一的游戏 worker，`Run` 在上一步完成后再安排下一步，`Pause / Refresh` 停止本地循环并刷新。HTTP API 为：

- `GET /api/status`：会话元信息、当前状态、已生成步骤的轻量时间线摘要，以及 `busy`/`complete` 状态。
- `GET /api/steps/:index`：按需读取一手的盘面前后快照、完整合法动作与策略证据。
- `POST /api/step`：立即入队并返回 `202 Accepted`；若当前策略正在运行、会话已结束或已失败，返回 `409 Conflict`。结果通过状态轮询读取。

服务仅绑定 loopback。HTTP 连接各自在线程中处理；状态为独立读模型，策略/MCTS 执行不会持有它的锁，因此状态请求不会被长计算阻塞。当前使用轮询控制，而非 WebSocket；后续可将相同会话读模型推送到 WebSocket，不改变回放数据模型。

## 回归约束

- 每一步的 chosen canonical 必须存在于保存的完整合法集，且该集合中的任一动作都能在决策前快照执行。
- 快照恢复后所见盘面、全局信息和当前玩家必须与原始记录一致。
- 固定 seed、座位策略和参数时，原生策略及状态迁移应保持确定性。
