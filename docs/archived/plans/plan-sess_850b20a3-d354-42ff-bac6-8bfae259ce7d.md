# 让网络 checkpoint 坐进 replay-web 对战

## 背景结论

replay-web 每个座位通过 `StrategyAdapter` 策略缝接入；`StrategySpec::Python { worker_config }` 已在 CLI 与 docs/replay-design.md 中预留但未实现（当前直接中止会话），文档规定了应采用「逐行 JSON 的 stdin/stdout worker 协议」。AI 侧 checkpoint 加载（`torch.load` → `net.load_state_dict(ckpt["model"])`）、`RustISMCTS`（Rust 搜索 + Python 网络回调）、PyO3 `snapshot()/from_snapshot()` 全部现成，缺的只是把它们接进策略缝。

方案：Rust 按文档实现 `PythonWorkerStrategy`——为每个网络座位 spawn 一个 Python worker 子进程，通过 stdin/stdout 逐行 JSON 通信，worker 持有 checkpoint 并做决策。保持纯内存会话不变。

## 一、worker 协议（docs/replay-design.md 已预留的缝）

- 启动握手：worker 就绪后输出 `{"type":"ready","name":...,"meta":{ckpt, mode, sims, device, action_feature_schema_version}}`；Rust 带超时读取，失败即报错退出（坏 checkpoint 路径在 HTTP 服务启动前就被发现）。
- 决策请求：`{"type":"choose","request_id":N,"snapshot":"<base64>","legal":["canonical",...]}`。snapshot 为 PyO3 兼容格式（`SNAPSHOT_MAGIC` + version + `state.snapshot_bytes()`，Rust 侧拼装，worker 直接 `be.GameState.from_snapshot()` 还原，无损含 RNG）。
- 决策响应：`{"type":"choice","request_id":N,"canonical":...,"evidence":{"mode":"mcts"|"policy","policy":{canonical: 概率, 覆盖全部合法动作},"visits":{canonical: 根访问数},"root_value": float}}`；出错时 `{"type":"error",...}` 或进程退出。
- 超时/EOF/返回不在合法集的 canonical → Rust 返回可诊断 Err（沿用文档要求）。
- stdin EOF 时 worker 自然退出；Rust Drop 时关 stdin + kill 兜底。stderr 直接继承到终端（torch 日志可见）。

## 二、Rust 侧

1. **新文件 `engine/src/ai/python_worker.rs`**：
   - `PythonWorkerStrategy`：持有子进程 stdin、reader 线程 + channel（实现超时用 `recv_timeout`）、request_id 计数、label。
   - `spawn(python_bin, worker_config, timeout) -> Result<Self, String>`：`python -u -m brass_ai.replay_worker <worker_config 原样透传>`，`PYTHONPATH` 注入 `{CARGO_MANIFEST_DIR}/../python`（与 replay_web 用 CARGO_MANIFEST_DIR 找 web 静态文件同一模式）。
   - 实现 `StrategyAdapter::choose`：编码请求 → 等响应 → 校验 canonical ∈ legal → 把 evidence 映射为 `DecisionTrace`（`evidence_kind = "net-mcts" | "net-policy"`；对完整合法集按操作逐行生成 `ActionDto`：`score` = 根访问数（mcts 模式，未展开为 None）、`note` 含 policy 概率与「未被搜索展开」标记、`card` 保留现有 move_card_name）。
2. **`engine/src/ai/replay.rs`**：
   - `DecisionTrace` 增加 `root_value: Option<f64>`（native 策略为 None）——兑现文档中「完整根访问/价值统计」的下一步扩展。
   - `ReplaySession::new` 按 spec 构建适配器：`Python` → `PythonWorkerStrategy`，其余 → `NativeStrategy`；`NativeStrategy` 中的 Python 分支报错代码移除。
3. **`engine/src/bin/replay_web.rs`**：新增 CLI `--python-bin`（默认 `python`）、`--worker-timeout`（默认 300 秒）。用法：
   ```sh
   cargo run --release -p brass-engine --bin replay_web -- --seed 7 --players 4 \
     --player heuristic \
     --player "python:--ckpt checkpoints/bootstrap-0830-smoke.pt --sims 200 --device cpu" \
     --player heuristic --player random
   ```
4. **`engine/Cargo.toml`**：加 `base64` 依赖。

## 三、Python 侧

1. **新文件 `python/brass_ai/replay_worker.py`**：
   - checkpoint 加载（校验 schema 版本字段，与 Trainer.load_state_dict 同一套约束）；`--mode mcts|policy`（默认 mcts）、`--sims`、`--device`（默认 cuda 可用则 cuda）。
   - `handle_request(snapshot_bytes, legal, cfg) -> dict` 拆成独立可测函数：`GameState.from_snapshot` 还原 → 全候选一次前向得 policy 概率 + value 头（作为 root_value）→ mcts 模式再跑 `RustISMCTS.search`（argmax 访问数选步，等价 temperature=0）；policy 模式直接按 policy argmax。返回 evidence dict。
2. **`python/tests/test_replay_worker.py`**：对 `handle_request` 直测（真实 snapshot + `checkpoints/bootstrap-0830-smoke.pt`，checkpoint 不存在则 skip）。

## 四、前端 `engine/web/replay.html`（小改）

- `renderActions` 按 `trace.evidence_kind` 动态列头（net-mcts 时 score 列显示「访问」）；证据面板展示 `root_value`（存在时）。时间线/策略名展示已通用，无需改动。

## 五、文档同步

- `docs/replay-design.md`：worker 协议从「预留」改为「已实现」，写明协议格式、两种 evidence_kind 的语义、net 座位不做确定性承诺（GPU 非确定性）。
- `docs/engine-tools.md`：`replay-web` 章节补网络座位用法示例。

## 六、验证

1. `cargo test -p brass-engine`（现有 replay 回归测试全过）。
2. `pytest python/tests/test_replay_worker.py`。
3. 手动冒烟：按上面命令起 replay-web（一个网络座位 + heuristic + random，`--sims 64 --device cpu`），浏览器观察网络座位的动作表（访问数/policy 概率/root_value）与整局推进。