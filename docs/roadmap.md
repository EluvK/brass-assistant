# 发展路线图

分阶段的后续发展规划。当前实现的架构与契约见 [architecture.md](architecture.md)，
已实现的运行命令见 [ai-tools.md](ai-tools.md) 与 [engine-tools.md](engine-tools.md)；
影响路线选择的设计原则见 [ai-advise.md](ai-advise.md)。

## 当前位置（2026-08）

按 AGENTS.md 的阶段划分：

- **阶段 1（Rust 引擎）**：完成。规则、合法动作生成、图连通性、快照、
  回放与批量评测工具齐备。
- **阶段 2（深度学习）**：主体完成。state/action 特征编码（schema v4）、
  candidate-scoring 网络（policy/rank/winner/econ 头）、Rust ISMCTS +
  Python 网络回调、启发式教师 imitation bootstrap 入口均已落地。
- **阶段 3（Self-Play 训练）**：模块能力保留（`selfplay.py` / `mp_selfplay.py` /
  `train.py::run_loop`），顶层训练入口未重建。
- **阶段 4（TTS 数据抽取）/ 阶段 5（UI 悬浮窗）**：未开始。

训练状态：full-legal imitation 处于早期；candidate recall 未测量。

## 近期：训练与验证闭环（阶段 2 收尾）

目标：证明"网络 + 搜索"这条链路在 imitation 下能稳定逼近并超越 heuristic。

1. **v4 imitation 训练迭代**：加大样本量与训练轮数；建立 held-out teacher
   validation 与固定指标基线（policy top-k、winner 命中、熵、benchmark 胜率）。
2. **测量 candidate recall**（定义见 [ai-action-encoding.md](ai-action-encoding.md) §7）：
   已训练策略在全合法集上的 top-1/top-3 落在 shortlist 内的比例。这是候选生成器
   上限的直接度量，也是当前方案最大的风险项——搜索只看到全合法集的一小部分。
3. **shortlist hard negatives 实验**：向 shortlist 训练候选加入被 teacher 排除的
   高价值备选，缩小训练与 MCTS 推理的候选分布差异。
4. **规模路径**：从 smoke 到正式训练的资源预算（`--max-candidate-batch` 显存、
   `--materialize-workers` CPU），固化可复现的正式训练命令。

验收信号：轮换座位的 benchmark（net-MCTS vs heuristic）胜率显著高于 50%，
且 candidate recall 指标稳定。

## 中期：self-play 闭环（阶段 3）

前提：近期阶段完成，且 imitation 网络在 benchmark 上不弱于 heuristic。

1. **重建 self-play 顶层入口**：基于现有模块组合（`play_game_with_roles` /
   `play_batch` + `SelfPlayPool` + `train_steps` / `run_loop`），不重写底层能力。
2. **teacher 退役路径**：用 MCTS visit 分布逐步替换 heuristic teacher 作为
   policy 目标来源；保留历史 checkpoint 组成 opponent pool（`mp_selfplay` 的
   matchmaking 钩子已预留）。
3. **评估纪律**：固定 seed 区间 + `sweep_scores` 大样本回归做版本对比；
   跨版本用轮换座位 benchmark 防 seat 偏差。
4. **PPO / actor-critic**：仅在 self-play + search 稳定后再评估是否带来额外
   价值；不是当前阻塞点。

## 远期：表示与不完全信息升级

先由错误案例证明瓶颈，再投入复杂度（原则见 [ai-advise.md](ai-advise.md)）：

1. **手牌/历史 token 化**：玩家全局量、手牌、历史目前仍是 flatten vector；
   当错误案例集中在"卡牌保留价值 / 对手竞争"判断时再做 token 化。
2. **belief 表示**：以公开历史（已出现卡牌、对手动作约束）形成对手手牌分布，
   替代纯 determinization 的独立随机采样。
3. **reward shaping**（如需要）：potential difference 形式，λ 随训练衰减，
   避免人工经济指标永久改变"争取第一名"的目标。
4. **schema 升级纪律**：增量升级 + bump schema version（维护清单见
   [ai-action-encoding.md](ai-action-encoding.md) §9）。

## 产品化（阶段 4 / 5）

1. **TTS 数据抽取**：参考 `reference/ikegami-tts-brass` 的 Lua 脚本，在 TTS
   中隐蔽读取公共盘面 + 己方手牌，POST 到 localhost。对手手牌不可读——推荐
   引擎须在对手手牌隐藏的真实约束下工作（与训练时 determinization 的假设一致）。
2. **推荐服务**：复用 replay worker 已验证的链路（`replay_worker.py` 加载
   checkpoint + `RustISMCTS`）；延迟预算 10~15 秒，用 `mcts_lab bench` 校准
   模拟数与批推理参数。
3. **UI 悬浮窗**：PySide6 透明置顶窗口，展示 Top-3 建议与预估收益；数据源为
   推荐服务输出的结构化证据（可参考 replay-web 的 evidence 形态）。
