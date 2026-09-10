# 发展路线图

当前实现的架构与契约见 [architecture.md](architecture.md) 与
[ai-action-encoding.md](ai-action-encoding.md)；运行命令见 [ai-tools.md](ai-tools.md)
与 [engine-tools.md](engine-tools.md)。

## 当前位置

- **阶段 1（Rust 引擎）**：完成。规则、合法动作生成、图连通性、快照、回放与批量评测齐备。
- **阶段 2（深度学习）**：表示层已重建。观测是 102 个 token，动作是实体引用，
  网络是小型 transformer + 动作引用打分，价值目标是 VP 效用。
- **阶段 3（self-play）**：入口（`selfplay_train.py` + `selfplay_loop.py`）、rolling
  replay window、对手池、轮换座位 arena 与 `metrics.jsonl` 已具备。**尚未开始正式
  规模训练**，前置问题见下。
- **阶段 4（TTS 数据抽取）/ 阶段 5（UI 悬浮窗）**：未开始。

## 近期：先把搜索的输入搞对

1. **树内动作身份**：已完成。节点保存动作与它支付的**语义卡牌**，每次 simulation 在当前
   determinization 下重新绑定手牌下标（`rebind_cards`），手牌里没有等价卡就剪枝。此前
   深层节点会静默打出另一张牌，污染子树价值，而根节点的访问分配正由这些价值驱动。

2. **Q 初始化未访问孩子**：已落地（`q_init`，默认开启）。`select_child` 用网络给出的
   边价值 `Q(s,a)` 估计从未访问过的孩子，而不是回退到父节点价值；这正是让全合法分支
   可搜索的那一步——否则先验项在 350 分支下小到无法区分兄弟招。

3. **重标定搜索参数**：未做。价值尺度换成零均值 VP 效用、Q 成为未访问孩子的初始估计
   之后，`c_puct` / `prior_top_k` / `fpu` 的既有默认值需要重新标定，并决定
   full-legal 与 top-K 剪枝哪个更划算。

4. **candidate recall**：未测。搜索与训练目前都用 full-legal，所以这条不是阻塞项；一旦
   为了吞吐切到 `--candidate-k`，候选分布就会与训练错位。测量口径：已训练策略在全合法
   集上的 top-1 / top-3 落在 shortlist 内的比例。

在 1–3 完成之前，任何规模的自对弈都是在错误的信号上花钱。

## 中期：imitation → self-play 闭环

前置：上面三步完成，且 imitation 网络在 benchmark 上不弱于 heuristic。

1. **imitation warm start**：`bootstrap_imitation.py` 用 heuristic 教师给出可用先验，
   建立 held-out 验证与固定指标基线（policy top-k、winner 命中、arena 胜率）。
2. **teacher 退役**：用 MCTS visit 分布逐步替换 teacher 作为 policy 目标来源，保留历史
   checkpoint 组成对手池；早期把 imitation 数据按比例混入 replay window 抑制遗忘。
3. **评估纪律**：固定 seed 区间 + 轮换座位 arena，按 Wilson 下界判断强度；`best.pt`
   只决定对手池内容，晋级不用硬门禁（40 局 arena 的标准误约 ±15%）。
4. **PPO / actor-critic**：仅在 self-play + search 稳定后再评估，不是当前阻塞点。

## 远期：表示与不完全信息升级

先由错误案例证明瓶颈，再投入复杂度。

1. **belief 表示**：目前对手手牌是单次 determinization 采样（座位 token 上带
   `hand_is_sampled` 标志）。若错误案例集中在对手竞争判断，用公开历史（已出现卡牌、
   对手动作约束）形成分布替代独立采样。
2. **批量 ISMCTS 的近似**：多个 simulation 共享同一节点评估时，当前只用一个
   determinization 编码。若误差可测，改为每个 determinization 独立建树、只在根聚合。
3. **reward shaping**（如需要）：potential difference 形式，λ 随训练衰减。

## 产品化（阶段 4 / 5）

1. **TTS 数据抽取**：参考 `reference/ikegami-tts-brass`，在 TTS 中读取公共盘面 + 己方
   手牌并 POST 到 localhost。对手手牌不可读——推荐引擎必须在隐藏信息约束下工作，这与
   训练时的 determinization 假设一致。
2. **推荐服务**：复用 `replay_worker.py` 已验证的链路（加载 checkpoint + `RustISMCTS`）；
   延迟预算 10–15 秒，用 `evaluate.py` 的单步计时校准模拟数与批推理参数。
3. **UI 悬浮窗**：PySide6 透明置顶窗口，展示 Top-3 建议与预估收益，数据源为推荐服务
   输出的结构化证据（参考 replay-web 的 evidence 形态）。
