# 发展路线图

当前实现的架构与契约见 [architecture.md](architecture.md) 与
[ai-action-encoding.md](ai-action-encoding.md)；运行命令见 [ai-tools.md](ai-tools.md)
与 [engine-tools.md](engine-tools.md)。

## 当前位置

- **阶段 1（Rust 引擎）**：完成。规则、合法动作生成、图连通性、快照、回放与批量评测齐备。
- **阶段 2（深度学习）**：表示层已重建。观测是 102 个 token，动作是实体引用，
  网络是小型 transformer + 动作引用打分，价值目标是 VP 效用。搜索侧的两个基础问题
  也已处理：树内动作身份按语义卡牌重新绑定（跨 determinization 复用不再会打出
  另一张牌），未访问孩子由网络给出的 `Q(s,a)` 初始化。
- **阶段 3（self-play）**：入口（`selfplay_train.py` + `selfplay_loop.py`）、rolling
  replay window、对手池、轮换座位 arena 与 `metrics.jsonl` 已具备。**尚未开始正式
  规模训练**，前置问题见下。
- **阶段 4（TTS 数据抽取）/ 阶段 5（UI 悬浮窗）**：未开始。

## 近期：把 imitation 训到能用的水平

1. **规模化 imitation（当前唯一阻塞项）**

   探针显示 200 局只有约 194 个优化器步，等价于没训：teacher 有 316–600 个候选，
   policy CE 从 `ln(N)≈5.7` 只降到 4.9。训练预算要按**步数**规划——2000 局约 25 万
   样本，每个 epoch 约 970 步。

   判据：policy CE 降到 3 以下；value 的 MSE 明显低于"恒输出 0"的基线（0.0847，
   即终局效用自身的方差）。注意价值头一局只有 4 个标签，它的有效样本量是**局数**，
   不是样本数，所以它的收敛需要比 policy 更多的对局。

2. **重标定搜索参数**：依赖 1 的产物。价值尺度换成零均值 VP 效用、Q 成为未访问孩子的
   初始估计之后，`c_puct` / `prior_top_k` / `fpu` 的既有默认值需要重新标定，并决定
   full-legal 与 top-K 剪枝哪个更划算。`bench_value_ranking.py` 是这一步的前置判据。

3. **candidate recall**：依赖 1 的产物。搜索与训练目前都用 full-legal，所以这条不是
   阻塞项；一旦为了吞吐切到 `--candidate-k`，候选分布就会与训练错位。测量口径：已训练
   策略在全合法集上的 top-1 / top-3 落在 shortlist 内的比例。

在这三步完成之前，任何规模的自对弈都是在错误的信号上花钱。

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
   `SEAT_HAND_SAMPLED_FLAG` 标志）。若错误案例集中在对手竞争判断，用公开历史（已出现卡牌、
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
