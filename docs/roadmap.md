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

这三件事没做完之前，任何规模的自对弈都是在错误的信号上花钱。

1. **树内动作身份**（正确性，最高优先级）

   搜索树节点保存的是 `ResolvedMove`，而 `ResolvedMove` 记录的是手牌下标。树节点活得
   比一次 determinization 长，对手手牌每次 simulation 会重采样，于是深层节点会指向
   另一张牌：规则重新校验的动作（Build）被拒，只做越界检查的动作（Network / Develop /
   Sell / Loan / Pass）会静默打出另一张牌。后者实测占树内动作的 30–55%，直接污染子树
   的价值估计，而根节点的访问分配正由这些价值驱动。

   做法见 [ai-action-encoding.md](ai-action-encoding.md) §5：节点保存结构动作与语义
   卡牌，每次 simulation 在当前 determinization 下重新解析，解析不到就剪枝。
   `failed_applies` / `rewritten_applies` 两个计数器保留，用来验证修复效果——修好后
   前者应当接近 0，后者应当为 0。

2. **把 Q 接进搜索**

   `q_head` 已经随训练一起训练，`flush_net` 也已把它取回，但 `select_child` 还没有用它。
   计划：用边的 Q 初始化未访问孩子的价值，替代 FPU 的父值回填。

   前置判据是 `python/bench_value_ranking.py`：Q 的 within-position Spearman 必须显著
   高于 V。V 只看状态，兄弟招对它只是状态微扰；Q 直接接收动作引用的实体，兄弟差异对
   它是一阶量。

3. **candidate recall**

   搜索与训练目前都用 full-legal，所以这条不是阻塞项；一旦为了吞吐切到
   `--candidate-k`，候选分布就会与训练错位。测量口径：已训练策略在全合法集上的
   top-1 / top-3 落在 shortlist 内的比例。

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
