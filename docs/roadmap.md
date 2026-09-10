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
- **阶段 3（Self-Play 训练）**：顶层入口已重建（`selfplay_train.py` +
  `brass_ai/selfplay_loop.py`）：snapshot 形态样本、rolling replay window、
  历史对手池 matchmaking、轮换座位 arena、`latest` / `best` checkpoint 与
  `metrics.jsonl` 均已落地；搜索侧的 `prior_top_k` / FPU 也已落地。
  尚未开始正式规模训练——搜索仍有一个未解决的信号瓶颈，见「阶段 3 的已知问题」。
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

当前状态：顶层入口与评估纪律已先落地（工具链先行），但 imitation 尚未证明稳定
超过 heuristic。正式长跑前仍建议先满足上面的前置条件，否则自对弈会以较弱的
先验起步，容易把 heuristic 的错误固化进 π 目标。

1. **重建 self-play 顶层入口**：基于现有模块组合（`play_game_with_roles` /
   `play_batch` + `SelfPlayPool` + `train_steps` / `run_loop`），不重写底层能力。
   已完成：见 [ai-tools.md](ai-tools.md) 的「Self-play 训练入口（阶段 3）」。
2. **teacher 退役路径**：用 MCTS visit 分布逐步替换 heuristic teacher 作为
   policy 目标来源；保留历史 checkpoint 组成 opponent pool（`mp_selfplay` 的
  matchmaking 钩子已预留）。
   部分完成：policy 目标已完全来自自对弈 visit 分布，opponent pool 已接入；
   尚未做的是把 imitation 数据按比例混入 replay window 以抑制早期遗忘。
3. **评估纪律**：固定 seed 区间 + `sweep_scores` 大样本回归做版本对比；
   跨版本用轮换座位 benchmark 防 seat 偏差。
   部分完成：arena 与 heuristic benchmark 都使用固定 seed + 轮换座位，并按
   Wilson 下界判断强度；`sweep_scores` 版本对比尚未接入。
4. **PPO / actor-critic**：仅在 self-play + search 稳定后再评估是否带来额外
   价值；不是当前阻塞点。

### 阶段 3 的已知问题（按优先级）

1. **搜索有效性**（已量化，机械问题已修，信号问题未解）

   现象：full-legal 根分支为 114–600。`batch_size=64`、500 次模拟只访问 15–36 个
   根子节点，而且**模拟数从 64 加到 1000，π 目标的形状几乎不变**（top8 占比
   0.70 → 0.63，top1 0.156 → 0.118）。搜索退化成先验的弱锐化器，14 倍的模拟量
   没有换来更好的策略目标。

   直接原因：PUCT 的探索项 `c_puct · P · sqrt(N) / (1 + n)` 在 P ≈ 1/350、N = 500、
   c = 2.5 时只有约 0.08，而任何已评估孩子的 Q ≈ 0.3，因此未访问的孩子永远选不
   中，额外模拟只是堆在同一批孩子上。

   已修复（`NnMctsConfig::prior_top_k` / `fpu`）：

   - `prior_top_k=K`：仍对全部合法动作打分算出先验，但树里只保留先验最高的 K 个
     孩子。K = 16 时 16 个孩子全部获得访问（此前是 15–36 个中任意一批）。
   - FPU：未访问孩子的 Q 取父节点值而不是 0。本项目的 value 是 `1 - rank/n`，取值
     [0, 0.75]、均值约 0.3，用 0 做 baseline 等于判定"未访问 = 最差"，与
     AlphaZero 的零均值 [-1, 1] 前提不符。

   仍未解决：即使 K = 16、`c_puct` 降到 0.1，根节点 top1 访问占比也只有 0.20
   （均匀分布是 0.0625），π 依旧偏平。根因是**价值头在兄弟层面的分辨力不足**：

   | 量 | 实测 |
   | --- | --- |
   | top-16 孩子的一步价值 sd | 0.014 |
   | top-16 孩子的一步价值极差 | 0.037–0.056 |
   | 单次评估随 determinization 的波动 sd | 0.0075 |
   | 价值头自身误差（rank MSE 0.019） | RMSE ≈ 0.14 |

   即：真正的兄弟差异（0.014）比价值头自身的误差小一个数量级，而 K = 16、c = 2.5
   时的探索项量级约 0.08，仍然压过 0.014。访问量因此由先验而非价值分配。

   已经排除的方向：**加深搜索**。沿先验 top-8 孩子各推进 1/3/5/7 步后测同一座位的
   价值，兄弟 sd 分别为 0.0176 / 0.0212 / 0.0136 / 0.0155——不随深度增长。这是合理的：
   伯明翰是长经济局，单步好坏相对终局名次本就是小量。所以"把树搜得更深"不会带来
   更大的兄弟区分度，只会更贵。

   也就是说，搜索现在的行为已经是价值头允许的上限：访问分配确实由 Q 驱动（K=16、
   c=0.1 时 top1 ≈ 0.20 与 Q 差 0.014 的理论平衡点一致），只是 Q 差本身就小。
   兄弟排序能力的直接测量（`python/bench_value_ranking.py`，以终局 rollout 为参照，
   36 个局面 × 先验 top-6）：V 头与 winner 头的 within-position Spearman 都落在
   ±0.1 以内，即**测不出排序兄弟招的能力**。

   已否决的方向：**中心化 value 目标**。`rank_head` 是带 bias 的线性层，常数偏移对它是
   免费的；减掉均值不改变兄弟之间的**差分**，而搜索只消费差分。

   当前方向：**动作条件化价值（Q 头）**。V 头只接收状态，兄弟招对它是状态微扰；Q 头把
   候选动作特征（与 policy 头共用 FiLM 调制表示）直接作为输入，"招 A 优于招 B"是一阶差异。
   网络头与损失已落地（见 [ai-action-encoding.md](ai-action-encoding.md) §5.3），搜索尚未消费它。

   下一步（按顺序）：

   a. 重跑 bootstrap 让 Q 头真正训起来，然后用 `bench_value_ranking.py` 做 go/no-go：
      Q 的兄弟排序相关必须显著高于 V（当前 V ≈ 0 ± 0.08）。
   b. go 之后再把 Q 接进 `select_child`：用边的 Q 初始化未访问孩子的价值，替代 FPU 的
      父值回填。这是把"能排序"变成"搜索真的据此分配访问"的一步。
   c. 动作分解（先选动作类型，再选地点/卡牌/资源来源）仍然排在后面：它解决吞吐与搜索
      宽度，不解决上面的信号问题。
2. **树节点跨 determinization 复用**（见 [ai-tools.md](ai-tools.md) 的搜索自检约定）：
   节点保存的是手牌下标，对手手牌每次 simulation 重采样后会指向另一张牌。
   非 Build 动作只做越界检查，因此会静默打出不同的牌（实测占树内动作的
   30–55%）。长期方案是节点保存语义动作并在当前 determinization 重新解析，
   或每棵 determinization 独立建树后只聚合根访问。
3. **candidate recall 仍未测量**：full-legal 展开下搜索不受 shortlist 限制，
   但一旦为了吞吐切到 `--candidate-k`，训练与搜索的候选分布就会错位，必须先测。

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
   checkpoint + `RustISMCTS`）；延迟预算 10~15 秒，用 Python 端 NN-MCTS
   基准（`evaluate.py` 的 `benchmark_*`、`RustISMCTS.search` 单步计时）校准
   模拟数与批推理参数。
3. **UI 悬浮窗**：PySide6 透明置顶窗口，展示 Top-3 建议与预估收益；数据源为
   推荐服务输出的结构化证据（可参考 replay-web 的 evidence 形态）。
