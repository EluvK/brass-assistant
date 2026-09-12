### Brass: Birmingham 从引擎性能优化到强化学习自对弈进化的完整工程实施方案

#### 阶段零：Rust 引擎启发式对弈流水线极致性能优化（P0 前置依赖）
1. **消除堆字符串分配（枚举哈希替代 `format!` 去重）**：
   - 改造 `engine/src/ai/heuristic_ai/mod.rs` 中的 `operation_key`，使用紧凑的纯数值哈希结构体替代 `format!("build:{loc:?}...")`，彻底消除每步数百次字符串分配开销。
2. **分支定界与启发式前瞻剪枝（Bound Pruning）**：
   - 在 `choose_lookahead` 中对首动列表按自身评分降序排序。
   - 引入最大加分上界估计，当首动当前得分加上理论最高二动加分依然无法超越全局最优时，提前截断，跳过昂贵的 `GameState::clone` 与后续分支搜索。
3. **手牌反向驱动的建造槽位生成（Card-directed Generation）**：
   - 优化 `score_top_builds`：优先根据玩家当前持有的最多 8 张手牌反查可建城市与行业，杜绝无脑扫描全图 25 个城市所有槽位再因无手牌截断的算力浪费。
4. **零拷贝借用型快照序列化**：
   - 重构 `StateSnapshot` 为借用切片结构体 `StateSnapshotBorrow<'a>`，消除 `snapshot_bytes()` 内部十几个 `Vec` 的冗余深拷贝。
   - **阶段目标**：将启发式单局耗时从 113.9ms 压缩至 25ms 以内，16 线程并行吞吐提升至 500+ 局/秒。

#### 阶段一：纯 Rust 原生百万级专家数据生成与绝对得分预训练
1. **编写 Rust 独立并行数据生成工具（`engine/src/bin/gen_imitation.rs`）**：
   - 基于 Rayon 多线程直接调度优化后的启发式引擎，过滤低质局，几十分钟内直出 50 万~100 万局二进制紧凑快照分片文件（`.bin`），彻底摆脱 Python 缓慢的单进程导出。
2. **升级多任务网络架构与绝对得分损失（`net.py`, `train.py`）**：
   - 在 `BrassNet` 中新增无界线性头 `abs_vp_head`，预测标准化绝对分 $(VP - 100) / 50$。
   - 损失函数采用 `SmoothL1Loss`（抗极端 180~200 分长尾噪音）。
3. **离线大 Batch 蒸馏基准验证**：
   - 完成预训练后，验证纯 Policy 动作（零搜索）对抗 3 个 Rust 老师，均分达到 115+ 分，奠定稳健基本盘。

#### 阶段二：打通纯 Policy 向量化极速对弈引擎（0.05 秒/局）
1. **构建 Python 极速走子控制器（`FastPolicyPlayer`）**：
   - 剥离 MCTS 树维护，直接以网络输出的 `candidate_logits` 执行带温度系数 $\tau$ 的概率采样走子。
2. **GPU 集中向量化批处理（Batched Actor Loop）**：
   - 在主进程维护 32~64 个并发 `GameState` 实例，每步集中打包状态 Tensor 一次性送入 GPU 前向，单秒产出 20~50 局完整对局，跑完 10 万局缩短至 1 小时内。

#### 阶段三：强化学习策略迭代、自适应 KL 锚定与考官门禁（突破 120 分）
1. **统一强化学习目标函数设计**：
   $$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{policy-adv}} + \alpha_1 \mathcal{L}_{V_{\text{rel}}} + \alpha_2 \mathcal{L}_{V_{\text{abs}}} + \beta \cdot D_{\text{KL}}(\pi_{\text{anchor}} \parallel \pi_{\theta})$$
   - 奖励函数融合：绝对经济分（做大蛋糕）+ 相对胜负优势 + 独赢加成 + 破产/低分惩罚。
2. **轻量自适应 KL 散度约束实现**：
   - 采样对局轨迹时直接记录当步动作分布概率，反向传播时无需重复推理 Base 模型。
   - 动态自适应调节 $\beta$：前期严格约束防策略崩溃，后期随收敛逐步放开探索高分奇招。
3. **防退化考官联赛（League Training）与阶梯晋升**：
   - 对局池按 60% 自对弈 + 40% 混入 120 分 Rust 老师与历史 Champion 进行对抗。
   - 晋升硬门槛：候选模型必须面对 Rust 老师胜率 $>35\%$ 且均分超 120 分，方可晋升为新 Champion 并覆写为下一代 Anchor 模型。

#### 阶段四：实战与推断阶段精准 MCTS 挂载（高阶辅助阶段）
- 仅在部署端和实时辅助时挂载 MCTS，且只对 Policy 剪枝出的前 5~10 个高优先级动作执行浅层 MCTS，消除训练期的搜索算力开销。