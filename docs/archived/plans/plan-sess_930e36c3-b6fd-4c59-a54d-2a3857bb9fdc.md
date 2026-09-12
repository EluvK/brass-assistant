# 强化学习自对弈系统补齐实施方案 (Zero-MCTS Fast Self-Play & RL League)

## 一、背景与核心痛点
在上一阶段的开发中，项目完成了底层核心组件的开发（包括 Rust 引擎优化、Rust 离线数据生成 `gen_imitation`、网络绝对分头 `abs_vp`、纯 Policy 推理算子 `FastPolicyPlayer` 与多环境向量化 `VectorizedSelfPlay`），并在 `docs/archived/plans/plan-sess_ae8bd5fb-9241-4f83-abb5-08ec6c6c945d.md` 中规划了完整的自对弈闭环。

但在落地执行时，留下了几个关键断档（Gap）：
1. **KL 散度正则断档**：`compute_kl_loss` 仅停留在单测中，未实际接入训练损失计算，无法防止纯策略自博弈下的策略崩溃与退化。
2. **策略更新缺乏优势加权**：当前向量化自对弈对所有采样动作做无差别“自行为克隆”，没有结合终局得分进行优势加权（Advantage Weighting）或高胜率过滤，容易自我强化昏招。
3. **自对弈席位单一**：`VectorizedSelfPlay` 目前仅支持单一网络自对弈，尚未支持按比例混入 120 分 Rust 启发式老师，无法打破自对弈共谋盲区。
4. **训练主程序与配套文档需深度整合**：新创建的 `python/fast_selfplay_train.py` 需要将上述能力完整集成，并提供详实的参数控制、指标监控与测试保障。

---

## 二、架构设计与实施模块

### 模块 1：训练器与损失函数升级（接入 KL 散度与样本优势加权）
* **涉及文件**：`python/brass_ai/train.py`、`python/brass_ai/selfplay.py`
* **设计目标**：
  1. 在 `TrainConfig` 中新增 `kl_lambda: float = 0.0`（KL 散度惩罚权重）与 `advantage_weighting: bool = True` 开关。
  2. 扩展 `Sample` 数据结构，允许携带 `weight: float`（样本优势权重）与可选的 `anchor_probs: np.ndarray`（基准模型动作分布先验）。
  3. 修改 `compute_loss`：
     - 若样本批次包含 `weight`，则将 Policy 损失和 Q 损失按优势权重加权（高分胜局强正向引导，低分昏局降权或忽略）。
     - 若 `kl_lambda > 0` 且样本包含 `anchor_probs`，调用 `rl_league.compute_kl_loss` 计算 $D_{\text{KL}}(\pi_{\text{anchor}} \parallel \pi_{\theta})$ 并计入总损失，防范灾难性策略偏离。

### 模块 2：纯策略自博弈轨迹生成强化（优势加权与质量过滤）
* **涉及文件**：`python/brass_ai/fast_policy.py`
* **设计目标**：
  1. 重构 `trajectory_to_samples(traj: GameTrajectory, min_vp_filter: float = 0.0, use_advantage: bool = True)`：
     - **门禁过滤**：支持丢弃最低分低于门限（如 30 分）的破产/死锁垃圾轨迹。
     - **优势标定**：计算每个座位的相对终局优势 $Adv_p = (VP_p - \overline{VP}) / 50$。
     - **动态权重**：获胜者及高于均分的动作赋予更高置信权重（例如 $w = \mathrm{clip}(\exp(Adv), 0.2, 3.0)$），让模型优先学习顺风局打法。
  2. 支持在 `RolloutStep` 中记录当步网络输出的完整动作概率 `action_probs` 作为后续自博弈或下一代 Anchor 的分布锚点。

### 模块 3：向量化对弈支持异构席位（混入 Rust 启发式老师）
* **涉及文件**：`python/brass_ai/fast_policy.py`
* **设计目标**：
  1. 在 `VectorizedSelfPlay` 中新增 `heuristic_prob: float = 0.0` 参数。
  2. 在多环境并发推演的主循环中：
     - 每局初始化时，为每个环境随机决定是否指派 1~3 个席位由 `HeuristicRoundPlayer` 担任。
     - 轮到启发式席位时，直接由 Rust 启发式 2-ply 规划走步，不占 GPU 推理 batch。
     - 仅采集当前神经网络座位的轨迹与样本，使网络在实战中直接向 120 分 Rust 老师学习破局思路。

### 模块 4：端到端训练脚本完善与自动门禁沉淀
* **涉及文件**：`python/fast_selfplay_train.py`
* **设计目标**：
  1. 完整暴露新增控制参数：
     - `--kl-lambda`：自适应 KL 散度约束权重（默认 0.1）。
     - `--heuristic-prob`：混入 Rust 老师席位的概率（默认 0.25）。
     - `--min-vp-filter`：劣质局过滤门限（默认 30.0）。
     - `--advantage-weighting`：是否启用优势加权。
  2. **Anchor 自动滚动升级机制**：
     - 当新模型通过考官门禁评测（`evaluate_vs_heuristic_teachers` 胜率 $\ge 35\%$ 且均分 $\ge 120$）晋升为新 Champion 时，将其权重自动更新为下一阶段的 Anchor Base，实现阶梯式自对弈强化进化。

### 模块 5：测试验证与文档同步
* **涉及文件**：`python/tests/test_fast_policy.py`、`python/tests/test_rl_league.py`、`docs/training-guide.md`
* **设计目标**：
  1. 编写新增特性的自动化单元测试：
     - 测试 `VectorizedSelfPlay` 混入启发式席位时的轨迹生成正确性与得分有效性；
     - 测试 `Trainer` 在启用 `kl_lambda` 与样本 `weight` 时的梯度反向传播正常性。
  2. 更新文档，将完整的自对弈强化学习理论与命令指南沉淀至手册中。

---

## 三、实施步骤与验证计划

1. **Step 1 (Trainer 与数据结构)**：在 `selfplay.py` 和 `train.py` 中引入 `weight` 与 `anchor_probs` 支持，并在 `compute_loss` 中实际集成 `compute_kl_loss`。
2. **Step 2 (轨迹优势计算与过滤)**：在 `fast_policy.py` 中升级 `trajectory_to_samples`，实现优势赋权与最低分过滤。
3. **Step 3 (向量化异构对局)**：在 `VectorizedSelfPlay` 中支持启发式老师混战席位。
4. **Step 4 (集成至训练 CLI)**：升级 `fast_selfplay_train.py`，串接动态 Anchor、KL 正则与考官晋升机制。
5. **Step 5 (全套验证)**：运行自动化单元测试套件 `pytest python/tests`，并执行端到端快速冒烟测试验证训练、门禁和损失收敛。
