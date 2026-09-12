# AI 训练与模型管理实战手册 (AI Training & Model Lifecycle Guide)

本文档是《工业革命：伯明翰》AI 策略模型的标准操作手册，涵盖全套训练命令、核心指标物理含义速查、语义化版本 (SemVer) 资产管理规范、多模型对战（League）设计以及常见故障排查。

2026-09-12：搜索与破产规则修复后，训练始终由通过验证的 best 生成对局，latest 作为候选继续学习。晋升要求 arena 达标、教师评测无零分且均分不低于 best。短程实验与更大样本的独立评测仍是必要步骤，不能以训练 loss 或单轮均分保证棋力增长。

---

## 目录
1. [训练全流程流水线 (The Golden Pipeline)](#一训练全流程流水线-the-golden-pipeline)
2. [模型版本规范与资产分层 (SemVer & Storage)](#二模型版本规范与资产分层-semver--storage)
3. [核心指标速查字典 (Metric Dictionary)](#三核心指标速查字典-metric-dictionary)
4. [多模型对战体系 (League & Tournament)](#四多模型对战体系-league--tournament)
5. [常见故障诊断与排障 (Troubleshooting)](#五常见故障诊断与排障-troubleshooting)

---

## 一、训练全流程流水线 (The Golden Pipeline)

整个 AI 训练分为四个标准环节：**纯 Rust 高速专家数据生成 $\to$ 绝对/相对分多任务模仿预训练 $\to$ 考官门禁体检 (Gatekeeper Arena) $\to$ 纯 Policy 向量化极速自对弈与自适应 KL 强化迭代**。

```text
[纯 Rust 引擎高速生成]
       │ cargo run --release --bin gen_imitation (~70 局/秒)
       ▼ 产出紧凑分片数据 data/imitation_shards/*.bin
[百万级专家数据集] ── 124 万步样本仅 2.8 GB，瞬间流式反序列化
       │
       ▼ bootstrap_imitation / train_one_epoch (多任务预训练)
[多任务底座模型] ── 拟合 Policy + 相对 Value + 绝对 abs_vp + 经济
       │
       ▼ evaluate_vs_heuristic_teachers (考官门禁体检)
[防退化门禁考官] ── 0 号位对抗 3 个 120 分 Rust 老师 (通过线: 胜率≥35% & 均分≥120)
       │ 通过后晋升并锁定为 Anchor Base
       ▼ VectorizedSelfPlay (纯 Policy 向量化自对弈)
[长期自对弈强化] ── 0.05s/局，单机百万局吞吐
       ├── 自适应 KL 散度约束 (D_KL(Anchor || Current)，防策略崩溃)
       ├── 绝对高分奖励驱动 (突破 130~140+，杜绝低分内卷纳什陷阱)
       └── 阶梯式考官门禁晋升与 Anchor 滚动升级
```

### 步骤 0：环境准备与编译
在执行任何训练之前，必须确保使用虚拟环境，并且 Rust 引擎最新修改已编译进 Python 扩展：

```bash
# 1. 激活虚拟环境 (Windows Git Bash)
source .venv/Scripts/activate

# 2. 编译并安装 Rust 核心扩展
python -m maturin develop --release --features python

# 3. 运行 Python 侧回归测试，检查失败与跳过原因
python -m pytest python/tests -q
```

### 步骤 1：训练高质量专家模仿底座 (Bootstrap Imitation)

我们推荐**解耦生成与训练**：先用纯 Rust 高性能生成器产出样本，再直接挂载训练。**整个过程完全不依赖 MCTS，纯启发式专家生成 + 纯监督深度学习**。

#### 方案 A（推荐首选）：纯 Rust 极速生成样本分片 + 挂载训练

**子步骤 1.1：纯 Rust 引擎高速生成二进制样本分片（~70 局/秒，零 MCTS 开销）**
```bash
# 生成 10,000 局高质量专家对局（开启 min-vp 30 质量硬门禁，产出约 124 万步样本）
cargo run --release --bin gen_imitation -- \
  --games 10000 \
  --out-dir data/imitation_shards \
  --min-vp 30
```
- `gen_imitation` 核心参数说明：
  - `--games / -n`：目标录取的合格对局数（默认 1000）。
  - `--out-dir / -o`：二进制分片输出目录（默认 `data/imitation_shards`）。
  - `--min-vp`：单人最低分硬门禁（推荐 30），任何玩家终局分低于此门槛整局作废。
  - `--min-avg-vp`：全桌平均分门禁（可选，例如 100）。
  - `--shard-size`：每个 `.bin` 分片容纳的动作步数（默认 32768 步，约 70MB/片）。

**子步骤 1.2：Python 启动神经网络预训练（流式加载 .bin 分片）**
```bash
python python/bootstrap_imitation.py \
  --sample-dir data/imitation_shards \
  --ckpt checkpoints/v1/bootstrap/b10k.pt \
  --epochs 3 \
  --batch 256 \
  --materialize-workers 8
```

#### 方案 B：在线生成并训练（无需提前生成分片）
若需在线小批量快速实验，可通过 Python 直接驱动生成与训练（开启 `--min-vp 30` 质量门禁，杜绝破产局）：

```bash
python python/bootstrap_imitation.py \
  --ckpt checkpoints/v1/bootstrap/b2000.pt \
  --games 2000 \
  --epochs 6 \
  --batch 256 \
  --workers 8 \
  --materialize-workers 8 \
  --min-vp 30 \
  --eval-games 20 \
  --eval-sims 64
```
- **核心参数**：
  - `--sample-dir`：指定已存在的二进制分片目录（自动识别 `.bin` 或 `.pkl`）。
  - `--ckpt`：输出的模型保存路径。
  - `--epochs 3`：跑 3 个全量 epoch（124 万步数据跑 3 遍），充分拟合 Policy、相对 Value 以及新增的绝对分 `abs_vp`。
  - `--batch 256`：GPU 批大小（可根据显存调整为 128~512）。
  - `--materialize-workers 8`：并行快照物化工作进程数。
  - `--resume`：支持中断后接着上一 epoch 续训。

| 指标 | 到底在预测什么？ | 数学计算方式 | 盲猜 baseline | 目标优秀值 |
| --- | --- | --- | --- | --- |
| policy | 眼前这一步选哪个动作（走法模仿） | 动作分布的交叉熵 (CE) | ~3.5 ~ 4.0 | 2.2 ~ 2.6 |
| value | 4 个座位的终局相对净胜分 | 归一化得分的均方误差 (MSE) | > 0.20 | < 0.08 |
| abs_vp | 4 个座位的终局绝对得分 $(VP-100)/50$ | 线性输出 + Smooth L1 损失 (beta=0.2) | > 0.15 | < 0.03 |
| winner | 4 个人谁最终夺冠（第一名） | 4 分类的交叉熵 (CE) | 1.386 | 0.5 ~ 0.8 |

### 步骤 2：考官门禁体检与 Anchor 锁定 (Gatekeeper Arena)

训练得到底座后，在开启长程强化自对弈之前，使用现成的独立评测脚本验证模型是否已经拥有扎实的基本功（零 MCTS 搜索下能否抗衡 120 分 Rust 老师）：

```bash
# 运行 40 局实测对抗（轮换 4 个座位对抗 3 个 120 分 Rust 启发式老师）
python python/gatekeeper_eval.py --ckpt checkpoints/v1/bootstrap/b10k.pt --games 40 --verbose
```
- **放行门槛**：胜率 $\ge 35\%$ 且学生均分 $\ge 120$ 分。通过后正式将该 Checkpoint 复制锁定为初代 **`Anchor Base`**。

#### 补充体检：微观价值与动作排序探针 (Q/V Ranking Probe)
在开启消耗算力的自对弈前，还可使用独立探针检测模型对微观动作与局面的敏感度：

```bash
python python/bench_value_ranking.py \
  --ckpt checkpoints/v1/bootstrap/b10k.pt \
  --positions 36
```
- **解释方式**：该探针以启发式续局作为参考，需要结合局面数、误差范围及具体候选分析，不能用一次相关系数大于 +0.08 作为放行证明。Q/V 排序弱时，自博弈不一定能自行纠正；开始长训练前还需独立对局比较策略直出与搜索，并记录经济崩溃轨迹。当前已知失败及修复证据见 [selfplay-validation.md](selfplay-validation.md)。

### 步骤 3：自对弈强化学习迭代 (Self-Play RL)
以干净的底座为起点，模型通过自我博弈并持续升级。系统支持基于 MCTS 树搜索的稳健自对弈，以及纯 Policy 向量化极速自对弈：

#### 方式一：MCTS 树搜索自对弈（生产基线）
以深度启发式树搜索指导策略与价值学习，产生更深刻的大局观和长线规划能力：

```bash
python python/selfplay_train.py \
  --ckpt-dir checkpoints/v1/runs/sp01 \
  --init-from checkpoints/v1/bootstrap/b2000.pt \
  --iterations 50 \
  --games-per-iter 16 \
  --sims 128 \
  --workers 8 \
  --c-puct 0.25 \
  --prior-top-k 32 \
  --heuristic-opponent-prob 0.25 \
  --temperature 0.0 \
  --eval-every 2 \
  --eval-games 12 \
  --heuristic-eval-games 12 \
  --promote-winrate 0.35
```

#### 方式二：纯 Policy 向量化极速自对弈（零 MCTS，极速吞吐）
完全剥离 MCTS 树搜索开销，基于多环境（Batched Envs）锁步并发推演，直接通过神经网络 Policy Logits 带温度采样快速生成对局轨迹，并由 Rust 老师门禁考官进行质量把关：

```bash
python python/fast_selfplay_train.py \
  --ckpt-dir checkpoints/v1/runs/fast_sp01 \
  --init-from checkpoints/v1/bootstrap/b2000.pt \
  --iterations 30 \
  --games-per-iter 64 \
  --env-count 16 \
  --temperature 0.8 \
  --heuristic-prob 0.25 \
  --kl-lambda 0.05 \
  --min-vp-filter 30.0 \
  --batch 256 \
  --lr 1e-4 \
  --eval-every 5 \
  --eval-games 20
```

- **方式二核心参数**：
  - `--env-count 16`：并发运行的 Rust 游戏环境数，在 GPU 上单步集中打包前向推理，吞吐可达 10~30 局/秒。
  - `--temperature 0.8`：走步采样温度（越接近 1.0 探索越广，越小越贪婪，推荐 0.6 ~ 0.8）。
  - `--heuristic-prob 0.25`：**启发式混战比例**：为对手席位以 25% 概率混入 120 分 Rust 老师执子，打破自博弈共谋盲区。
  - `--kl-lambda 0.05`：**KL 散度正则约束权重**：约束策略相对 Champion 锚点分布的偏离度 $D_{\text{KL}}(\pi_{\text{anchor}} \parallel \pi_{\theta})$，考官晋升新模型时自动滚动锚点，防范自对弈策略崩溃。
  - `--min-vp-filter 30.0`：**劣质局过滤硬门槛**：全盘任一玩家最终分低于 30 分的崩溃局直接弃用，确保梯度健康。
  - 内部自动启用 **Advantage 优势加权**：依据终局相对胜负净分动态赋予动作置信度，胜者额外强化 1.5x，避免无脑模仿臭棋。
  - `--eval-every 5`：每隔 5 轮调用 `evaluate_vs_heuristic_teachers`，自动与 3 位 Rust 启发式老师切磋，考官均分提升时自动沉淀 `best.pt`。

#### 两种自对弈路径对比与选型建议：
| 对比维度 | 方式一：MCTS 树搜索自对弈 (`selfplay_train.py`) | 方式二：纯 Policy 向量化极速自对弈 (`fast_selfplay_train.py`) |
| :--- | :--- | :--- |
| **MCTS 依赖** | **依赖**（默认 128 次推演/步） | **完全不依赖（Zero MCTS）** |
| **生成速度** | 较慢（约 0.5 ~ 2 局/秒） | **极快（约 10 ~ 30 局/秒）** |
| **显存/CPU要求** | CPU 密集（多进程并行 MCTS 树） | GPU 集中打包推理（显存利用率高） |
| **策略特点** | 局部战术计算深，防守与借贷时机更准 | 适合海量探索，快速验证网络结构与超参数 |
| **适用阶段** | 冲刺高分上限、参加正式竞技锦标赛 | 快速原型验证、轻量机器日常训练、大规模策略迭代 |

#### 关键参数配置指南：
| 参数 | 推荐值 | 物理含义与调优理由 |
| :--- | :--- | :--- |
| `--init-from` | `checkpoints/v1/bootstrap/b2000.pt` | 干净的高质量预训练底座，跳过前期盲目随机探索 |
| `--sims` | `128` | 每步 MCTS 模拟推演次数（兼顾搜索质量与生成吞吐量） |
| `--c-puct` | `0.25` | 探索常数，与 $VP\_SCALE=50$ 深度对齐 |
| `--prior-top-k` | `32` | **分层保底 Top-K**：确保 6 大基础动作类型（Build/Network/Develop/Sell/Loan/Pass）各保底保留前 3~4 个最优候选，彻底避免建厂动作挤占卖货/借贷名额 |
| `--heuristic-opponent-prob` | `0.25` | 每个非保留座位有 25% 概率使用启发式教师。用于增加对手多样性；混合局的经济健康和棋力仍需评测，教师计算也有开销。 |
| `--temperature` | `0.0` | 确定性走步，依赖根节点 Dirichlet 噪声探索，避免开局自毁走法 |
| `--eval-every` | `2` | 每 2 轮进行一次正规对决评估，保持高频监控 |
| `--eval-games` | `12` | 挑战历史最佳 `best.pt` 的竞技局数（3 组严格轮换座位） |
| `--heuristic-eval-games`| `12` | 对抗启发式规则裁判的局数（3 组严格轮换座位） |
| `--promote-winrate` | `0.35` | **4人局晋升胜率门槛**：在 1 vs 3 模式下基准期望仅为 25%，0.35 要求胜率明显超越基准，并结合 Wilson 下限置信度判定刷新 `best.pt` |

### 步骤 4：断点续训 (Resume)
按 `Ctrl+C` 会中断当前工作，已完成轮次的 `latest.pt` 保留；未完成轮次和内存 replay buffer 不会恢复。之后可从已保存轮次续训：

```bash
python python/selfplay_train.py \
  --ckpt-dir checkpoints/v1/runs/sp01 \
  --resume \
  --iterations 100 \
  --games-per-iter 16 \
  --sims 128 \
  --workers 8 \
  --c-puct 0.25 \
  --prior-top-k 32 \
  --heuristic-opponent-prob 0.25 \
  --temperature 0.0 \
  --eval-every 2 \
  --eval-games 12 \
  --heuristic-eval-games 12
```

---

## 二、模型版本规范与资产分层 (SemVer & Storage)

### 1. 语义化版本（X.Y.Z）在博弈 AI 中的映射法则

为避免文件名无限拼接（如 `v6-b2000-sp4-it50-league2-xxx.pt`），采用**物理隔离 + 扁平短名 + 内部元数据**管理：

| 版本维度 | 含义与范围 | 兼容性特征 | 物理落位 |
| :--- | :--- | :--- | :--- |
| **X (主版本)** | **物理 Schema 变更**<br>(状态 Token / 动作特征维度变动) | **完全不兼容**。不同 X 之间的模型无法 forward，不能同台对弈。 | 顶层根目录物理隔离（如 `checkpoints/v1/`、`checkpoints/v2/`）。 |
| **Y (次版本)** | **策略环境 / 数据流演变**<br>(如修复运河期破产 Bug，策略风格跃迁) | **逻辑兼容，网络结构一致**。可以同桌对战，权重可迁移微调。 | 区分底座代号与实验流水线（如 `bootstrap/b2000.pt`、`runs/sp01/`）。 |
| **Z (修订版)** | **训练步数 / 检查点**<br>(如第 20 轮、50 轮迭代产物) | **同一流水线内部连续演进**。 | 目录内标准文件名（`latest.pt`、`best.pt`、`it020.pt`）。 |

### 2. 标准目录结构

```text
checkpoints/
├── v1/                                 # 【X 级：对应当前引擎 STATE_TOKEN_SCHEMA_VERSION=1 & ACTION_SCHEMA_VERSION=1】
│   ├── bootstrap/                      # 干净的起步底座（长期保留）
│   │   └── b2000.pt                    # 官方标准起手底座 (2000局模仿训练)
│   │
│   ├── runs/                           # 活跃的实验工场（一个实验一个目录）
│   │   └── sp01/                       # 第一期自对弈流水线
│   │       ├── latest.pt               # 实时续训断点（含完整优化器与调度器状态）
│   │       ├── best.pt                 # 该 run 内部历史最佳模型
│   │       ├── latest.json             # 迭代进度元数据
│   │       └── metrics.jsonl           # 逐轮完整指标明细
│   │
│   └── zoo/                            # 【名人堂 / 对抗池 (Hall of Fame)】
│       ├── manifest.json               # 选手花名册与各项历史天梯指标
│       ├── heuristic                   # 规则引擎裁判标杆
│       ├── v1_bootstrap_b2000.pt       # 纯模仿底座代表
│       └── (晋升入驻的优秀模型...)
│
└── archive/                            # 【历史封存库】
    └── legacy-pre-v1/                  # 历史探索期废弃模型与破产数据
```

### 3. 一键检查模型身世与兼容性 (Checkpoint Inspector)

随时使用内置探针检查任意 checkpoint 的参数量、训练 Epoch 以及与当前引擎的兼容性：

```bash
python python/inspect_ckpt.py checkpoints/v1/bootstrap/b2000.pt
```

输出示例：
```text
================ Checkpoint Inspector ================
Path:      checkpoints/v1/bootstrap/b2000.pt
File Size: 26.94 MB
Type:      Full Trainer State (Resumeable)
Weights:   99 tensors, 2,343,890 parameters (~8.94 MB)
Epoch/It:  6
Schema:    Action Schema v1 (dim=55), State Token Schema v1
Contents:  optimizer=True, scheduler=True, scaler=True
Compatibility: [OK] Matches current engine schemas (Action v1, State v1)
======================================================
```

### 4. 磁盘瘦身与清理规则
- **`.imitation/*.pkl` 样本分片**：生成完底座模型（`.pt`）后，所有的 `.pkl` 缓存分片属于中间产物，**可随时全量删除**，不影响模型加载与推理。
- **`latest.pt` vs `best.pt`**：`latest.pt` 保存了 Adam 优化器动量与学习率调度器（体积较大），用于 `--resume`；实验结束后只需将精简权重 `best.pt` 移入 `zoo/`，实验工作区可根据需要清理或压缩。

---

## 三、核心指标速查字典 (Metric Dictionary)

记忆口诀：**Policy 学招法，Value 算得失，Winner 猜谁赢，Q-rank 辨微调**。

### 1. 神经网络三头损失（Losses）

| 指标字段 | 预测目标 | 数学原理 | 盲猜 Baseline | 优秀标准 | 当前物理含义解读 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`policy`** | 当前局面选哪个合法动作 | 动作多分类**交叉熵 (CE)** | ~3.5 ~ 4.0 | **2.2 ~ 2.6** | 衡量走子模仿精度。越低说明越聚焦于专家/搜索推荐的高价值动作。 |
| **`value`** | 4 个座位的终局相对净胜分 | 归一化净分的**均方误差 (MSE)** | > 0.20 | **< 0.08** | 预测局面的胜负差距。若 MSE=0.063，换算成真实 VP 误差仅约 $\sqrt{0.063}\times 50 \approx 12.5$ 分。 |
| **`abs_vp`** | 4 个座位的终局绝对得分 $(VP-100)/50$ | 线性输出 + **Smooth L1 损失** (beta=0.2) | > 0.15 | **< 0.03** | **打破低分内卷的核心驱动**。预测绝对得分能力，无 Sigmoid 截断，天然兼容 150~200+ 顺风高分局。 |
| **`winner`** | 谁是最终第一名夺冠者 | 4 选 1 **交叉熵 (CE)** | **1.386** ($-\ln 0.25$) | **0.5 ~ 0.8** | 宏观大局观。德式桌游前期变数极大，初期此值往往贴近 1.386，随自对弈深入逐渐降至 0.8 以下。 |
| **`q`** | 动作边预测价值 $Q(s, a)$ | 被走动作的回归误差 | - | **< 0.03** | 配合树搜索的初始化（`q_init`），引导 MCTS 优先展开高价值分支。 |

### 2. 实战对弈与健康体征（Game Stats）

在自对弈日志 `metrics.jsonl` 中，必须密切监控以下字段：

- **`min_vp` 与 `zero_vp_players`**：
  - 结合完整局数、零分座位数、现金/收入回放诊断，不能规定所有正常对局都必须大于 50 分。
  - 已取消“停滞两次即永久破产并清零”。零分仍可能来自真实欠款扣分或差策略，需要追踪具体行动。
  - `--min-vp-filter` 默认 0；过滤不是修复，失败轨迹可提供价值监督。被过滤局的 VP 仍如实计入日志。
- **`avg_vp` 与 `winner_avg_vp`**：
  - 都是所有角色混合后的统计，不是固定对手下当前模型的棋力。应同时看 `collected_avg_vp` 和固定对手评测。
  - 不同 seed、座位、对手组成可引起波动；不要根据一两轮下降断言退化。
- **`heuristic_winrate`**：
  - 对抗 Rust 硬编码启发式裁判的胜率。
  - 同时记录 `heuristic_avg_vp`、`heuristic_zero_games` 与 `champion_heuristic_avg_vp`；没有保证随轮数增长的胜率曲线。
- **`arena_winrate` 与 `PROMOTED`**：
  - 新模型挑战当前历史最佳 `best.pt` 的胜率。
  - 默认要求胜率 ≥ 0.35 且 Wilson 单侧下限 ≥ 0.25，并通过教师健康检查，才触发 `PROMOTED`。未评估轮次显示 `--`。

### 3. 探针排序指标（bench_value_ranking）

- **`V(s) Spearman`**：走完一步后对后续状态估值的排序相关性。必须为**正数**（通常在 $+0.10 \sim +0.30$）。
- **`Q(s,a) Spearman`**：在当前状态直接对候选好动作进行排序的能力。底座阶段约在 $0$ 附近，自对弈经 MCTS 磨炼后应跃升至 **$> +0.25$**。

---

## 四、多模型对战体系 (League & Tournament)

《伯明翰》是 4 人不对称经济博弈游戏，单模型自我对弈容易产生策略盲区。建立多智能体天梯体系是通往超人级 AI 的必由之路。

### 1. 对手池机制（Self-Play Matchmaking）
在 `selfplay_train.py` 中，已原生集成动态对手池：
- `--pool-size 6`：保留最近 6 个已晋升模型（启动时包含底座）。
- `--mm-prob 0.25`：每个非指定座位有 25% 概率使用历史模型；`--heuristic-opponent-prob` 同样按座位抽签。混合局保留所有当前 champion 座位的样本。

### 2. 跨版本 4 人锦标赛（Tournament 准则）
组织不同代际选手（如 `v1_sp01_best` vs `v1_bootstrap_b2000` vs `heuristic`）进行正规对抗时，必须遵循：
- **严格座位轮换**：每 4 局为一个标准对抗单元，4 位选手依次轮换 1、2、3、4 号位，完全抵消出牌顺位带来的固有优势。
- **统一确定性搜索**：正式对抗中强制关闭采样温度（`temperature=0.0`），采用固定模拟次数（如 `sims=128`），确保比拼的是纯粹的模型策略质量。

---

## 五、常见故障诊断与排障 (Troubleshooting)

### Q1: 训练过程中频繁出现 `min_vp: 0.0`，均分不足 80 分？
- 先确认扩展已重建，再查看零分比例及回放中的借贷、现金、收入、卖货、欠款扣分路径。不能只凭 min=0 判断底座或训练样本污染。
- 检查 `completed_games`、`filtered_games` 与采样角色；过滤局也会出现在均分中。
- 比较底座直接 policy 与 MCTS 的固定对局，再判断是搜索还是模型问题。历史诊断见 [2026-09-12 诊断](training-diagnosis-2026-09-12.md)。

### Q2: 自对弈中 Policy loss 或 Value loss 震荡不降？
- **排查**：
  - 探索常数 `--c-puct` 默认 `0.25`，需要与候选宽度及价值质量共同做对照；不存在所有模型都必须使用同一个常数的保证。
  - 检查采样温度 `--temperature`：推荐设为 `0.0`。如果前期设置了长时间的高温随机采样，会导致高质量局面被噪声破坏。

### Q3: 运行自对弈时报内存不足 (OOM) 或速度极慢？
- **排查**：
  - `--workers`：建议设为 `CPU核心数 - 2`（通常 6~8 个 worker）。
  - `--max-candidate-batch`：若 GPU 显存紧张，可从默认的 `65536` 下调至 `32768`。
  - `--prior-top-k`：CLI 默认 `32`。设为 0 会保留所有合法动作，通常增加搜索开销。
