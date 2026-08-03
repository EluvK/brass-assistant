# 系统架构设计

> 总体架构与阶段规划详见 `startup.md`；本文档描述目标架构蓝图与模块边界。

## 总体架构

```
+-------------------------------------------------------------------+
|                        1. Tabletop Simulator                      |
|  [TTS 客户端] --(隐蔽挂载小棋子 Lua 脚本)--+                      |
+--------------------------------------------|----------------------+
                                             | HTTP / WebRequest (JSON)
                                             v (localhost)
+-------------------------------------------------------------------+
|                       2. AI Service (Python)                      |
|  +---------------------------+     +---------------------------+  |
|  |  FastAPI / gRPC Web Server |     | PySide6 桌面透明悬浮窗    |  |
|  +-------------+-------------+     +-------------+-------------+  |
|                |                                 ^                |
|                v                                 | (Top-3 推荐操作)|
|  +-----------------------------------------------+-------------+  |
|  | ISMCTS 搜索决策器 + PyTorch (Policy-Value Network)          |  |
|  +-----------------------------+-------------------------------+  |
+--------------------------------|----------------------------------+
                                 | Pybind11 / CFFI 接口调用
                                 v
+-------------------------------------------------------------------+
|                   3. Game Engine (Rust)                           |
|  * 极速游戏状态 (State) 更新                                      |
|  * 合法动作生成器 (Action Generator)                              |
|  * 图论连通性分析 (Graph Connectivity & Resource Flow)            |
+-------------------------------------------------------------------+
```

## 模块职责

### 1. Game Engine (Rust) — 阶段 1
- `state`：纯数据、可序列化的游戏状态
- `action_gen`：合法动作生成器（动作空间可能上百）
- `graph`：城市网络连通性（**静态邻接表 + u32 位掩码 BFS，零堆分配**）
- `rules`：7 种行动的资源结算与合法性校验
- `heuristic_ai`：启发式 AI（Baseline，可参考 npow `aiPlayer.js` 的评估思路，但不直接照抄）
- `search_ai`：2-ply 确定性前瞻（同回合两动 Combo，`score(first) + α·score(best second)`）
- `mcts_ai`：ISMCTS + Determinization（隐藏牌池采样对手手牌，PUCT，1-ply Prior/叶评估）
- 数据录入：可初始对照 `reference/npow-brass-birmingham/js/gameData.js`，但需由本仓库实现与测试锁定

### 2. TTS 隐蔽数据抽取 (Lua) — 阶段 2
- 个人 Saved Objects 小物件脚本，快捷键触发
- 读取公共盘面 + 己方手牌 → JSON → `localhost` HTTP POST
- 参考：`reference/ikegami-tts-brass/objs/` 的 API 用法

### 3. AI Service (Python) — 阶段 3
- Pybind11 / CFFI 桥接 Rust 引擎
- 状态向量化（Feature Encoding）
- Policy-Value 双头网络 + ISMCTS
- FastAPI 接收盘面 JSON，返回 Top-3 推荐

### 4. Self-Play 训练 — 阶段 4
- 自我对弈数据闭环，3060 Ti FP16 训练
- 目标：单次 10,000 次 MCTS 模拟 ≤ 10 秒

### 5. UI 悬浮窗 (PySide6) — 阶段 5
- 无边框透明置顶，实时展示推荐操作

## 关键决策约束

1. **性能热路径**（simulation / self-play）一律走 Rust，Python 只做胶水
2. **状态必须可序列化**：便于确定化采样（Determinization）、存档、调试
3. **数据单一事实来源**：最终以本仓库已验证的数据/实现为准；reference 只用于初始化录入与排查，不作为绝对真源

## 一致性校验

- 对高风险规则写最小复现与测试，避免把 reference 的 bug 当成真规则
- npow / ikegami 可作为差异比对样本，但冲突时优先检查规则书结论与用户裁决

---

## 阶段 3 先行工作：PyO3 绑定与状态编码（A–D）

### 模块划分（`src/engine/src/`）

| 模块 | 职责 |
| --- | --- |
| `pymod.rs` | PyO3 绑定：`brass_engine` 模块 + `GameState` pyclass（包装 `GameState<StdRng>`） |
| `move_codec.rs` | `Move` ⇄ 规范字符串（lossless，含资源源/卡牌选择），训练数据可序列化 |
| `policy.rs` | 固定动作表（Policy 头输出空间）+ `move_slots` + `legal_mask` |
| `encode.rs` | `state_to_tensor`：板面/网络/全局/手牌特征编码 |

### Python API（`import brass_engine`）

```
GameState(seed, players)        # 2-4 人
  .current_player_id / .player_count / .era(0运河1铁路) / .round / .game_over
  .current_player_money / .has_pending_bonus        # 只读属性
  .legal_moves()  -> list[(policy_slot:int, canonical:str, describe:str)]
  .apply_move(canonical) -> str   # 完整 step：动作 + 回合推进 + 时代结算
  .determinize() -> GameState     # 对手手牌采样（ISMCTS）
  .legal_mask() -> list[int]      # 合法策略槽位集合
  .state_to_tensor() -> (board, links, global, own_hand, opp_hands)
  .choose_heuristic() / .choose_2ply() -> (canonical, describe, score)
brass_engine.describe_slot(slot) / moves_to_slots(canonical)
brass_engine.policy_table_size / network_double_cells  # 常量
```

构建：`maturin develop`（venv 见 `src/engine/.venv`），`cargo test` 与
`.venv/Scripts/python.exe -m pytest tests/test_engine.py` 双套测试。

### 固定动作表（`policy.rs`，共 715 槽）

```
Build(城市) 20×4×6=480 | Build(农场) 2 | Network 39
NetworkDouble 静态共享端点铁路对(102) | Develop 6+36 | Sell 47 | Loan/Scout/Pass 3
```

`move_slots` 用动作关键参数做 canonical 槽位（资源源/卡牌选择折叠，与 MCTS 树
匹配同构）；`legal_mask` 由合法动作去重得到，网络 softmax 只作用其上。偏移量随
NetworkDouble 对数动态计算（`develop_offset()` 等函数）。

### 特征张量（`encode.rs`，共 ~1257 维）

- **board `(17, 49)`**：47 城市槽 + 2 农场；occupied/owner4/industry6/flipped/cubes/level/vp/income/is_farm
- **links `(6, 39)`**：built/is_canal/owner4
- **global `(50,)`**：时代、轮次、行动数、每玩家 money/income/vp/links/wild、煤铁市场、顺位、待结算奖励
- **own_hand `(35,)`** 与 **opp_hands `(3×35,)`**：bag-of-cards（27 城 + 6 产业 + 2 wild 计数），对手手牌取 determinized 状态内的具体手牌

网络骨架：cell-encoder（共享 MLP）→ concat(global) → trunk → 双头（policy 715 + value 标量）。

### 已修复的引擎 bug（Python 全局长棋暴露）

- `rules.rs` 双铁路枚举在「煤仅经首铁路可达」时对回滚状态重算 c2 煤源 → 空数组
  `coal2[0]` 越界。修复：`SecondRailOption` 携带在首铁路就位状态下枚举的
  `coal2_opts`（含预算过滤），`legal_moves` 与 heuristic 直接复用，不再重算。

### 已知遗留（规则裁决挂起）

- **双铁路是否要求共享端点**：引擎与 npow 参考实现一致允许「两条铁路只需都在
  玩家网络内、不要求共享端点」。因此 policy 表把双铁路域扩为**全部无序铁路对**
  （38 选 2 = 703，表总量 715 → **1316**）。若物理规则书要求共享端点，需用户裁决
  后再收紧引擎 `get_second_rail_options`（表保持超集、无害）。
- **双铁路煤源枚举不一致**：`get_second_rail_options` 按「内部最便宜 coal1 已消耗」
  的状态算 `coal2_opts`，而 canonical 实际用的 outer coal1 可能不同 → `legal_moves`
  会产出少量执行必失败的 NetDouble。Rust MCTS 靠捕获 apply 错误跳过；Python MCTS
  对每个 slot 保留全部 canonical 逐个尝试（`mcts.py _make_child`）。待后续修引擎。

---

## 阶段 E：AI 层（Python MCTS + Policy-Value 网络，`src/ai/`）

### 模块

| 模块 | 职责 |
| --- | --- |
| `build_input.py` | `state_to_tensor` → torch 张量（支持批量 + 任意玩家视角） |
| `net.py` | Policy-Value 双头网络：cell-encoder(共享 MLP) → concat(global/hands) → trunk → policy(1316) + value(tanh) |
| `mcts.py` | ISMCTS：单次根确定性 + PUCT + MaxN 价值向量 + 网络先验；slot 为树标识 |
| `selfplay.py` | 自对弈：每步记录 (state, visit 分布 policy 目标, 归一化最终 VP 价值目标) |
| `train.py` | AlphaZero 训练循环（policy CE + value MSE + L2，AMP 预留） |
| `evaluate.py` | MCTS vs heuristic/2ply 席位轮换对局评估 |

### 关键设计

- **价值目标**（用户已裁决）：`z_p = (vp_p - mean) / max(std, eps)`，样本携带视角玩家的 z。
- **MaxN**：叶子对 4 个玩家视角批量编码一次前向 → 价值向量；选择时该节点行动者
  最大化自己的 Q+PUCT（非零和，对手不会结盟）。
- **树标识 = policy slot**：资源源/卡牌选择折叠；slot 的 canonical 逐个尝试直到可执行。
- **确定化**：每次 `search` 对根状态 `determinize()` 一次（单世界树）。Rust `mcts_ai`
  每模拟重采样；Python 版为简化起见先用单世界，训练闭环验证后再对齐。
- 运行：`PYTHONPATH=src/ai .venv/Scripts/python.exe -m pytest src/ai/tests -q`（CPU 版，
  9 项测试）。CUDA 训练：把 `TrainConfig.device` 设为 `"cuda"` 并安装 cu12x torch。

## 阶段 F 初步：CUDA 训练验证（已完成验证）

### 换 CUDA 步骤

```bash
# 安装 CUDA 版 torch（官方源，清华镜像只有 CPU 版；~2.5GB）
# 3060 Ti 驱动 595.97 支持 CUDA 13.2，用 cu126 wheel（有匹配 2.13.0+cu126）
pip install --force-reinstall "torch==2.13.0+cu126" --index-url https://download.pytorch.org/whl/cu126
python -c "import torch; print(torch.cuda.is_available())"   # -> True
```

代码无需改动：`TrainConfig.device` 默认 `cuda if available`；AMP 自动走 FP16
autocast；`ISMCTS(device=...)` 把网络与 batch 搬到 GPU。`mcts.py` 已支持 device
（`_encode_perspectives` 返回的 batch `.to(device)`）。

### 性能与发现（3060 Ti）

- 网络仅 ~40 万参数，GPU 前向微秒级；**瓶颈在 Python 状态操作**（legal_moves /
  clone / apply_move 的 Rust 跨边界调用 + 张量组装），单决策 ~7ms/sim，GPU 对小
  网络无明显提速。大网络 / 批量推理（阶段 H 搬回 Rust）才有量级收益。
- 纯随机自对弈短训（5 轮 × 124 样本）**严重过拟合**：loss 降但 MCTS 退化到 0 VP。
  这是随机自对弈 + 小数据的预期现象，非流程问题。
- **启发式行为克隆预热**（`src/ai/bootstrap_imitation.py`）可快速验证学习：
  - 60 局（7517 样本）: MCTS vs 启发式 17.5/50.3 VP
  - 150 局（18796 样本）: MCTS vs 启发式 34.5/69.8，vs 2ply 61.2/77.8
  - 网络从 0 VP → 34.5 VP，逼近 2ply 基线，证明「数据→训练→MCTS→评估」闭环有效。

### 结论

流程已端到端验证 OK。正式训练（阶段 F 主体）方向：
1. 以 bootstrap 网络为起点，切回纯 AlphaZero 自对弈（比随机起点快得多）；
2. 提高自对弈吞吐（提高 sims / 多进程 / 阶段 H 搬回 Rust）以提供足够多样数据；
3. 持久化 optimizer + LR schedule，避免每迭代重建 Adam 的动量丢失。

### Step 0 去风险验证（已完成，2026-08）

`train.py` 重构为持久 `Trainer`（常驻 AdamW + CosineAnnealingLR，跨代保留状态，
`state_dict` 可存/载）。`src/ai/selfplay_from_bootstrap.py` 从 `bootstrap.pt` 起点，
低 LR(1e-4) 微调，自对弈 3 轮（sims=120，2 局/轮），每轮仅 vs 启发式快评（2 局）。

**结果：未塌缩。** MCTS vs 启发式 VP：48.5 → 27.0 → 35.5 → **59.0**（启发式 79.8→66.0），
逼近基线。对照随机起点的 5 轮塌缩到 0 VP，bootstrap 起点 + 持久优化器 + 低 LR
微调的组合方向正确。Trainer 状态存 `checkpoints/selfplay0_it*.pt`。

注意点（供 Step 1）：
- 单次评估仅 2 局噪声大（VP 波动 ±15）；正式评估需多局或滚动平均。
- 本次 `t_max=iters*epochs=9` 使 LR 快速退到 1e-5；正式训练应把 `t_max` 设为总轮数。
- 自对弈吞吐 ~3min/2 局（sims=120），多进程仍是关键杠杆。

### Step 1 发现与实验 V1（2026-08）

多进程 worker 池（`mp_selfplay.py`，spawn，8 workers → 4.7× 吞吐）、持久 Trainer、
评估门控回退均已实现。但诊断出两个核心事实：

1. **自对弈训练塌缩**：从 bootstrap 起点跑 8 轮（sims=200，16 局/轮）后 MCTS VP 塌缩到
   2.6（起点 ~40）。根因：弱 value 使自对弈目标嘈杂 → value 被污染 → MCTS Q 变噪声 →
   负反馈。`train_mp.py` 现支持评估门控（VP 退化自动回滚到最佳权重）作为防线。
2. **网络本质弱点**（诊断数据）：
   - 纯 policy 贪心 = ~7 VP（policy 头模仿不充分）
   - MCTS（同网络）= ~37 VP（**是 value 头在扛 Q 选择**）
   - 启发式 = 60-70 VP

**实验 V1（强化 value 头，`exp_value.py`）**：250 局启发式数据，两阶段训练 value——
- Phase A 只训 value_head（冻结 trunk）：MCTS 39.8 → **42-46 VP**（+3-6，真实但小）
- Phase B 解冻 trunk 训 value：**变差到 38.5**（共享 trunk 漂移污染冻结的 policy_head；
  val_mse 虽降到 0.49 但净效果负）
- 结论：value 头被轻微低估（可免费 +3-6 VP，`best_value.pt` 已存为改进基线），但**不是
  关键瓶颈**；解冻 trunk 训 value 不可取。

**性能优化**：子节点惰性物化（`mcts.py`，展开时不再一次性 clone 全部 ~68 个孩子，首次
下降才物化）→ 单模拟 5.6ms → **3.3ms（1.7×）**。全链路接入 `progress.py`（自对弈/评估/
训练均有 elapsed + ETA 输出）。

### 关键修复：masked policy loss（2026-08，重大突破）

**根因**：`compute_loss` 的 `log_softmax` 在全部 1316 槽上归一化，但目标 `pi` 只在合法槽
非零。703 个 double-rail 幽灵槽（绝大多数状态非法）仍进分母——初始随机权重下约 **53% 的
概率质量在幽灵槽上**，网络把大量梯度用于压制它们，与「分辨真实合法动作」争抢信号。
这解释了 policy 贪心仅 ~7 VP 的异常弱。

**修复**（`train.py compute_loss`）：用每样本的 `legal` 掩码做 masked log_softmax——
`logits.masked_fill(~mask, -inf)` 归一化只覆盖合法槽，再把非法槽 log_probs 清零避免
`0×-inf=NaN`。`Sample` 新增 `legal` 字段（`state.legal_mask()` 生成），经 `_to_batch` /
`mp_selfplay` 打包透传。

**效果**（`experiments/exp_masked_loss.py`，250 局模仿 + 15 epoch）：

| 指标 | 旧（unmasked） | 新（masked） |
| --- | --- | --- |
| Greedy policy VP（10 局） | 7.2 | **50.2** |
| MCTS vs heuristic（8 局） | 34.5 / 69.8 | **72.6 / 65.2，胜率 62%** |
| MCTS vs 2ply（8 局） | ~61 / 78 | **70.8 / 75.8** |

结论：masked-loss 是弱 policy 的元凶，修复后 MCTS 反超启发式、逼近 2ply。当前最佳基线
为 `checkpoints/best_masked.pt`。这同时回答了「1316 维动作空间会不会有问题」——问题不在
维度本身，而在 loss 未掩码；掩码后幽灵槽零梯度，维度开销可忽略。

### 新 Schema 重设计（2026-08，分叉 policy + 4 玩家 value）

**动机**：可靠基准（20 固定 seeds）显示旧 best_masked 胜率卡 50%，因每 20 局 3-5 局**灾难对局**
（MCTS 得 20-48 VP）。诊断（`experiments/diagnose.py`）确认灾难 = value 头崩溃 → 晚期动作退化
（seed 14 铁路时代 16 连 Pass、seed 11 建 10 不卖、seed 3 铁路 0 Build）。自对弈训练两次漂移退化
（val_mse 0.5→0.74），确认 value 头是瓶颈。

**新网络**（`brass_ai/net.py`，2026-08）：
- 分叉 policy：`type_head Linear(256→7)` + `goal_head Linear(256→1316)`，`logit(s)=type[t(s)]+goal[s]`
- value 头 `Linear(256→4)` 预测 4 玩家终局 z，去 tanh（encode global 已含每玩家统计）
- Rust `nn_mcts::flush_net` 每 request 只发 1 行（当前玩家视角），Rust 合并分叉先验，MaxN 直接用 4 向量

**效果**（`checkpoints/new_best.pt`，2000 局启发式 BC，20 固定 seeds）：
| 指标 (sims=1000) | new_best（新架构） | best_masked（旧架构） |
| --- | --- | --- |
| 胜率 | **0.65** | 0.50 |
| mean / median | **93.3 / 96.5** | 77.8 / 80.5 |
| min | **57（零灾难）** | 20 |

**训练门控**（`train_mp.py`）：放弃噪声 rolling-VP 门控，改 `benchmark_mcts_vs_heuristic`
（固定 seeds 胜率 + median VP）作接受标准；配合 matchmaking（`--mm_prob`，对手座位用历史
checkpoint 池）防漂移。短训冒烟验证通过。






