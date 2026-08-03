# AGENTS.md — Brass Assistant 项目背景知识

> 本文件是给后续进入本仓库工作的 **AI agent / 协作者** 使用的背景知识文档。
> 在开始任何任务之前，请先阅读本文件，了解项目全貌、游戏规则与参考材料位置。

---

## 1. 项目定位

**Brass Assistant（项目代号：Brass-Mind）** 是一个为桌游《工业革命：伯明翰》（**Brass: Birmingham**）构建的 **AI 策略分析与实时推荐辅助系统**。

核心愿景：
- 在 Tabletop Simulator (TTS) 真人对战时，隐蔽地读取公共盘面 + 己方手牌
- 由本地 AI 服务在 10~15 秒内完成数万次模拟推演
- 通过置顶悬浮窗给出 Top-3 操作建议及预估收益

长期目标：从零构建基于 **ISMCTS + Policy-Value 神经网络** 的《伯明翰》决策引擎。

---

## 2. 开发者背景（重要约束）

- 具备 **后端工程能力**（工程化思维、数据结构、架构设计、性能优化）
- **零深度学习背景**，学习路径需要循序渐进
- 硬件：RTX 3060 Ti (8GB 显存)，FP16 混合精度训练
- 工作语言：中文沟通，代码与文档中英文均可

---

## 3. 游戏规则速览

完整规则参考见 `docs/rules/brass-birmingham-rules.md`。此处为 agent 需要理解的核心要点：

### 3.1 基本流程
- 2~4 人对战，分 **运河时代** 与 **铁路时代** 两个时代，各 8 轮左右
- **首轮每名玩家仅 1 个行动，其余轮次每人 2 个行动**（2 人局真实规则为 3 个）
- **每次行动必须弃 1 张手牌**，行动后补满至 8 张
- 每个时代末结算一次分数；两个时代后总分离者胜

### 3.2 回合行动（每回合必须选一个）
1. **Build 建厂** — 打出地点牌（在指定城市）或工业牌（在自己的网络内），消耗对应资源(煤/铁)建设建筑；运河时代每城市每玩家限 1 个板块
2. **Network 建网** — 打出任意 1 张牌，建造运河（时代1，£3）/ 铁路（时代2，£5+1煤；双铁路 £15+1煤+1啤酒）
3. **Develop 研发** — 打出任意 1 张牌，消耗铁（1铁移除1个，最多2铁移除2个），移除面板上的低级建筑板块，解锁更高级；陶器 I/III 不可研发
4. **Sell 卖货** — 打出任意 1 张牌（不限地点牌），可一次行动售出多张未翻面产业板块
5. **Loan 贷款** — 打出任意 1 张牌，贷款 £30，收入等级 **-3 级（不是 -10，收入可为负，下限 -10，回合末负收入需偿付）**
6. **换取万能牌** — 弃 3 张手牌，换 1 张万能地点牌 + 1 张万能产业牌
7. **Pass 过牌** — 弃 1 张牌跳过本回合

### 3.3 六大产业
| 产业 | 作用 |
| --- | --- |
| 棉纺厂 Cotton Mill | 卖货得高分，昂贵 |
| 煤矿 Coal Mine | 供煤（建网/建厂消耗），廉价，收入高 |
| 铁厂 Iron Works | 供铁（建厂/研发消耗），可卡对手 |
| 制造厂 Manufacturer | 8 级，收益多样 |
| 陶器厂 Pottery | 最高 20 VP，需规划 |
| 啤酒厂 Brewery | 卖货必需啤酒，最核心的争夺资源 |

### 3.4 AI 视角的关键规则（状态空间要点）
- **资源获取**：煤需要连通（煤矿或经运河/铁路的市场）；铁可从任意铁厂拿（无需连通）；啤酒卖货时来源 = 己方啤酒厂(任意位置) + 连通的对手啤酒厂 + 市场桶
- **翻面得分**：运河时代结束时，所有运河与 1 级建筑移除；2 级以上保留到铁路时代并再次得分
- **重建规则**：己方同类型更高等级可重建；对手煤矿/铁厂仅在其资源全球耗尽时可重建
- **顺位规则**：每轮先手 = 上轮花钱最少者（平手保持当前顺序）；贷款使收入 -3 级
- **收入规则**：收入可为负（下限 -10），回合末负收入需偿付（卖板块折半 / 扣 VP）
- **胜负与平手**：最终 VP 最高者胜；平手看收入等级，再平手看剩余现金；起始资金 £17
- **非完全信息**：手牌隐藏（AI 用 ISMCTS / Determinization 处理），对手手牌不可读

---

## 4. 技术路线与阶段规划

总体架构：`TTS(Lua 隐蔽脚本) → FastAPI 服务 → ISMCTS + PyTorch → 悬浮窗 UI`

```
[阶段 1] Rust 游戏引擎     → 规则 / 状态 / 合法动作生成器 / 图连通性
[阶段 2] TTS 数据抽取       → Lua 脚本隐蔽读取盘面 → localhost JSON
[阶段 3] 深度学习           → Policy-Value 双头网络 + ISMCTS
[阶段 4] Self-Play 训练     → 自我对弈数据闭环迭代
[阶段 5] UI 集成            → PySide6 透明置顶悬浮窗实时推荐
```

详细 roadmap 见 `startup.md`（原始讨论稿）。

### 关键技术栈
- **引擎**：Rust（推荐）或 C++ — 千万不能拿 Python 写内核（训练性能瓶颈）
- **AI**：PyTorch + ISMCTS（信息集蒙特卡洛树搜索），AlphaZero 风格 Policy-Value 网络
- **接口**：Pybind11 / CFFI 桥接 Rust ↔ Python
- **服务**：FastAPI / gRPC
- **UI**：PySide6 无边框透明置顶悬浮窗

---

## 5. 仓库结构

```
brass-assistant/
├── AGENTS.md                     # 本文件：agent 背景知识
├── README.md                     # 项目总览
├── startup.md                    # 原始讨论稿（需求 / roadmap / 可行性分析）
├── docs/
│   ├── rules/
│   │   └── brass-birmingham-rules.md   # 规则精要（自整理）
│   ├── architecture.md           # 系统架构设计
│   └── reference-notes/          # 参考项目拆解笔记
│       ├── npow-brass-birmingham.md
│       └── ikegami-tts-brass.md
├── reference/                    # 克隆的参考项目（只读，勿修改）
│   ├── npow-brass-birmingham/    # HTML/JS 完整规则实现 + 启发式 AI
│   └── ikegami-tts-brass/        # TTS Mod Lua 脚本实现
└── src/                          # 本项目源码（按阶段填充）
```

---

## 6. 参考材料导航（重点）

### reference/npow-brass-birmingham
纯 HTML/CSS/JS 的完整规则实现（无构建、无依赖），是本项目 **规则引擎的黄金参考**：
- `js/gameData.js` — 所有常量：城市、连接、卡牌、市场、商家桶
- `js/gameState.js` — 游戏状态、市场机制、网络寻路、回合管理
- `js/gameLogic.js` — 7 种行动 + `getDisabledReason()`（合法性判定）
- `js/aiPlayer.js` — **488 行启发式 AI**（VP 等价权重评估），是轻量 AI 的起点参考
- `index.html` — 布局与交互

用法：写 Rust 引擎时，把 `gameLogic.js` 的规则校验逻辑逐条翻译成 Rust；`gameData.js` 是地图/卡牌数据的事实来源。

### reference/ikegami-tts-brass
TTS 官方热门的《伯明翰》Mod 的 **Lua 脚本**，是 TTS 集成与数据抽取的黄金参考：
- `lib/Global.ttslua`、`lib/State.ttslua`、`lib/App.ttslua` — 核心游戏逻辑
- `lib/App/Birmingham.ttslua` — 伯明翰专属逻辑
- `objs/` — 各类物体的局部脚本（读盘面/手牌 API 用法示例）
- `notes/Birmingham/` — 变更日志与信息说明

用法：写 TTS 隐蔽导出脚本时，参考这里的 `HandZone` / `getObjects` / `WebRequest` API 用法。

---

## 7. Agent 工作准则

1. **改动参考项目前**：`reference/` 目录是只读参考，如需借鉴请复制到 `src/` 再改，切勿直接修改
2. **先读后写**：涉及规则实现前，先查阅 `docs/rules/brass-birmingham-rules.md` 与对应参考源码
3. **文档同步**：修改架构/规则认知后，请同步更新对应 docs
4. **性能优先**：任何热路径（simulation / self-play）不得用 Python 实现核心循环
5. **合规红线**：规则文档为自整理摘要，不复制受版权保护的官方规则书原文；参考项目保持其原有 License
6. **提交规范**：仅在用户明确要求时进行 git commit
7. **规则分歧裁决（重要）**：`docs/rules/*.md` 为 AI 生成的摘要，**可能出错**；`reference/npow-brass-birmingham` 是完整可运行实现，视为更高可信度。**当规则文档与参考实现（或用户描述）存在分歧时，不要自行定论**——把冲突点明确列出来（附上两边各自的说法与依据位置），等待用户裁决后再写入引擎与文档。

---

## 8. 已知坑点（来自可行性分析）

- 动作空间极大：单步合法动作可能上百种（卡牌 × 位置 × 资源路径）
- 非零和博弈：蹭对手煤/铁/运河会送分，状态向量化时要体现这些耦合
- 手牌隐藏 → 必须用 Determinization 采样，不能做确定性搜索
- TTS 读对手手牌会被引擎隔离返回 nil（恰好符合规则，AI 不应依赖对手手牌）
- **消耗场上资源（对手煤/铁/酒）是免费的**：只翻面对手板块并推进其收入，不付钱给资源拥有者；只有从市场买资源才付市场价（`state.rs consume_from_city / find_coal_sources` 已正确实现，勿改成"给对手钱"）
- **建铁/煤矿会立即回现金**：连通市场时建厂即 `auto_sell_to_market` 卖出资源桶（`state.rs:604`），AI 评分应计入此收益（`heuristic_ai.rs market_cash_back`）
- **rail_era（2级+）板块翻面后 VP 在两个时代末各计一次**（`scoring.rs score_era` 在运河/铁路各调一次）；AI 建厂评分用 1.1x（运河）/2x（铁路）反映（`heuristic_ai.rs double_vp`）
- **可卖板块（棉/陶/制造）只有卖出才翻面**：AI 的 `estimate_flip_probability` 必须要求"连通接受该产业的商家 + 有啤酒可用"才给高翻面概率，否则低分（这是贷款→建厂→翻面→回收入经济链的关键）
- **啤酒按需供给**：酒桶数应匹配自己未翻面可卖板块的 `beers_to_sell` 需求 + 铁路建网缓冲，超出即浪费（`heuristic_ai.rs sellable_beer_demand`）
- **煤/铁源选择必须显式建模**（与啤酒选择同原则）：`Move::Build/Network/NetworkDouble/Develop` 都携带显式的 `coal`/`iron` 源列表；执行时校验（a）数量匹配（b）所选源当前可用（c）**免费优先**：有免费源时不得用市场源（防玩家故意不翻对手建筑）。`rules.rs` 的 `source_options`/`validate_source_choice` 是唯一事实来源，AI/heuristic/MCTS 都必须从这里取合法源，禁止执行函数内再自动挑源（`cheapest_coal_for_connection` 仅用于候选生成的成本估算）
- **Policy 头必须对合法槽位掩码后再算 CE loss（重要教训）**：策略表含 ~703 个 double-rail 幽灵槽（绝大多数状态非法）。若 loss 用全空间 `log_softmax`，幽灵槽仍进分母，初始约 53% 概率质量压在幽灵槽上，网络浪费梯度压制它们 → policy 极弱（贪心仅 ~7 VP）。修复见 `train.py compute_loss`：用每样本 `legal` 掩码 `masked_fill(~mask, -inf)`，并把非法槽 log_probs 清零避免 `0×-inf=NaN`。修复后贪心 7→50 VP、MCTS 反超启发式。MCTS 侧 `_masked_softmax` 一直是对的；只有训练侧曾漏掩码
- **双铁路允许不共享端点（已裁决）**：引擎与 npow 参考实现一致，两条铁路只需都在玩家网络内。因此策略表双铁路域是全部无序铁路对（703），不能用"共享端点"缩小
- **新 Schema（2026-08，已落地）**：分叉 policy（`type_head 7` + `goal_head 1316`，`logit(s)=type[t(s)]+goal[s]`，`t(s)` 来自 `policy.rs slot_type` 带算术）+ **4 玩家 value 头**（`Linear(256→4)` 去 tanh，单视角预测全部玩家终局 z，因 `encode.rs global` 已含每玩家 money/income/vp、`opp_hands` 含全部手牌）。Rust `nn_mcts::flush_net` 每 request 只发 **1 行**，Rust 侧合并分叉先验 + 直接用 4 向量（省 4× 视角编码）。**`net(batch)` 返回 3 元组 `(type, goal, value)`**；训练 loss 与单线程 `mcts.py` 用 `net.merge_logits` 合并。效果：BC 基线 `new_best.pt` @sims=1000 胜率 0.65/mean 93.3/**min 57（零灾难对局）**，优于旧 best_masked（0.50/77.8/20）
- **消耗型翻面 ≠ 卖货**：煤/铁/酒被消耗即自动翻面推进收入（`state.rs auto_sell_to_market` 等），无需 Sell 操作。诊断时"翻面数"不等于"Sell 次数"，不要据此误判模型行为

## 9. 已裁决的规则分歧（以用户规则书为准）

> 当参考实现与物理规则书冲突，且用户已裁决时，记录于此；后续勿再照 reference 实现改回。

| 冲突点 | npow 参考实现 | 用户裁决（规则书） | 裁决时间 |
| --- | --- | --- | --- |
| 商家万能牌数量（4 人局） | 2 张（2p/4p 组各 1 张累加，`gameData.js:390-394`） | **1 张**；4 人局 9 张 = 空3/万能1/棉2/箱2/陶1（各人数独立牌堆，见 `docs/rules` 5.4 表格） | 2026-08 |
| Sell 一次卖几张 | 可卖多张（`executeSell` + sellPlan） | **可一次卖多张**（与实现一致，规则文档已订正） | 2026-08 |
| 资金折算分 | 无（`money_value=0`） | 无（无"每£10=1VP"规则） | 2026-08 |
| 平手规则 | 未实现 | 收入等级高者胜，再平手现金多者胜 | 2026-08 |
| 起始资金 | £17 | £17 | 2026-08 |
| 2 人局每轮行动数 | 固定 2 | 2 人局为 3（引擎暂未支持 2 人局特殊处理，见「已知歧义」） | 待定 |
| 铁路时代是否再次埋每人1张弃牌 | 会再次 `seedDiscardPiles()`（`gameState.js:793-794`） | **不会**；仅初始设置时埋 1 张，铁路时代重洗后直接发 8 张，因此 4 人局铁路时代应完整 8 轮/64 动 | 2026-08 |
