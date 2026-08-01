# 参考项目拆解：npow/brass-birmingham

> 本笔记供后续 agent 快速定位该参考项目的关键实现，避免重复通读源码。
> 该项目位于 `reference/npow-brass-birmingham/`（只读，勿修改）。

---

## 项目概览

纯 HTML/CSS/JS 的数字版《伯明翰》（无构建、无依赖、无框架）。约 5900 行，是本项目 **规则引擎的黄金参考**。
用浏览器打开 `index.html` 即可本地跑起来（或 `python3 -m http.server`）。

---

## 文件地图（含行数）

| 文件 | 行数 | 内容 | 对 Rust 引擎的用途 |
| --- | --- | --- | --- |
| `js/gameData.js` | 604 | 全部常量：城市、连接、卡牌、市场、商家 | **数据事实来源**，直接翻译成常量表 |
| `js/gameState.js` | 977 | 状态、市场机制、网络寻路、回合管理 | State 结构 + 图连通算法参考 |
| `js/gameLogic.js` | 1125 | 7 种行动 + `getDisabledReason()` 合法性判定 | 规则校验逐条翻译目标 |
| `js/aiPlayer.js` | 488 | 启发式 AI（VP 等价权重） | 轻量 AI / 评估函数起点 |
| `js/uiManager.js` | 1327 | 阶段栏、卡牌选择、日志、弹窗 | 交互层（UI 阶段再参考） |
| `js/boardRenderer.js` | 961 | SVG 版图渲染 | 不需要 |
| `js/industryIcons.js` | 127 | 产业图标 | 不需要 |
| `js/main.js` | 117 | 入口 | 不需要 |
| `index.html` | 223 | 布局 | 不需要 |

---

## 关键实现位置

### 规则数据 → `js/gameData.js`
- 城市节点、连接（运河/铁路）、城市槽位（可建产业类型）
- 卡牌构成（地点牌/工业牌/万用牌）
- 市场轨道（煤/铁/棉/陶）、商家点位
- 各产业各等级的 **费用（£ / 煤 / 铁）、收入、VP** —— 所有数值的源头

### 网络寻路 → `js/gameState.js`
- 如何判断"网络是否连通到某城市 / 市场 / 商家"
- 连通性与煤供给的绑定逻辑
- 建议：Rust 里用图 + 增量 BFS / 并查集重写，这里读的是语义

### 合法性判定 → `js/gameLogic.js`
- 每个行动的先决条件、资源校验、卡牌校验
- **`getDisabledReason()`**：返回"为什么不能做"——等于一份完整的规则校验清单，翻译成 Rust 时按这条函数逐条对照

### 启发式 AI → `js/aiPlayer.js`
- `vpEquivalent()`：把 VP/收入/资金/灵活性统一折算成 VP 当量分数
- `cardUsefulness()`：卡牌保留价值评估（Scout/弃牌决策）
- `estimateFlipProbability()`：翻面概率估计
- 结论：**阶段 1 的"规则 AI"可直接借鉴此文件**，比纯随机强得多

---

## 借鉴策略（给 Rust 引擎阶段）

1. 先读 `gameData.js`，把数据结构定义成 Rust 常量（可用脚本或手工翻译）
2. 再读 `gameLogic.js` 的 `getDisabledReason()`，逐条实现合法性判定
3. 用 `aiPlayer.js` 的权重思想写启发式 AI 作为 Baseline
4. 用 npm 化/手工对比：JS 输出与 Rust 输出对同一盘面跑随机对弈做一致性校验

> 注意：该项目是 2-4 人 hotseat 逻辑，未含 TTS 相关代码。
