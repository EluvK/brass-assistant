# 参考项目拆解：ikegami/tts_brass

> 本笔记供后续 agent 快速定位该 TTS Mod 的关键实现，避免重复通读源码。
> 该项目位于 `reference/ikegami-tts-brass/`（只读，勿修改）。

---

## 项目概览

TTS 上广受欢迎的《伯明翰》（及《兰开夏》）脚本化 Mod 的源码（Lua 脚本），由 Kinithin（ikegami）维护。
**`lib/` 下的 `.ttslua` 是给 TTS 桌面全局脚本用的源码；`.json` 文件是可直接导入 TTS 的成品 Mod。**

本项目 **TTS 集成与隐蔽数据抽取** 阶段的主要参考。

---

## 文件地图

| 路径 | 内容 |
| --- | --- |
| `lib/Global.ttslua` | 全局脚本入口（30 行，主要是 include 声明） |
| `lib/State.ttslua` | 状态管理与存档/读档 |
| `lib/App.ttslua` | 通用应用逻辑（2868 行） |
| `lib/App/Birmingham.ttslua` | 伯明翰专属逻辑（594 行） |
| `lib/App/Lancashire.ttslua` | 兰开夏专属逻辑（732 行） |
| `lib/Bowl.ttslua` | 资源碗脚本 |
| `lib/DistantMarket.ttslua` | 远程市场 |
| `objs/` | 各类物体的局部脚本（读盘面/手牌 API 用法示例） |
| `notes/Birmingham/` | 变更日志 + 使用说明（Information.txt 有操作指引） |
| `*.json` | TTS 成品 Mod（可直接导入 Saves 目录） |

---

## 对本项目的价值

### 1. TTS API 用法范例（阶段 2 必读）
- `objs/` 中的局部脚本展示了：`getObjects()`、`HandZone`、`getPosition()`、`getGUID()`、`setHidingMode()` 等读盘面/手牌 API
- `WebRequest.post()` 用于向本地服务发数据
- 结论：读 **公共盘面 + 己方手牌** 完全可行；对手手牌会被引擎隔离（返回 nil）

### 2. 规则交叉验证（阶段 1 辅助）
- `lib/App/Birmingham.ttslua` 可用于观察 TTS Mod 的边界行为与数据组织方式
- 但它同样只是参考实现，不应单独作为规则定论依据

### 3. 状态无状态化设计理念
- 根据 `notes/Birmingham/Information.txt`：脚本是 **quasi-stateless** 设计，所有状态（钱、盘面、卡牌、市场、商家）都能在脚本外手动改
- 启示：规则引擎的状态也应做成**纯数据、可序列化、可校验**的结构

---

## 使用提示（阶段 2 目标形态）

隐蔽导出脚本的设计要点（源自 startup.md 讨论结论）：
1. 不要改房间全局脚本（会被所有玩家看到），改为**个人 Saved Objects 小物件**
2. 用快捷键触发，走 `localhost` HTTP，不打印日志、无画面闪烁
3. 只读己方手牌 + 公共盘面，不触碰对手 Hand Zone

---

## 附：TTS 安装位置（Windows）
`%UserProfile%\Documents\My Games\Tabletop Simulator\Saves`
