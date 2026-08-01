# Brass Assistant（代号 Brass-Mind）

为桌游《工业革命：伯明翰》（Brass: Birmingham）构建的 **AI 策略分析与实时推荐辅助系统**。

在 Tabletop Simulator (TTS) 真人对战时，读取公共盘面与己方手牌，由本地 AI 在 10~15 秒内完成数万次模拟推演，通过置顶悬浮窗给出 Top-3 操作建议。

## 技术栈

- **规则引擎**：Rust（目标：每秒数万次 step，支撑 Self-Play 训练）
- **AI**：PyTorch + ISMCTS，AlphaZero 风格 Policy-Value 双头网络
- **桥接**：Pybind11 / CFFI
- **服务**：FastAPI / gRPC
- **UI**：PySide6 透明置顶悬浮窗
- **数据采集**：TTS Lua 隐蔽脚本

## 阶段路线图

```
[阶段 1] Rust 游戏引擎     → 规则 / 状态 / 合法动作生成器 / 图连通性
[阶段 2] TTS 数据抽取       → Lua 脚本隐蔽读取盘面 → localhost JSON
[阶段 3] 深度学习           → Policy-Value 双头网络 + ISMCTS
[阶段 4] Self-Play 训练     → 自我对弈数据闭环迭代
[阶段 5] UI 集成            → PySide6 透明置顶悬浮窗实时推荐
```

## 仓库结构

```
├── AGENTS.md                  # 背景知识（给后续 AI agent）
├── startup.md                 # 原始讨论稿（需求 / roadmap / 可行性）
├── docs/
│   ├── rules/                 # 规则精要（自整理，非官方规则书）
│   ├── architecture.md        # 系统架构设计
│   └── reference-notes/       # 参考项目拆解笔记
├── reference/                 # 克隆的参考项目（只读）
│   ├── npow-brass-birmingham/ # HTML/JS 完整规则实现 + 启发式 AI
│   └── ikegami-tts-brass/     # TTS Mod Lua 脚本实现
└── src/                       # 本项目源码（按阶段填充）
```

## 文档导航

| 文档 | 用途 |
| --- | --- |
| `AGENTS.md` | 新 agent 必读：项目背景、规则速览、参考导航、工作准则 |
| `docs/rules/brass-birmingham-rules.md` | 游戏规则精要（引擎实现参考） |
| `docs/architecture.md` | 系统架构与模块边界 |
| `docs/reference-notes/*` | 两个参考项目的拆解与借鉴策略 |
| `reference/*` | 参考项目源码（只读） |

## 合规声明

本仓库为学习研究项目。规则文档为自整理摘要，非官方规则书转载；参考项目保留其原有 License。游戏版权归 Roxley Games 及原作者所有。
