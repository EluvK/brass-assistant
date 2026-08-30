# AGENTS.md — Brass Assistant 项目背景知识

> 本文件是给后续进入本仓库工作的 AI agent / 协作者 使用的背景知识文档。

## 1. 项目定位

Brass Assistant 是一个为桌游《工业革命：伯明翰》（Brass: Birmingham）构建的 AI 策略分析与实时推荐辅助系统。

核心愿景：
- 在 Tabletop Simulator (TTS) 真人对战时，读取公共盘面 + 己方手牌
- 由本地 AI 服务在 10~15 秒内完成数万次模拟推演
- 通过置顶悬浮窗给出 Top-3 操作建议及预估收益

长期目标：从零构建基于 ISMCTS + Policy-Value 神经网络 的《伯明翰》决策引擎。

## 2. 开发者背景

- 具备 后端工程能力（工程化思维、数据结构、架构设计、性能优化）
- 弱深度学习背景，相关概念有基础了解，可能需要详细的指导

---

## 3. 游戏规则速览

需要参阅完整规则时，参考见 `docs/brass-birmingham-rules.md`。

## 4. 阶段规划

阶段 1: Rust 游戏引擎     → 规则 / 状态 / 合法动作生成器 / 图连通性
阶段 2: 深度学习           → Policy-Value 双头网络 + ISMCTS
阶段 3: Self-Play 训练     → 自我对弈数据闭环迭代
阶段 4: TTS 数据抽取       → Lua 脚本隐蔽读取盘面 → localhost JSON
阶段 5: UI 集成            → PySide6 透明置顶悬浮窗实时推荐

### 关键技术栈
- 游戏引擎：Rust 模拟游戏规则、状态、合法动作生成器、图连通性
- AI：PyTorch + ISMCTS（信息集蒙特卡洛树搜索），AlphaZero 风格 Policy-Value 网络
- 接口：pyo3 桥接 Rust ↔ Python

## 5. 仓库结构

reference 目录为参考项目（只读，勿修改）。
reference/npow-brass-birmingham 是本项目游戏引擎规则相关的参考，如和规则文档有冲突，由用户裁决后再继续。
reference/ikegami-tts-brass 是 TTS 伯明翰 Mod 的 Lua 脚本，后续做 TTS 相关功能时可能需要参考。

docs 目录为本项目文档，其中 docs/archived/ 下的文档为历史记录，docs/ 下的其余文档为当前有效文档。历史文档非直接指令请勿阅读和参考。

`engine` 为 Rust 游戏引擎，`python` 为 Python AI 模块。

## 7. Agent 工作准则

1. 涉及到规则相关实现，以用户指令>规则文档>参考项目的顺序为准，不要自行定论
2. 文档同步：修改架构/规则认知后，请同步更新对应 docs
3. 工作语言：中文沟通，代码与文档中英文均可，参考已有内容保持一致。

## 8. 项目难点

- 动作空间极大：单步合法动作可能上千种（卡牌 × 位置 × 资源路径）
- 延迟奖励：每轮行动的收益可能在未来几轮才会体现。
- 手牌隐藏 → 必须用信息集搜索（ISMCTS）而非普通 MCTS
