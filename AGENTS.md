# AGENTS.md

Brass Assistant 是一个为桌游《工业革命：伯明翰》（Brass: Birmingham）构建的 AI 策略分析与实时推荐辅助系统。

长期目标：从零构建基于 ISMCTS + Policy-Value 神经网络 的《伯明翰》决策引擎。

## 关键技术栈

- 游戏引擎：Rust 模拟游戏规则、状态、合法动作生成器、图连通性
- AI：PyTorch + ISMCTS（信息集蒙特卡洛树搜索），AlphaZero 风格 Policy-Value 网络
- 接口：pyo3 桥接 Rust ↔ Python

## 仓库结构

engine 目录为 Rust 游戏引擎，python 目录为 Python AI 模块。

reference 目录为参考项目（只读，勿修改）。
reference/npow-brass-birmingham 是本项目游戏引擎规则相关的参考，如和规则文档有冲突，由用户裁决后再继续。
reference/ikegami-tts-brass 是 TTS 伯明翰 Mod 的 Lua 脚本，后续做 TTS 相关功能时可能需要参考。

docs 目录为本项目文档，其中 docs/archived/ 下的文档为历史记录，docs/tutorial/ 下为教程文档，非直接指令请勿阅读这两个子目录。
docs 目录下直接文档为当前有效文档，需要优先参考这些文档并维护其内容的准确性。其中重点文档需要时优先阅读和更新：

- `docs/brass-birmingham-rules.md`：完整游戏规则文档
- `docs/ai-tools.md`：Python AI 模块架构与操作手册
- `docs/ai-action-encoding.md`：动作特征编码说明
- `docs/engine-tools.md`：Rust 游戏引擎相关文档

## Agent 工作准则

1. 涉及到规则相关实现，以用户指令>规则文档>参考项目的顺序为准，不要自行定论
2. 文档同步：修改架构/规则认知后，请同步更新对应 docs 目录下的文档。
3. 工作语言：中文沟通，代码与文档中英文均可，参考已有内容保持一致。
4. 项目环境：使用项目根目录下的 .venv 虚拟环境，任何 python / cargo 命令均需在该虚拟环境下执行。不要在系统全局环境，其他目录下创建修改任何文件。

## 项目难点

- 动作空间极大：单步合法动作可能上千种（卡牌 × 位置 × 资源路径）
- 延迟奖励：每轮行动的收益可能在未来几轮才会体现。
- 手牌隐藏 → 必须用信息集搜索（ISMCTS）而非普通 MCTS
