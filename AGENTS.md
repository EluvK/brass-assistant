# AGENTS.md

本项目为桌游 Brass: Birmingham 构建的 AI 策略分析与实时推荐辅助系统。

## 仓库结构

engine 目录为 Rust 游戏引擎，模拟游戏规则、状态、合法动作生成器、图连通性
python 目录为 Python AI 模块。AlphaZero 风格 Policy-Value 网络
两者通过 pyo3 桥接，实现 Rust ↔ Python 的交互。

reference 目录为参考项目（仅当明确指令时才可访问，且只读，勿修改）。

docs 目录下的 `*.md` 为本项目有效文档，需要优先参考这些文档并维护其内容的准确性，非直接指令请勿阅读 docs 下其他子目录。

- `docs/brass-birmingham-rules.md`：完整游戏规则文档
- `docs/ai-tools.md`：Python AI 模块架构与操作手册
- `docs/ai-action-encoding.md`：动作特征编码说明
- `docs/engine-tools.md`：Rust 游戏引擎相关文档

## Agent 工作准则

1. 工作语言：中文沟通，代码与文档中英文均可，参考已有内容保持一致。
2. 项目环境：使用项目根目录下的 .venv 虚拟环境，任何 python / cargo 命令均需在该虚拟环境下执行。不要在系统全局环境，其他目录下创建修改任何文件。
