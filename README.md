# Brass Assistant（代号 Brass-Mind）

为桌游《工业革命：伯明翰》（Brass: Birmingham）构建的 **AI 策略分析与实时推荐辅助系统**。

在 Tabletop Simulator (TTS) 真人对战时，读取公共盘面与己方手牌，由本地 AI 在 10~15 秒内完成数万次模拟推演，通过置顶悬浮窗给出 Top-3 操作建议。

## 技术栈

- **规则引擎**：Rust（合法动作生成 / 图连通性 / 市场与时代结算，已跑通）
- **AI**：PyTorch + ISMCTS，AlphaZero 风格 Policy-Value 双头网络
- **桥接**：PyO3（`brass_engine` 扩展，maturin 构建）
- **服务**：FastAPI / gRPC（未开始）
- **UI**：PySide6 透明置顶悬浮窗（未开始）
- **数据采集**：TTS Lua 隐蔽脚本（未开始）

## 当前进度

| 模块 | 状态 |
| --- | --- |
| Rust 引擎（7 种行动 + 3 层 AI + 31 项测试） | ✅ 已提交 |
| PyO3 绑定 + State→Tensor 接口（`state_to_tensor` / `legal_moves` / 策略表 1316 槽） | ✅ 已提交 |
| AI 层：Policy-Value 网络 + Python ISMCTS + 多进程自对弈 + 持久 Trainer | ✅ 已实现（待提交） |
| **masked policy loss 修复**（幽灵槽污染软最大化分母 → 贪心 policy 7→50 VP，MCTS 反超启发式） | ✅ 已验证 |
| Self-Play 正式训练闭环（从 `best_masked.pt` 起点） | ⏳ 下一步 |
| TTS 数据抽取 / 悬浮窗 UI | ⏳ 未开始 |

## 阶段路线图

```
[阶段 1] Rust 游戏引擎     → ✅ 规则 / 状态 / 合法动作生成器 / 图连通性
[阶段 2] TTS 数据抽取       → ⏳ Lua 脚本隐蔽读取盘面 → localhost JSON
[阶段 3] 深度学习           → ✅ Policy-Value 双头网络 + ISMCTS（已跑通，反超启发式基线）
[阶段 4] Self-Play 训练     → ⏳ 自我对弈数据闭环迭代（多进程基建已就绪）
[阶段 5] UI 集成            → ⏳ PySide6 透明置顶悬浮窗实时推荐
```

## 仓库结构

```
├── AGENTS.md                  # 背景知识（给后续 AI agent）
├── startup.md                 # 原始讨论稿（需求 / roadmap / 可行性）
├── docs/
│   ├── rules/                 # 规则精要（自整理，非官方规则书）
│   ├── architecture.md        # 系统架构 + 训练日志（含 masked-loss 突破）
│   └── reference-notes/       # 参考项目拆解笔记
├── reference/                 # 克隆的参考项目（只读）
│   ├── npow-brass-birmingham/ # HTML/JS 完整规则实现 + 启发式 AI
│   └── ikegami-tts-brass/     # TTS Mod Lua 脚本实现
└── src/
    ├── engine/                # Rust 引擎 + PyO3 绑定（maturin 构建 brass_engine）
    └── ai/                    # Python AI 层
        ├── brass_ai/          # net / mcts / selfplay / train / mp_selfplay / evaluate
        ├── experiments/       # 实验脚本（exp_masked_loss 等）
        ├── train_mp.py        # 多进程 AlphaZero 训练入口
        └── bootstrap_imitation.py  # 启发式行为克隆基线
```

## 运行

```bash
# 构建 Rust 绑定（需 maturin + venv，见 src/engine/）
cd src/engine && .venv/Scripts/python.exe -m maturin develop

# 从 repo 根目录训练/评估（GPU 版 torch 见 docs/architecture.md）
PYTHONPATH=src/ai ./src/engine/.venv/Scripts/python.exe src/ai/experiments/exp_masked_loss.py
PYTHONPATH=src/ai ./src/engine/.venv/Scripts/python.exe src/ai/train_mp.py
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
