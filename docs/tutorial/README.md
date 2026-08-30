# Brass AI 训练网络教程 —— 面向没有神经网络背景的工程师

这套教程的目标：**让一个完全不懂神经网络的工程师，看懂本项目 Python 训练侧的每一行代码，并在过程中顺手学会所需的全部神经网络知识。**

不需要任何机器学习背景；假设你会 Python、有一般后端工程经验。所有概念从零讲起，所有代码片段都来自本仓库的真实源码。

---

## 先看全局：AlphaZero 闭环

整套代码只做一件事——把下面这个循环跑起来：

```text
        ┌────────────────────────────────────────────────────────┐
        │                                                        ▼
  ┌───────────┐   visits=policy 目标   ┌──────────┐   训练   ┌────────────┐
  │ Rust MCTS │ ────────────────────▶ │  Sample  │ ───────▶ │  Trainer   │
  │ (当前网络  │      value 目标        │  (考题)  │          │  更新网络   │
  │  引导搜索) │ ◀──────────────────── └──────────┘          └─────┬──────┘
  └───────────┘        先验 P + 叶子 value                        │
        ▲                                                        │
        └──────────── 新网络权重(下一次搜索更强) ◀──────────────────┘
```

网络引导 MCTS 下棋 → 搜索产生训练数据 → 训练更新网络 → 更强的网络引导下一轮搜索。bootstrap（模仿学习）只是把第一圈数据从"搜索产物"换成"老师产物"。每个教程章节就是这个循环的一块零件。

---

## 章节目录

| 章 | 文件 | 回答的问题 | 主读代码 |
| --- | --- | --- | --- |
| 00 | [nn-basics](00-nn-basics.md) | 神经网络是什么？训练是怎么发生的？ | `toy_nn.py`（可运行） |
| 01 | [samples](01-samples.md) | 一局棋怎么变成训练数据？`Sample` 里有什么？ | `selfplay.py` |
| 02 | [encoding](02-encoding.md) | 牌面和动作怎么变成数字？ | `hierarchical_policy.py` + Rust 编码 |
| 03 | [network](03-network.md) | 网络内部每一层在做什么？（核心章） | `net.py` |
| 04 | [training](04-training.md) | 损失怎么算？batch 怎么进 GPU？checkpoint 怎么存？epoch 一轮轮学下去会发生什么、何时停？ | `train.py` |
| 05 | [mcts](05-mcts.md) | MCTS 在做什么？网络在搜索里的两个角色？闭环怎么合上？ | `rust_mcts.py`、`evaluate.py` |
| 06 | [end-to-end](06-end-to-end.md) | 一条命令跑通全流程：参数、产物、输出解读、断点恢复 | `bootstrap_imitation.py` |
| 07 | [appendix](07-appendix.md) | 术语表（中英对照）+ 常见问题 FAQ | — |
| 08 | [selfplay-loop](08-selfplay-loop.md) | **怎么持续变强**：回放缓冲、对手池、温度、门禁——组装生产级 self-play 迭代闭环 | `run_loop`、`SelfPlayPool`、`evaluate.py` |
| 09 | [evolution](09-evolution.md) | **演进方向全景**：度量/数据/网络/搜索/工程五层杠杆盘点 + 实验方法论 + 决策树 | `roadmap.md`、`ai-advise.md` |

**建议按编号顺序通读**。时间紧的最短路径：00 → 03 → 04（概念与训练核心），其余按需查；想规划训练路线、研究后续演进，加读 08 → 09。

---

## 阅读约定

- **shape 追踪**：全篇用 `(B, N, 301)` 这类标注讲数据流转（`B`=批大小、`N`=候选数）。对工程师来说，这是建立神经网络直觉最有效的单一工具。
- **术语中英对照**：术语首次出现给英文原文（代码注释和报错都是英文）；完整对照表在 [07 章](07-appendix.md)。
- **五段式讲解**：每个概念 = 是什么 → 为什么这里需要 → 本项目哪里 → 真实代码片段 → 输出被谁消费。
- **公式必配手算例子**：所有数学（softmax、交叉熵、PUCT）都有能手算的小数字例。
- **章节有验收标准和练习**：00 章的验收标准是能读懂 `train.py` 头部的损失公式；每章末尾的练习大多可动跑。

## 动手环境

```bash
# 教程配套的最小训练脚本(与本项目代码无关,只需 PyTorch)
./.venv/Scripts/python.exe docs/tutorial/toy_nn.py

# 端到端冒烟(06 章)
./.venv/Scripts/python.exe python/bootstrap_imitation.py \
    --games 20 --epochs 1 --eval-games 4 --eval-sims 20 --ckpt checkpoints/smoke.pt
```

## 与其他文档的分工

| 文档 | 性质 | 关系 |
| --- | --- | --- |
| 本教程 `docs/tutorial/` | **教学**：为什么这么设计、代码怎么读 | 从这里出发 |
| [ai-python-code-map.md](../ai-python-code-map.md) | API 字典：每个函数干什么 | 速查 |
| [ai-action-encoding.md](../ai-action-encoding.md) | 编码规范：301 维布局的权威定义 | 02 章引用其细节 |
| [architecture.md](../architecture.md) | 系统架构全景 | 背景补充 |
| [replay-design.md](../replay-design.md) | replay worker 协议 | 06 章 6.7 的延伸 |

## 维护

本教程与代码强耦合（引用真实函数名与行为）。**修改 `net.py` / `train.py` / `selfplay.py` 的结构或损失定义时，同步更新对应章节**——该约定已写入仓库根目录 `AGENTS.md` 的工作准则。
