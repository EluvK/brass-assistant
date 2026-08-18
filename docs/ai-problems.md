# Python AI 待修复问题

本文记录审阅 `src/ai/` 时发现的实现问题、架构风险和测试缺口。除非特别说明，均为静态审阅或轻量命令验证结果；本次未修改 Python 源码。

## P1：`replay_net.py --help` 必然异常

- 位置：[src/ai/experiments/replay_net.py](../src/ai/experiments/replay_net.py:109)
- 现象：执行 `python src/ai/experiments/replay_net.py --help` 会在 `argparse` 格式化帮助时抛出 `ValueError: unsupported format character ','`。
- 原因：`--seat` 的 help 文本含有未转义的 `%`（`seed % 4`）。`argparse` 会对 help 文本做 `%` 插值。
- 影响：使用 `--help` 无法查阅参数；任何会触发帮助格式化的路径都会失败。
- 建议：将字面 `%` 改为 `%%`，或移除该字符；补充一个逐个执行所有 CLI `--help` 的 smoke test。
- 验证：2026-08-18 已在项目 venv 中复现；其余主要入口的 `--help` 正常。

## P1：gate 回滚没有回滚优化器状态

- 位置：[src/ai/train_mp.py](../src/ai/train_mp.py:189)
- 现象：基准退化时仅执行 `net.load_state_dict(best_state)`，而 `Trainer.optimizer` 的 AdamW 动量、方差估计仍对应被拒绝的候选模型。
- 影响：下一轮会在“最佳模型权重 + 被拒绝模型的优化器状态”这一不一致组合上训练；`--gate` 不能完全兑现回滚语义，实验可比性变差。
- 建议：保存并恢复完整 trainer state（模型、optimizer、scheduler、epoch），或者在回滚后明确重置优化器并记录该策略；增加回归测试，断言回滚后 optimizer state 与被接受 checkpoint 相同。

## P1：恢复训练会丢失 gate 的历史基线和候选池

- 位置：[src/ai/train_mp.py](../src/ai/train_mp.py:144)、[src/ai/train_mp.py](../src/ai/train_mp.py:154)
- 现象：`--resume` 恢复了 `latest.pt` 与 replay，但 `best_win`、`best_median`、`best_state` 一律从 `-1/-1/None` 开始；matchmaking pool 也只从 `--ckpt` 重建，未恢复此前已接受的快照。
- 影响：恢复后的第一次 benchmark 必然成为新的 “best”，即使它比中断前最佳模型差；`--gate` 与 matchmaking 的跨进程连续性被破坏。
- 建议：把 gate 最佳完整 trainer state、基准指标与 matchmaking snapshot 清单作为 run 状态持久化；恢复时读取它们。若暂不实现，命令行应拒绝 `--resume --gate` 或明确标为新一段 run。

## P2：matchmaking 的 learner 座位并未跨 worker 全局轮换

- 位置：[src/ai/brass_ai/mp_selfplay.py](../src/ai/brass_ai/mp_selfplay.py:69)
- 现象：learner 由每个 worker 内部的 `gi % 4` 决定。所有 worker 的 `gi` 都从 0 开始，因此例如 `--games_per_worker 2` 时，每轮只收集座位 0、1 的 learner 样本，座位 2、3 永远不会成为 learner。
- 影响：在 `mm_prob > 0` 时，样本收集和当前网络所面对的对手位置产生系统性座位偏差；与“rotating learner seat”的注释不一致。
- 建议：把 worker id 或全局游戏序号纳入座位计算，例如 `(worker_id * games + gi) % 4`；增加多 worker、少 games 的座位分布测试。

## P2：自博弈的随机性不可由 `--seed` 完整复现

- 位置：[src/ai/brass_ai/mp_selfplay.py](../src/ai/brass_ai/mp_selfplay.py:65)、[src/ai/brass_ai/selfplay.py](../src/ai/brass_ai/selfplay.py:90)
- 现象：游戏局面 seed 确实由 `seed_base`、worker 和 offset 派生，但对局采样和 matchmaking 使用模块级 `np.random`，worker 中没有显式播种；PyTorch worker RNG 也未播种。
- 影响：相同 CLI `--seed` 不能保证同一自博弈轨迹或对手抽样，难以复现异常样本和比较训练运行。
- 建议：按每局稳定派生 NumPy/PyTorch RNG，并将 RNG 策略写入 manifest；让 `_sample_move` 和 matchmaking 接受 generator 而不是使用全局随机源。

## P2：评测对截断局仍返回中间 VP 与排名

- 位置：[src/ai/brass_ai/evaluate.py](../src/ai/brass_ai/evaluate.py:27)
- 现象：`play_game_with_policies` 达到 `max_moves` 或没有可行动作时直接返回 `state.player_vps()`、`state.final_ranking()`，没有确认 `state.game_over`，也没有像自博弈一样拒绝样本或像 `net_all_vs_all.py` 一样显式调用 `finish_game()`。
- 影响：异常局可能把未完成局面的 VP 混入 benchmark，进而影响 `--gate` 的接受/拒绝结果。
- 建议：统一回合驱动的终止策略。首选是未自然终局就抛错并让 benchmark 标记失败；若规则允许强制结算，则统一调用 `finish_game()` 并在结果中标记为截断。
- 测试缺口：尚无覆盖 `max_moves=1` 的评测测试。

## P2：四人网络契约与可配置玩家数不一致

- 位置：[src/ai/brass_ai/net.py](../src/ai/brass_ai/net.py:38)、[src/ai/brass_ai/selfplay.py](../src/ai/brass_ai/selfplay.py:59)
- 现象：网络价值头固定 `N_PLAYERS = 4`、对手手牌固定为 3 组，但 `SelfPlayConfig.players` 公开为可修改参数，并传给可支持 2--4 人的 Rust `GameState`。
- 影响：两人或三人自博弈会生成与网络输出维度不一致的样本/损失，或在 bridge 编码处失败；接口暗示的能力与实际不符。
- 建议：当前阶段明确断言 `players == 4`，或者把输入、value head、标签和 MCTS 契约全面参数化；增加非法人数配置测试。

## P3：样本 IPC 包使用硬编码尺寸，和 bridge 单一事实源脱节

- 位置：[src/ai/brass_ai/mp_selfplay.py](../src/ai/brass_ai/mp_selfplay.py:95)
- 现象：空样本的预分配形状硬编码为 `(17,49)`、`(6,39)`、`(50,)`、`(35,)`、`(105,)`、`(1316,)`，而网络/输入模块其他位置使用 `brass_engine` 导出的常量。
- 影响：改变 Rust 特征或策略表尺寸时，非空样本路径可能正常但空样本路径悄然产生不兼容数据，或在后续拼接时失败。
- 建议：通过 bridge 常量或集中式 tensor spec 生成所有形状；增加“更新契约后空 worker 回包”的测试。

## P3：策略温度定义偏离常见 AlphaZero 访问次数温度

- 位置：[src/ai/brass_ai/selfplay.py](../src/ai/brass_ai/selfplay.py:76)
- 现象：当前采样权重为 `softmax(visits / temperature)`；常见 AlphaZero 定义为 `visits ** (1 / temperature)` 后归一化。
- 影响：访问次数稍大时当前指数形式会比幂函数更快地集中到最大访问动作，`temperature` 的含义与经验参数不一致，降低探索或使调参直觉失效。
- 建议：先确认这是有意设计还是实现偏差。若要遵循 AlphaZero，改为幂温度并处理零访问；为固定 visit 分布添加概率单元测试。此项需要策略层决策，不应未经确认直接修改。

## P3：`bc_baseline.py` 是独立旧实验，且不会创建输出目录

- 位置：[src/ai/experiments/bc_baseline.py](../src/ai/experiments/bc_baseline.py:95)
- 现象：脚本直接 `torch.save(net.state_dict(), args.out)`，默认路径 `checkpoints/new_best.pt` 的父目录不存在时会失败；训练参数、保存格式和评测方式也与正式入口不同。
- 影响：首次运行体验不稳定，产物不能直接用于 `--resume`，容易被误用为正式训练流水线的一部分。
- 建议：创建输出父目录，并在脚本名称/帮助中标注“历史对照实验”；长期可迁移其有价值部分到正式入口后删除重复实现。

## 验证基线

本次审阅执行：

```powershell
$env:PYTHONPATH = "src/ai"
src/engine/.venv/Scripts/python.exe -m pytest src/ai/tests -q
```

结果：`12 passed in 6.79s`。当前测试覆盖输入形状、网络输出、搜索可执行性、完整自博弈、截断自博弈拒绝、分片 round-trip、训练损失下降及模仿筛选；未覆盖本文列出的 gate 恢复、worker seat/RNG、评测截断、CLI help 与多人数契约问题。
