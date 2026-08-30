# 01 · 一局棋怎么变成训练数据

> **本章目标**：搞清楚训练数据的"生产车间"——一个 `Sample` 里有什么、两种出题方式（模仿学习 / 自我对弈）分别怎么造出它、为什么工程上要存 snapshot 延迟物化。
>
> 主读代码：`python/brass_ai/selfplay.py`（627 行）、`python/bootstrap_imitation.py` 的生成入口。
> 前置：00 章。02 章会解释"状态/动作张量里到底是什么数"，本章只关心**数据的组织和流转**。

---

## 1.1 训练数据的原子单位：`Sample`

神经网络的训练需要海量"考题 + 标准答案"。本项目里一道考题就是一个 `Sample`（`selfplay.py`），它横跨"下棋时"和"训练时"两个阶段，字段比普通 dataclass 多几个"延迟填写"的槽位：

```python
@dataclass
class Sample:
    pid: int                          # 这个决策轮到谁(0~3)
    board: np.ndarray | None = None   # (24,49)  棋盘 49 格 × 24 平面 → 网络输入
    links: np.ndarray | None = None   # (7,39)   39 条连接 × 7 平面   → 网络输入
    global_vec: np.ndarray | None = None  # (168,) 全局信息           → 网络输入
    own_hand: np.ndarray | None = None    # (35,)  己方手牌           → 网络输入
    opp_hands: np.ndarray | None = None   # (105,) 三个对手手牌        → 网络输入
    candidates: np.ndarray | None = None  # (N,301) N 个候选动作的特征行 → 网络输入
    policy: np.ndarray | None = None      # (N,)   标准答案:每个候选该走的比例
    rank: np.ndarray | float = 0.0        # (4,)   终局名次/4(结束时回填)
    winner: np.ndarray | float = 0.0      # (4,)   冠军 one-hot(结束时回填)
    era: int = 0                          # 记录时处于运河(0)还是铁路(1)时代
    econ: np.ndarray = None               # (2,)   (收入等级, 现金)目标
    snapshot: bytes | None = None         # 引擎状态快照(延迟物化用,见 1.5)
    teacher_canonical: str | None = None  # 老师选的动作(模仿模式用)
```

把它读成一张"考卷"：

| 考卷部位 | 字段 | 什么时候填 |
| --- | --- | --- |
| 题面 | `board / links / global_vec / own_hand / opp_hands` | 下棋时立刻记录 |
| 选项 | `candidates`（N 个候选动作，每行 301 维特征） | 下棋时**或**训练前物化（1.5） |
| 标准答案（policy） | `policy`（每个候选该分到多少概率质量） | 下棋时（由出题人决定） |
| 标准答案（value） | `rank / winner / econ` | **对局结束后**回填——下棋时没人知道结局 |
| 存根 | `snapshot + teacher_canonical` | 模仿模式的省内存方案（1.5） |

"答案要等对局结束才能回填"是理解 self-play 代码的关键：每一步先记下题面和 policy 答案（此时值目标是空占位 `0.0`），整局下完后统一把终局名次、冠军、经济抄写到这一局的所有 Sample 上（`_generate_imitation_game` 和 `play_game_with_roles` 的结尾都是这个"回填"循环）。

---

## 1.2 两种出题人

### 出题人 A：启发式模仿（heuristic imitation，bootstrap 默认）

Rust 引擎里有一个人类知识写成的启发式 AI（`state.heuristic_candidates()` / `choose_heuristic()`）。让它自己和自己下棋，把**它每一步的选择**作为标准答案——这是"抄作业"（模仿学习，imitation learning）。

为什么先抄作业？纯 AlphaZero 式自我对弈从**随机初始化的网络**起步，早期对局质量极差，需要海量对局才能"自学成才"，成本高得离谱。先用便宜的启发式对局把网络拉到"会玩"的水平（warm-start 暖启动），之后再切换到昂贵的 MCTS 自我对弈精修——这是 AlphaZero 类系统的标准起手式，也是 `bootstrap_imitation.py` 名字的由来。

### 出题人 B：MCTS 自我对弈（self-play）

当前网络引导 MCTS 搜索后，**搜索的访问次数分布**就是标准答案——"搜索认为该走的比例"。这是 AlphaZero 的核心闭环，细节在 05 章。本章只需要知道：它走 `play_game_with_roles`，题面字段当场填好（`candidates` 直接算全合法集），policy 来自 visit 分布。

bootstrap 阶段只用出题人 A；05 章再回来读出题人 B 的代码。

---

## 1.3 一局模仿对局的完整数据流

走读 `_generate_imitation_game(args)`（worker 进程里跑，一局一进程任务）：

```python
state = be.GameState(seed=seed, players=players)   # Rust 开新局
while not state.game_over and moves < max_moves:
    pid = state.current_player_id
    # 问 Rust 启发式要"带分数的候选短列表 + 它的选择"
    _features, _scores, _card_scores, canon, _index, _score, _card_score = (
        encode_teacher_candidates(state)
    )
    # 全合法模式(v4 默认):只存快照和老师的选择,题面之后再生(1.5)
    local.append(Sample(
        pid=pid, era=state.era, rank=0.0, winner=0.0,
        snapshot=bytes(state.snapshot()), teacher_canonical=canon,
    ))
    state.apply_move(canon)                        # 执行老师的选择,进入下一步
```

循环结束后（对局打完），统一回填答案：

```python
vps = np.asarray(state.player_vps(), dtype=np.float64)
rank, winner = _rank_targets(state.final_ranking(), state.player_count)
for sample in local:
    sample.rank = rank
    sample.winner = winner
    if sample.econ is None:
        sample.econ = np.asarray(final_econ[sample.pid], dtype=np.float32)
```

`_rank_targets` 把引擎的终局排名（按 VP → 收入 → 现金破平局）转成两个目标向量：

```python
for place, pid in enumerate(ranking, start=1):
    rank[pid] = place / n_players          # 第 1 名 → 0.25,第 4 名 → 1.0
winner[ranking[0]] = 1.0                   # 冠军 one-hot
```

注意 rank 是**除以人数归一化的名次**，不是分数差：这样 4 人局和（未来可能支持的）其他人数局的目标在同一尺度上，且同一局内保持大小顺序。搜索时用的"价值"再翻一次：`value = 1 − rank`（第 1 名 0.75 分，第 4 名 0.0 分），保证"越大越好"，与 Rust 搜索的终局回传值同尺度。

### 经济目标（econ）的"盖章"规则

econ 目标 `(收入等级, 现金)` 按时代分两段：

- **运河时代的样本** → 盖上**运河时代结束那一刻**的 `(income, money)`（运河存下的钱是铁路时代扩张的本钱，是关键里程碑）；
- **铁路时代的样本** → 盖上**终局**的 `(income, money)`。

对应代码：模仿对局在 `apply_move` 前后检测 `prev_era == 0 and state.era == 1`，一旦跨入铁路时代就把 `state.canal_econ()` 盖到之前所有运河样本上（`canal_samples` 列表）；对局结束再把 `final_econ()` 补给没盖过章的样本。这个设计让"运河头"和"铁路头"各自只见一种目标定义（详见 04 章损失部分）。

---

## 1.4 policy 答案的两种形式

**模仿学习（shortlist 模式）**：启发式给每个候选打了分，直接把分数 softmax 成分布（温度 1.0）：

```python
score_values = teacher_scores.numpy().astype(np.float64)
weights = np.exp((score_values - score_values.max()) / 1.0)
policy = coalesce_equivalent_policy(candidate_tensor.numpy(),
                                    (weights / weights.sum()).astype(np.float32))
```

`减 max 再 exp` 是数值稳定技巧：教师分数可能很大，直接 `exp` 会溢出成 inf（减去最大值后指数最大为 1，softmax 结果不变）。

**模仿学习（full-legal 模式，v4 默认）**：policy 是老师的 one-hot——但有一个微妙处理。v4 动作特征刻意不编码"手牌第几张"这类执行身份（02 章），于是会出现**若干个具体动作特征完全相同**（执行不同、打分上无法区分）的"等价类"。网络对相同输入必然输出相同分数，如果答案只指向其中一个，训练信号就自相矛盾。所以 `teacher_equivalence_policy` 把 one-hot **均匀摊到整个等价类**上：

```python
equivalent = np.all(array == array[teacher_index], axis=1)  # 特征完全相同的行
policy = equivalent.astype(np.float32)
return policy / policy.sum()                                 # 均匀分布
```

`coalesce_equivalent_policy` 对 MCTS visit 目标做同样的事（把某候选的访问数摊给和它特征相同的所有候选）。

---

## 1.5 snapshot 延迟物化：省内存的核心工程决策

**问题**：full-legal 模式下，每个 Sample 要带 `(N, 301)` 的候选矩阵，而 N（全合法动作数）可以到几百甚至上千。一局 300 步 × 平均几百候选 × 301 维 × 4 字节 ≈ 每局数百 MB。1000 局的 shard 直接爆内存。

**解法**（`_generate_imitation_game` 的注释原话是 "Persisting only this snapshot ... avoids the full-legal replay/IPC explosion"）：下棋时只存两样小东西——

1. `snapshot = bytes(state.snapshot())`：Rust 引擎状态的完整字节快照（几十 KB）；
2. `teacher_canonical`：老师选的动作字符串。

候选矩阵 `(N,301)` 是**状态的确定性函数**，需要时再算。训练时 `materialize_sample` 恢复它：

```python
def materialize_sample(sample: Sample) -> Sample:
    if sample.snapshot is None:
        return sample
    state = be.GameState.from_snapshot(sample.snapshot)   # 从字节恢复引擎状态
    ...
    canonicals, candidates = encode_legal_candidates(state)  # 重新枚举全合法候选
    teacher_index = canonicals.index(sample.teacher_canonical)
    policy = teacher_equivalence_policy(candidates.numpy(), teacher_index)
    return Sample(..., candidates=candidates.numpy(), policy=policy, ...)
```

还原后还会做一致性校验：快照的当前玩家/时代必须和记录时一致，老师动作必须在恢复出的合法集里——防止快照与元数据错配悄悄污染训练数据。

`candidates=None` 的 Sample 在 04 章的训练循环里会看到对应的物化调度：子进程池**提前一批**物化，与 GPU 训练流水线重叠。

**uint8 二次压缩**：301 维特征全是 0.25 的整数倍（02 章），所以跨进程传候选行时按 ×4 打包成 uint8（`compress_candidate_features`），传输量再降 4×；到 GPU 上再 `/4` 还原成 float。这就是 `Sample.candidates` 可能是 uint8 的原因——`_to_batch` 和 `net.forward` 都处理两种 dtype。

---

## 1.6 shard：把数据流出到磁盘

`generate_imitation_sample_shards`（bootstrap 调用的入口）边生成边落盘：

- 每攒够 32768 个 Sample，flush 成一个 `imitation-000000.pkl`（pickle 序列化的 `list[Sample]`）；
- 父进程**从不**在内存里保留完整数据集——每个 shard 可独立加载、训练、释放；
- 支持质量过滤：`min_avg_vp` / `min_vp` 只保留"全员 VP 都不太差"的对局（防止畸形对局污染数据），`max_attempts` 默认 10× 防止过滤条件太严导致死循环。

bootstrap 复用 `--sample-dir` 或默认目录 `{ckpt}.imitation/`：目录里已有 shard 就跳过生成（中断恢复时不用重下一千局棋）。

---

## 1.7 并行生成

对局之间完全独立（不同 seed），所以用 `ProcessPoolExecutor` 并行下棋（`generate_imitation_samples` / `_generate_imitation_with_sink` 共享调度核心）。两个值得注意的工程细节：

- Windows 上必须用 `spawn` 启动子进程（重新 import），所以 worker 环境变量里把 `OPENBLAS_NUM_THREADS` 等设为 1——否则每个子进程各起一套 BLAS 线程池，内存先爆；
- 调度保持"每个 worker 同时只有一个任务在飞"，而不是 `Executor.map` 一次性提交全部任务——后者会让几千局的序列化结果同时挤在内存里。

这些细节和训练循环（04 章）的多进程思想一脉相承：**spawn 代价 + 内存上限决定调度形态**。

---

## 1.8 小结

```text
Rust 引擎开局 ──▶ 每步:问老师要选择 ──▶ 记 Sample(题面/snapshot + teacher)
                        │                                 │
                        ▼                                 │ 对局结束
                apply_move 推进                            ▼
                                            回填 rank/winner/econ 答案
                                                   │
                                                   ▼
                                  按 32768 个一 shard 落盘 .pkl
                                                   │
                                                   ▼ (训练时,04 章)
                              materialize: snapshot → (N,301) 候选 + 等价类 policy
```

记住三件事就够：

1. `Sample` = 考题（状态五件套 + 候选）+ 答案（policy 当场定，value 终局回填）；
2. bootstrap 的答案是**老师的选择**（模仿），05 章换成 MCTS visit 分布（自我对弈）；
3. 全合法候选太胖，用 snapshot 延迟物化 + uint8 压缩，这是本项目数据管线的主线工程。

## 练习

1. 用小脚本生成 2 局模仿数据（不开质量过滤），打印其中一个 Sample 的各字段 shape：
   ```python
   import sys; sys.path.insert(0, "python")
   from brass_ai.selfplay import generate_imitation_samples
   samples = generate_imitation_samples(2, full_legal_candidates=False, workers=1)
   s = samples[0]
   print(s.pid, s.board.shape, s.candidates.shape, s.policy.shape, s.econ, s.rank, s.winner)
   ```
   注意 `policy` 的总和是 1（概率分布），非零位通常集中在教师喜欢的少数候选上。
2. 找一个 `policy` 有多个非零位的样本，打印 `candidates[policy > 0]` 的前几行，观察这些"老师看好"的动作分别是什么。
3. 思考题：为什么 `rank` 目标用"名次/人数"而不用"VP 差值"？（提示：一个样本的 value 目标要能在**任意局面**下与"这局最后谁赢"对齐，且跨对局可比；VP 差值既不保序于名次，也难以归一。）
