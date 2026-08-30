# 08 · self-play 闭环：让网络持续变强

> **本章目标**：bootstrap 教会网络"模仿老师"，但老师不会变强。本章讲清楚**持续自我提升的机制设计**——怎么把现有零件拼成一个"生成 → 训练 → 评估 → 门禁 → 再生成"的迭代闭环，每个旋钮（回放池、对手池、温度、门禁）为什么存在、调错了会怎样。
>
> 主读代码：`train.py` 的 `run_loop` / `train_steps` / `step_lr`、`mp_selfplay.py` 的 `SelfPlayPool`、`evaluate.py`、`selfplay.py` 的 `play_game_with_roles`。
> 前置：05 章（闭环概念）、04 章（训练 API）、06 章（bootstrap）。
> 状态说明：本章标注"**待建**"的部分是当前代码库还没有的顶层逻辑——这正是 [roadmap.md](../roadmap.md) 中期（阶段 3）的工作内容，也是你读完本教程后最值得动手的方向。

---

## 8.1 bootstrap 的天花板在哪

bootstrap 是"一次性数据集 + 固定老师"：启发式老师的能力不会提升，网络最多逼近老师的上限。要继续变强，只能让**数据自己进化**——这就是 05 章闭环图要落地的部分：

> 当前网络引导 MCTS 下棋 → 搜索的 visit 分布**优于**网络自己的直觉（搜索做了推演）→ 用它训练出新网络 → 新网络引导更高质量的搜索。

关键洞察：**训练目标永远比当前网络高一点**。MCTS 花几百次模拟得到的结论，网络一次前向就学走了——下次搜索的起点更好，模拟的结论更好，如此螺旋上升。

先把闭环四块零件的现状盘点清楚：

| 零件 | 需要什么 | 现状 |
| --- | --- | --- |
| 数据生成 | 多进程 self-play、对手池 | ✅ `SelfPlayPool`（8.4） |
| 训练 | 可复用历史数据的训练 API | ✅ `train_steps` + `step_lr`（8.3） |
| 评估 | 对 heuristic、对历史版本的基准 | ✅ `benchmark_mcts_vs_heuristic`、`play_game_with_policies`（8.5） |
| **顶层入口** | 把三块串成带门禁的迭代循环 | ❌ **待建**（`run_loop` 是教学骨架，8.2） |

roadmap 中期第 1 条写的就是最后一行："基于现有模块组合……不重写底层能力"。

---

## 8.2 参考骨架：解剖 `run_loop`

`train.py` 里的 `run_loop` 是闭环的**最小可读实现**（当前没有入口调用它，属于教学/实验骨架）：

```python
@dataclass
class LoopConfig:
    iterations: int = 10
    games_per_iter: int = 8
    mcts_sims: int = 80

def run_loop(net, loop, on_iters=None):
    mcts = RustISMCTS(net, RustMCTSConfig(c_puct=2.5, max_depth=10))
    trainer = Trainer(net, loop.train or TrainConfig())
    for it in range(loop.iterations):
        samples, avg_vps, _ = play_batch(mcts, loop.games_per_iter, sp_cfg)  # ① 用当前网生成
        losses = trainer.train_on_samples(samples)                            # ② 训练
        if on_iters:
            stop = on_iters(it, trainer, stats)                               # ③ 观察点,可提前终止
    return net
```

三步循环干净地展示了闭环骨架，但要变成生产级循环，它缺四样东西——正好对应后面四节：

1. **数据不复用**（8.3）：每迭代丢弃上一轮全部样本；
2. **对手没有多样性**（8.4）：四个座位都是同一个最新网络；
3. **没有门禁**（8.5）：训练完直接用，哪怕新网络更弱；
4. **单进程生成**（9.6 的工程话题）：没用 `SelfPlayPool`。

---

## 8.3 回放缓冲（replay buffer）：数据要复用

**问题**：`play_batch` 一局几百个样本，生成一局要几分钟（每步几十次 MCTS 模拟），训练一批只要毫秒。把最贵的数据用一次就扔，是最大的浪费。

**更深层的原因**：本迭代的新样本只代表"**上一版**网络"的棋力。训练后网络变了，但新样本和几轮前的样本对"学下棋"这个目标都有效——自我对弈的改进是渐进的，相邻版本的网络数据分布差异不大。

04 章 4.6 提到的 `train_steps` 就是为这个准备的：

```python
def train_steps(self, samples, n_steps, batch_size=None):
    """`n_steps` gradient steps, each on a random minibatch drawn WITH
    replacement from `samples` — supports a growing replay buffer that
    reuses self-play data across iterations. Does NOT step the LR scheduler."""
```

它从给定的样本池**有放回**随机抽批训练（经典 experience replay），且刻意不碰学习率调度器——调度节奏由 `step_lr()` 单独控制，让余弦周期跟随"迭代数"而不是"训练步数"（04 章 4.7）。

**缓冲管理（待建，属于入口层逻辑）**：

- 每迭代开始：把本轮新样本 push 进缓冲；
- 缓冲有上限（比如最近 5~20 个迭代的样本），超出就淘汰最老的——**太老的数据是负资产**：它们由能力弱得多的旧网络产生，策略分布与当前搜索差距过大，会往回拽训练；
- 每迭代训练：`trainer.train_steps(buffer, n_steps=…)`，`trainer.step_lr()` 一次。

配比直觉：每迭代"新数据全看几遍 + 老数据混着抽"。混合比例是最值得实验的超参之一（9.7 章）。

---

## 8.4 对手池与多样性：防止"近亲繁殖"

**纯 self-play 的隐患**：四个座位都是同一个网络时，它只会遇到"自己人"的策略分布。自我对弈可能漂移（drift）或塌缩（collapse）——大家默默达成一些只在"内战"里成立的开局默契，遇到别的风格就崩。

`mp_selfplay.py` 的 docstring 一句话点明了设计动机：

> Matchmaking: with `mm_prob > 0`, each game has one rotating "learner" seat (current net, whose samples are collected) and opponent seats drawn from a pool of historical nets (plus the current net). This anchors training against a stable reference and **prevents the drift/collapse seen in pure self-play**.

机制（`SelfPlayPool.generate` 的 `mm_pool` / `mm_prob` 参数，worker 端 `_worker_fn`）：

```python
learner = gi % 4                                  # 每局轮换"学习者"座位
roles = [mcts.search] * 4
for seat in range(4):
    if seat == learner:
        continue
    if np.random.rand() < mm_prob:                # 概率引入历史对手
        opp = pool[np.random.randint(len(pool))]  # 从历史 checkpoint 池抽
        roles[seat] = opp.search
samples, _ = play_game_with_roles(roles, cfg, collect={learner})  # 只收集学习者样本
```

三个设计点值得咀嚼：

- **learner 座位轮换**（`gi % 4`）：样本不偏向某个座位——尽管引擎规则对座位本应对称，轮换是免费的保险；
- **只收集 learner 的样本**（`collect={learner}`）：对手池里的历史网络也参与产生 visit 分布，但那些样本属于"旧策略的目标"，不该进当前训练集；
- **历史池怎么维护**（待建）：典型做法是保留最近 K 个 gate 通过的 checkpoint。历史对手太弱没有锻炼价值，太强又全是输棋样本（value 目标全是 0.0 也不健康）——池子里放"近期各版本"是实践上最好用的折中。

### 其余的多样性旋钮

- **根噪声**（05 章 5.3 的 Dirichlet）：self-play 恒开（`add_root_noise=True`），防止开局死板；
- **温度**（05 章 5.4）：目前 `SelfPlayConfig.temperature` 对整局统一。AlphaZero 的经典做法是**按手数衰减**——前 N 手高温（多探不同开局），之后降温贪心（认真分胜负）。这个改动落在 `_sample_move` 的调用处，是 08 章练习的题目；
- **seed**：每局独立 seed（`SelfPlayPool` 里 `seed_base + worker_id*100_000 + …`），洗牌器/发牌天然多样。

---

## 8.5 门禁（gate）：每次更新必须证明自己更强

没有门禁的迭代是危险的：坏超参、脏数据、运气差的一批样本，都可能让新网络**更弱**，而你毫无察觉地继续在上面迭代。门禁规则一句话：

> **挑战者（新网络）必须在基准对局中击败卫冕者（当前网络），否则不替换。**

好消息是拼出对局所需的所有零件都在：两个 `RustISMCTS`（各自抱一个 checkpoint）+ `evaluate.play_game_with_policies`（它接受任意"每座位一个策略函数"，05 章 5.5 的 heuristic benchmark 就是这么拼的）：

```python
def benchmark_net_vs_net(challenger_mcts, champion_mcts, games=40, sims=60):
    """挑战者 vs 卫冕者;座位轮换防 seat 偏差。返回挑战者胜率。"""
    pol_c = lambda state: challenger_mcts.search(state, sims, False).best
    pol_p = lambda state: champion_mcts.search(state, sims, False).best
    wins = 0
    for g in range(games):
        seat = g % 4                                   # 挑战者轮换坐 4 个座位
        policies = [pol_c if p == seat else pol_p for p in range(4)]
        vps, ranking = play_game_with_policies(policies, seed=10_000 + g)
        if ranking[0] == seat:
            wins += 1
    return wins / games
```

（示意代码：`benchmark_mcts_vs_heuristic` 的同款结构，只把 heuristic 换成第二个 MCTS。）

**判定纪律**（衔接 05 章 5.5 的统计警告）：

- 4 人局对局里"击败"指**拿第一名**，随机基线 25%——挑战者胜率显著超过 25% 才算赢；
- games=40 的胜率粒度是 2.5%，随机波动仍大。门禁阈值建议定在基线之上留足余量（比如 >32%），或者加大 `games`；
- **失败怎么处理**：保留卫冕者，检查本次迭代（数据量？学习率？对手池？），修正后重跑。绝不"先替换看看下次能不能涨回来"——坏基座上的迭代全是沉没成本；
- 通过门禁后：checkpoint 落版本目录（如 `checkpoints/iter-007.pt`），并把它加入 8.4 的历史对手池。

> **为什么不用对 heuristic 的胜率当门禁？** 可以当**外部参照**（它衡量"离人类启发式多远"），但门禁要的是"这次更新有没有变好"——对手必须是上一个版本的自己，否则老师（heuristic）的短板会成为玻璃天花板。两个基准各有用途，见 8.7。

---

## 8.6 组装：生产级迭代入口的伪代码（待建）

把 8.3~8.5 拼起来，就是 roadmap 中期说的"重建 self-play 顶层入口"。伪代码（真实实现按 06 章风格做成 argparse 入口）：

```text
输入: bootstrap checkpoint, 迭代数 I, 每迭代局数 G, 模拟数 S, 缓冲窗口 W

buffer = []            # 回放缓冲(8.3)
pool   = [bootstrap]   # 历史对手池(8.4)
champion = load(bootstrap)

for it in 1..I:
    # ① 生成:多进程, learner=当前网, 对手大概率来自 pool (8.4)
    samples = SelfPlayPool.generate(champion, games=G, sims=S,
                                    mm_pool=pool, mm_prob=0.5,
                                    temperature=temperature_schedule(it))
    buffer.push(samples); buffer.evict_older_than(W)      # ② 复用+淘汰 (8.3)

    # ③ 训练:在缓冲上做 n_steps 步;学习率按迭代推进 (8.3 / 04 章 4.6)
    challenger = copy(champion)
    trainer = Trainer(challenger)
    trainer.train_steps(buffer, n_steps=N_STEPS)
    trainer.step_lr()

    # ④ 门禁:挑战者 vs 卫冕者,轮换座位 (8.5)
    if benchmark_net_vs_net(challenger, champion, games=40) > GATE_WIN_RATE:
        champion = challenger
        pool.append(champion); pool.trim(K)
        save(champion, f"checkpoints/iter-{it:03d}.pt")
        log(it, "accepted", stats)
    else:
        log(it, "rejected", stats)      # 保留卫冕者,不替换
```

对照它再读一遍真实模块，你会发现**每一行调用的底层能力都已存在**：`SelfPlayPool.generate`（mp_selfplay.py）、`Trainer.train_steps` / `step_lr`（train.py）、`play_game_with_policies`（evaluate.py）、checkpoint 原子写（06 章 6.3 的 `save_checkpoint` 模式）。"待建"的只是这个组装层和它的参数——这就是"不重写底层能力"的含义。

---

## 8.7 迭代监控：该盯哪些曲线

每个迭代记录一次，画成时间序列。（注意与模仿阶段"固定数据集逐 epoch"的监控节奏区分开——那套节奏的停止判断见 04 章 4.10。）健康/病态的典型形态：

| 指标 | 来源 | 健康形态 | 警报形态 |
| --- | --- | --- | --- |
| 门禁胜率（vs 上版本） | 8.5 | 50% 上下波动，略高于 50% | 长期 <25%（越训越弱）或长期 >75%（门禁太松/步长太小） |
| benchmark vs heuristic | `benchmark_mcts_vs_heuristic` | 缓慢爬升后趋平 | 倒退（门禁没拦住的话） |
| loss 分项（policy/rank/winner/econ） | 04 章 4.2 | 缓慢下降后平台 | policy 降、门禁不涨 = 过拟合数据分布而非变强 |
| policy 熵 | 04 章 4.8 | 平稳缓降 | 急速坠落（过早一言堂）|
| 对 teacher 的 top-k | `evaluate_policy` | 缓慢上升 | **teacher 退役后此指标自然失效**（9.3），换成对历史版本的 top-k |
| 平均局长 / 无效动作率 | 生成侧 | 稳定 | 局长暴涨（下成循环拉锯）或无效动作率上升（引擎契约出问题） |

数据侧还要盯**缓冲构成**：新/老样本比例、被淘汰的迭代年龄——这些决定了 8.3 的复用策略是否失衡。

---

## 8.8 常见反模式清单

1. **无门禁更新**：最危险的错误。训练必然有噪声，没有 8.5 的闸门，一次坏迭代就污染之后所有迭代。
2. **只训最新数据**：每迭代新数据用一次就丢（`run_loop` 骨架的现状）。生成是瓶颈时这是自我设限（8.3）。
3. **温度全程为 0**：开局多样性枯竭，self-play 数据分布越来越窄，最终"内战无敌、外战外行"。温度全程为 1 也一样糟：终局阶段靠运气分胜负，value 目标噪声大。**按手数衰减**是标准解。
4. **sims 一开始就拉满**：模拟数是数据质量的"单价"。早期网络弱，高 sims 只是把弱策略搜得更深，性价比极低；先低 sims 快速迭代，网络成型后再抬（9.7 的实验纪律）。
5. **门禁、训练、数据同时改**：一次改一个变量（9.7），否则涨了不知道为什么涨，跌了不知道为什么跌。
6. **忽略统计粒度**：用 10 局对局决定是否替换网络，等于抛硬币做工程决策（05 章 5.5、06 章 6.4）。

---

## 练习

1. **跑通骨架**：写 20 行脚本实例化 `PolicyValueNet` + `Trainer` + `RustISMCTS`，调 `run_loop(LoopConfig(iterations=2, games_per_iter=2, mcts_sims=20), on_iters=lambda it, t, s: print(s) or False)`，亲眼看到闭环转两圈（CPU 也能跑，就是慢）。
2. **温度调度**：给 `SelfPlayConfig` 加一个 `temperature_by_move(move_index) -> float` 钩子（如前 30 手 1.0，之后线性降到 0.1），在 `play_game_with_roles` 的 `_sample_move(result, cfg.temperature)` 调用处接入。注意保持默认行为不变（这是一次无风险的第一次改造）。
3. **门禁函数**：把 8.5 的示意 `benchmark_net_vs_net` 写成真实代码（放进 `evaluate.py` 或独立脚本），用 bootstrap checkpoint 对抗它自己的浅搜索版（sims=15 vs sims=60），验证"高 sims 应该赢"——这同时是你对评测管线的一次校准。
4. **组装最小闭环**：按 8.6 伪代码实现 `python/train_selfplay.py`（迭代 3 轮、每轮 4 局、缓冲窗口 2），打印每轮的门禁结果。跑完你会拥有这个项目第一个端到端的自我提升入口。

下一步读 [09 章](09-evolution.md)：闭环转起来之后，"变强"的所有杠杆盘点与实验方法论。
