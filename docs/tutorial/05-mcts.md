# 05 · MCTS：网络怎么被使用，以及 AlphaZero 闭环

> **本章目标**：理解蒙特卡洛树搜索（MCTS）在做什么、网络在搜索里扮演哪两个角色、搜索结果怎么变成训练数据（闭环的关键一步）、以及怎么用对局评测网络棋力。
> 主读代码：`python/brass_ai/rust_mcts.py`、`python/brass_ai/selfplay.py` 的对局部分、`python/brass_ai/evaluate.py`。
> 说明：搜索树的构建、PUCT 选择、回传这些"重活"在 Rust 侧（`engine/src/bridge` → `nn_mcts.rs`），Python 负责把网络挂上去和消费结果。本章以 Python 视角为主，Rust 侧只讲机制不讲实现。
> 前置：00/03 章（policy 与 value 的含义）、04 章（训练目标）。

---

## 5.1 为什么光有网络还不够

训练好的网络能干什么？给它一个局面 + 一批候选，它立刻给出：

- 每个候选的先验分（policy logits）——"**感觉**哪个动作像好棋"；
- 局面的终局名次预测（value）——"**感觉**这个局面最后谁赢"。

两个"感觉"都是**一步到位的直觉**，没有推演。人类高手不是这样下棋的：会先算"如果我建这条铁路，对手下一手卖货给我断连怎么办"。MCTS 就是把"推演"交给程序：**往前模拟大量可能的后续对局，用模拟结果修正直觉**。

---

## 5.2 MCTS 的直觉：一个越来越准的"未来模拟器"

把当前局面作为树根，反复执行以下四步（每次叫一次模拟/simulation），本项目 bootstrap 默认 60 次（`--eval-sims 60`），正式 self-play 通常上百：

1. **选择（selection）**：从根出发，在每个分叉口挑"最值得再探一步"的分支往下走（标准见 5.3）；
2. **扩展（expansion）**：走到一个还没展开过所有孩子的节点，把它的一部分合法动作挂成新孩子（本项目用候选短列表，见 5.5）；
3. **评估（evaluation）**：到了没走过的局面，**不真的下到终局**，直接问网络："这个局面最后四个人排名如何？"——用 `value = 1 − rank` 当这一支的评分（03 章：这个定义与 Rust 终局回传同尺度）。偶尔模拟到真正的终局（游戏结束），就用真实名次；
4. **回传（backup）**：把这次评分沿来路传回根，沿途每个节点更新两个统计量——**访问次数 n**（这条路被探过多少次）和**平均价值 Q**（走过的那些次平均打多少分）。

几百次模拟后，根节点每个候选的 **n（访问次数）** 就是"搜索认为这个动作多好"的最终答案：访问越集中 = 越看好。AlphaZero 的关键洞察是：**这个访问分布本身就是比网络原始输出更好的 policy 训练目标**（搜索过程已经替网络做完了推演）。

### PUCT：每个分叉口怎么挑分支

选择的标准是 PUCT 公式（`c_puct` 是本项目默认 2.5）：

```text
score(分支) = Q + c_puct × P × √N / (1 + n)

Q = 这个分支的平均价值(搜过的结论)
P = 网络先验给它打的分(policy,03 章)
N = 父节点的总访问数, n = 这个分支自己的访问数
```

两项的分工：**Q 是利用（exploitation）**——已知的肥沃分支继续挖；**P × √N/(1+n) 是探索（exploration）**——访问少的分支有加成，且父节点越"热"（N 大）探索项越大。数字例子（c_puct=2.5）：

```text
根节点 N=10。两个候选:
A: 访问 n=8, Q=0.6, 先验 P=0.5
B: 访问 n=2, Q=0.4, 先验 P=0.4

A: 0.6 + 2.5 × 0.5 × √10/(1+8) = 0.6 + 0.44 ≈ 1.04
B: 0.4 + 2.5 × 0.4 × √10/(1+2) = 0.4 + 1.05 ≈ 1.45   ← 选 B
```

B 虽然 Q 值低，但"看过的次数少"让探索项占上风——模拟资源被匀给欠采样的分支，避免搜索一开始就迷信先验。随着 B 被访问，`1+n` 增大、探索项衰减，最终访问收敛到真正好的分支上。

### ISMCTS：手牌是藏着的，怎么办

普通 MCTS 假设"整个局面完全可见"，但 Brass 里对手的手牌是**隐藏信息**。本项目用的是信息集 MCTS（ISMCTS，I = Information Set）：**每次模拟开始时，把看不到的信息随机补全**（Rust 引擎的 `determinize()`——给三个对手随机发一手合理的手牌），然后在补全后的"可能世界"里正常搜索。不同模拟补不同的牌，最终统计出来的是"在**所有可能世界**里平均表现最好"的动作——这正是对隐藏信息做决策的正确姿势。（训练数据里状态编码带了对手手牌（02 章），是因为 self-play 四个座位都是自己的网络、数据本身无秘密；搜索时的信息隐藏由 determinize 处理。）

---

## 5.3 架构：树在 Rust，网络在 Python

```text
┌─────────────── Rust (engine) ───────────────┐      ┌──── Python ────┐
│ GameState.search_net(net_fn, sims, …)       │      │ PolicyValueNet │
│   逐次模拟:选择→扩展→问网络→回传             │      │  (GPU 上)      │
│   攒一批叶子局面(≤ batch_size=64) ───────────┼────▶ │ net_fn(batch)  │
│   收到 (logits, value) 后继续搜索            │◀──── │  policy_value  │
└─────────────────────────────────────────────┘      └────────────────┘
```

树的全部状态（节点、访问计数、Q 值）都在 Rust 里；Python 网络被包装成一个**回调函数**塞给 Rust（`_engine.pyi` 里的 `NetCallback` 类型）。关键性能设计是**攒批**：Rust 不会每片叶子单独问一次网络（GPU 一次算 1 个局面浪费 99% 算力），而是把待评估的叶子攒到最多 `batch_size=64` 个，一次回调批量推理——和 04 章 batch 训练是同一个 GPU 吞吐逻辑。

Python 侧的包装（`rust_mcts.py`）：

```python
def make_net_fn(net, device="cuda"):
    def net_fn(board, links, global_vec, own_hand, opp_hands, candidates, candidate_mask):
        batch = {   # numpy 行向量 → 恢复成网络的形状合同(03 章),搬上 GPU
            "board": torch.from_numpy(np.asarray(board, dtype=np.float32))
                        .reshape(-1, be.BOARD_PLANES, be.BOARD_CELLS),
            ...
        }
        out = net.policy_value(batch, action_features, mask)   # 03 章的推理入口
        return (
            out["candidate_logits"].detach().cpu().numpy(),    # → numpy 给 Rust
            out["value"].detach().cpu().numpy(),
        )
    return net_fn
```

边界两侧是两个世界：Rust 递 numpy 数组，Python 转 tensor 过网络再转回来。`detach().cpu()` 就是"脱离计算图、搬回内存"。

### 配置与结果对象

```python
@dataclass
class RustMCTSConfig:
    c_puct: float = 2.5            # PUCT 探索系数(5.2)
    max_depth: int = 10            # 模拟最大深度(防止对局模拟走太远太慢)
    dirichlet_alpha: float = 0.3   # 根噪声参数(见下)
    dirichlet_weight: float = 0.15
    batch_size: int = 64           # 攒批大小
    candidate_k: int = 4           # 候选短列表宽度;0 = 全合法集
    device: str = ...

@dataclass
class SearchResult:
    best: str | None               # 最终选择的 canonical 动作
    visits: dict                   # {候选 id: 访问次数}
    canon_by_candidate: dict       # {候选 id: canonical 字符串}
```

两个配置值得展开：

- **候选短列表（`candidate_k=4`）**：扩展节点时不挂全部合法动作（可能上千），只挂 Rust 候选生成器挑出的前几个"几何体"（与启发式老师用的是**同一个**生成器，[ai-action-encoding.md §6](../ai-action-encoding.md)）。这让 60 次模拟集中火力在像样的分支上。代价是**候选生成器成为整个系统的头号风险**——好动作若不在短列表里，搜索永远看不见它（同文档 §7 把它列为三层风险之首）。
- **根噪声（Dirichlet noise）**：self-play 时给根节点的先验混入随机噪声（权重 0.15）——强制搜索偶尔去探"网络不看好"的开局，保证自我对弈数据的多样性（AlphaZero 标准技巧）。评测/实战时 `add_root_noise=False` 关掉。

---

## 5.4 从搜索结果到训练样本：闭环的最后一块拼图

`play_game_with_roles`（selfplay.py）每一步做的事：

```python
canonical_candidates, candidate_tensor = encode_legal_candidates(state)  # 全合法候选
result = roles[pid](state, cfg.sims, True)     # 跑 MCTS(根噪声=True)
...
policy = coalesce_equivalent_policy(           # visit 分布 → 训练目标
    candidate_tensor.numpy(),
    _candidate_policy(canonical_candidates, result))
s = Sample(..., policy=policy, candidates=candidate_tensor.numpy(), era=state.era)
...
chosen = _sample_move(result, cfg.temperature) # 按 visit 分布采样实际落子
summary, ok = state.apply_move_raw(chosen)
```

三个环节：

1. **`_candidate_policy`**：MCTS 在短列表上搜索，visit 记在候选 id 上；这个函数按 canonical 字符串把 visit 数**对齐回全合法候选的顺序**（02 章：canonical 是跨边界身份证），归一化成 `(N,)` 分布。短列表之外的候选 visit 为 0——它们确实没被探索，目标如实反映。
2. **`coalesce_equivalent_policy`**：把特征完全相同的候选的 visit 数摊平均（01 章的等价类规则，MCTS 版）。
3. **`_sample_move`**：实际落子按 visit 分布的**温度采样**——温度 1.0 严格按比例随机（增加对局多样性），温度趋近 0 变成贪心取 best。数字例子：

   ```text
   visits = {A: 60, B: 30, C: 10}
   温度 1.0 → 采样概率 (0.6, 0.3, 0.1)
   温度 0.5 → exp(差值/0.5) → 概率 (≈1.0, ≈0.0, ≈0.0)  几乎贪心
   ```

   代码里 `counts − counts.max()` 再 exp 的注释值得读：不减最大值时，几百次访问除以小温度会指数溢出成 NaN——又一次"数值稳定优先于公式简洁"。

落子后 `apply_move_raw` 若报告动作失效（引擎偶发的边界 bug），防御性回退到 `result.best`、再退到第一个合法动作——搜索可以冒进，引擎状态绝不能卡死。对局结束后与模仿模式相同的回填：`rank/winner/econ` 抄到这一局所有 Sample 上（01 章）。

### 闭环全图（整个教程的主图）

```text
        ┌────────────────────────────────────────────────────────┐
        │                                                        ▼
  ┌───────────┐   visits=policy 目标   ┌──────────┐   训练   ┌────────────┐
  │ Rust MCTS │ ────────────────────▶ │  Sample  │ ───────▶ │  Trainer   │
  │ (当前网络  │      value 目标        │  (01 章) │  (04 章) │  更新网络   │
  │  引导搜索) │ ◀──────────────────── └──────────┘          └─────┬──────┘
  └───────────┘        先验 P + 叶子 value                        │
        ▲                                                        │
        └──────────────── 新网络权重(下一次搜索更强) ◀──────────────┘
```

每一轮 self-play：当前网络引导搜索下棋 → 搜索的 visit 分布与终局名次成为新样本 → 训练更新网络 → 更强的网络再引导下一轮搜索。网络既是搜索的"引擎零件"，又是搜索的"学徒"——数据自己产生自己，这就是 AlphaZero。bootstrap 的模仿阶段只是把这个闭环的第一圈数据换成了老师给的（01 章）。

---

## 5.5 对局评测：`evaluate.py`

训练有没有用，最终要靠**对局**说话。`benchmark_mcts_vs_heuristic` 让网络 MCTS 与 Rust 启发式打对抗：

```python
for g in range(games):
    seat = g % players                       # MCTS 轮换坐 1/2/3/4 号位
    policies = [mcts_pol if p == seat else heuristic_policy for p in range(players)]
    vps, ranking = play_game_with_policies(policies, seed=g, players=players)
    if ranking[0] == seat:
        wins += 1                            # 拿第一名算赢(4 人局!)
```

设计要点：

- **座位轮换**（`seat = g % players`）+ **种子轮换**（`seed=g`）：每局 MCTS 坐不同位置、打不同开局，防止"只会坐 2 号位"的假棋力；
- **4 人局的胜率基线**：均匀随机是 25%（1/4）。"MCTS vs 启发式 win_rate=50%" 的真实含义是：MCTS 第一名次数是随机水平的两倍，且对手是写了人类知识的启发式；
- `mcts_mean` / `base_mean` 是 VP 均值对比——比胜率更连续，20 局的小样本下看 VP 差更稳；
- `play_game_with_policies` 里同样有失效动作的防御性回退（保底取第一个合法动作）。

**统计意义提醒**：20 局的 win_rate 粒度是 5%，随机波动巨大。用它判断"新 checkpoint 是否更强"时，差异至少要超过 10~15 个百分点才可靠；更严谨的做法是加大 `--eval-games`（代价：每局 60 模拟 × 数百步的推理时间）。

`benchmark_net_vs_heuristic` 是一步到位的入口：给网络，建 MCTS，跑 benchmark。

---

## 5.6 小结

> MCTS 用"选择(PUCT 平衡利用与探索)→扩展(候选短列表)→评估(网络 value 直报终局)→回传"四步模拟未来；手牌隐藏用 determinize 补全成多个可能世界（ISMCTS）；树在 Rust、网络在 Python，攒批回调喂 GPU。搜索产出两样东西：一个更强的落子（visit 分布采样），以及一份更准的训练答案（visit 分布 + 终局名次）——后者喂回网络完成 AlphaZero 闭环。评测则用座位/种子轮换的对抗局，把"棋力"变成 win_rate 和 VP 均值两个可跟踪的数。

## 练习

1. 用 bootstrap 训出的 checkpoint 跑一次小评测，亲眼看指标：
   ```bash
   ./.venv/Scripts/python.exe -c "
   import sys; sys.path.insert(0, 'python')
   import torch
   from brass_ai.net import PolicyValueNet
   from brass_ai.rust_mcts import RustISMCTS, RustMCTSConfig
   from brass_ai.train import Trainer
   from brass_ai.evaluate import benchmark_mcts_vs_heuristic
   trainer = Trainer(PolicyValueNet())
   trainer.load_state_dict(torch.load('checkpoints/bootstrap.pt', map_location='cpu'))
   mcts = RustISMCTS(trainer.net, RustMCTSConfig(c_puct=2.5, max_depth=10, device='cpu'))
   print(benchmark_mcts_vs_heuristic(mcts, sims=20, games=8))
   "
   ```
   （没有 checkpoint 时先跑 06 章的冒烟命令。）把 `sims` 从 20 调到 60，观察 win_rate 的变化——模拟次数就是"思考时间"。
2. 手算验证：根节点 N=100，候选 A（n=50, Q=0.7, P=0.6）与候选 B（n=10, Q=0.3, P=0.35），c_puct=2.5，PUCT 各是多少、选谁？把 N 换成 1000 再算一遍，体会"探索项随 N 增大而整体抬升"。
3. 思考题：为什么 self-play 要加根噪声而评测不加？（提示：训练需要数据多样性来避免网络死记一条开局路线；评测要的是这个网络真实棋力的无偏估计。）
