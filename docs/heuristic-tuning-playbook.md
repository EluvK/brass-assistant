# Heuristic 调参方法论（Playbook）

> 本文档沉淀 2026-08 期间对 `heuristic_ai` / `search_ai`（2ply）做强度调优时使用到的
> **工具链、分析套路、判断准则**。目标是让后续进入本仓库的 agent 可以照做，而不是重新摸索。
> 结论性基线见 `docs/current-baseline.md`；当时的代码快照与逐轮决策参考 `git log`。

---

## 1. 工具链速查（都在 `src/engine` 下）

| 工具 | 命令 | 用途 |
| --- | --- | --- |
| 正确性 | `cargo test --release` | 3 unit + 27 integration，改动后必跑 |
| 单局统计 | `cargo run --release --bin stat_game -- <seed> <players> <policy>` | 时代末汇总、动作统计、翻面列表，快速定位病理 |
| 单局回放 | `cargo run --release --bin replay -- <seed> <players> <policy>` | 逐步日志（手牌/盘面/商家/决策），深挖细节 |
| 批量强度 | `cargo run --release --bin brass-engine -- 500 4 <policy>` | 500 局均值：VP/人、built/flipped/links、动作分布、收入、胜率 |
| MCTS 诊断 | `cargo run --release --bin bench_mcts / sweep_mcts / debug_mcts` | 可选，MCTS 相关 |

`policy` 取值：`heuristic`（默认）、`2ply`、`mcts`、`random`、`mcts-vs-heur` 等。

---

## 2. 核心分析循环（重要）

一次"找病根 → 修复 → 验证"的标准循环：

1. **导出全种子分数**，找出 bottom 局（见 §3 脚本）。
2. 对选中的 seed 跑 `stat_game`，看：
   - 时代末收入 / VP 是否异常（负收入是崩盘信号）；
   - 动作分布是否失衡（过牌爆炸 / 建网过多 / 研发过多 / 0 卖出）。
3. 跑 `replay`，看关键转折点的手牌、现金、商家桶、牌堆。
4. **插桩看真实决策值**（见 §4），确认"为什么 AI 选了这一步"——不要靠猜。
5. 定位评估函数的量化错误 → 修复（改启发式即可，不动引擎架构）。
6. 验证三连：
   - 目标 seed 的 `stat_game` 分数是否改善；
   - `brass-engine -- 500 4 2ply` 整体均值 / bottom-20 / pass 是否改善；
   - 重新导出 500 种子对比 bottom-20 / top-20 分布。
   - **同时必须确认"没有把别处打坏"**：改某类动作的权重，常见副作用是另一类动作飙升（如压研发 → 建网暴涨）。

---

## 3. 全种子分数导出脚本（核心工具）

用 `replay` 串行跑 500 局并解析终局 VP，导出 CSV，同时打印均值与 bottom-20。

```python
import subprocess,re
from pathlib import Path
exe=Path('target/release/replay.exe')
out=Path('seed_scores_2ply_0_499.csv')
rows=[]; vp_re=re.compile(r'玩家(\d+): .*?VP(\d+)')
for seed in range(500):
    p=subprocess.run([str(exe),str(seed),'4','2ply'],capture_output=True,text=True,
                      encoding='utf-8',errors='ignore')
    ms=vp_re.findall(p.stdout); vals=[0,0,0,0]
    if len(ms)>=4:
        for pid,v in ms[-4:]: vals[int(pid)]=int(v)
    avg=sum(vals)/4; rows.append((seed,*vals,avg))
with out.open('w',encoding='utf-8') as f:
    f.write('seed,p0,p1,p2,p3,avg\n')
    for r in rows: f.write(f'{r[0]},{r[1]},{r[2]},{r[3]},{r[4]},{r[5]:.2f}\n')
avgs=[r[5] for r in rows]
print('table mean',round(sum(avgs)/len(avgs),2))
print('games with any player <40:', len([r for r in rows if any(v<40 for v in r[1:5])]))
print('games with any player <15:', len([r for r in rows if any(v<15 for v in r[1:5])]))
s=sorted(rows,key=lambda r:r[5],reverse=True)
print('bottom20 avg', round(sum(r[5] for r in s[-20:])/20,2))
for r in s[-20:]: print(f'seed={r[0]:3d} avg={r[5]:6.2f} vps={r[1:5]}')
```

> 注意：500 局串行约 10 分钟。跑完后 CSV 是后续 A/B 对比的基准，改动前后各导一份，
> 对比 bottom-20 / top-20 / 均值。

**快速 top 种子清单**（给用户挑高分局回放看）：

```python
# 接在上面的 rows 之后
s=sorted(rows,key=lambda r:r[5],reverse=True)
for r in s[:10]: print(f'seed={r[0]:3d} avg={r[5]:6.2f} vps={r[1:5]}')
```

---

## 4. 插桩看真实决策值（避免猜测）

`search_ai::choose_action_2ply` 决策基于 `c1.score + ALPHA*best_second + end_of_turn_penalty`，
只看日志猜不出"为什么选它"。临时加一行 `eprintln!`（环境变量门控），跑一次再删：

```rust
// 在 search_ai.rs 的 choose_action_2ply 循环里，value 计算后加：
if std::env::var("BRASS_DEBUG_CANDS").is_ok() {
    eprintln!("[cand] era={:?} round={} pid={} c1={} s={:.2} v={:.2}",
              state.era, state.round, pid, c1.mv.describe(state), c1.score, value);
}
// 选完后再加一条：
if std::env::var("BRASS_DEBUG_CANDS").is_ok() {
    eprintln!("[pick] era={:?} round={} pid={} chosen={:?} v={:?}",
              state.era, state.round, pid, best.as_ref().map(|b| b.0.describe(state)),
              best.as_ref().map(|b| b.1));
}
```

```bash
BRASS_DEBUG_CANDS=1 ./target/release/stat_game.exe 349 4 2ply 2>cands.txt
grep "\[pick\]" cands.txt | grep -E "pid=0"
```

关键收益：能直接看到某类动作的**真实分数**（如"运河期建 Lv2 制造厂被估 33 分"），
从而确认是权重膨胀还是别的因素，而不是靠推断。用完整删。

---

## 5. 本次调优中确认的判断准则（经验库）

以下是被数据验证过的"评估函数错误模式"，后续遇到同类症状可对号入座：

### 5.1 崩盘症状 → 根因映射
| 症状（stat_game / replay 可见） | 根因（评估函数） |
| --- | --- |
| 全桌债务螺旋、铁路全过牌 | 2ply 只看本回合 combo，不管"下回合有没有钱/可行动作" → 加回合末流动性惩罚 |
| 铁路期狂建网、煤市见底仍不补煤 | 铁路建网只按 £5 计费，没把 1 煤成本算进去 → 建网显式计入最便宜煤源成本 |
| 高价低级板块/制造厂泡沫（单动作估 15~33 分） | 运河 Lv≥2 无条件 ×2.0，叠 double_vp 1.1 → 乘数改为按翻面率分级 |
| 可卖板块叠一堆卖不掉（陶器 xN） | 无啤酒容量概念、商家无桶也给翻面加分 → 商家加分必须有酒 + 新增啤酒饱和度 |
| 无生产还狂铺网（8-14 条链接 0 VP） | 建网惩罚只 1.0 且要求"完全无板块" → 按"链接数 vs 生产板块数"分级惩罚 |
| 研发次数失控 | 研发是行动机会成本 → 行动数护栏（运河≤4 次、铁路≤1 次，超出陡峭惩罚） |

### 5.2 具体修复点（本次落地，文件 `src/engine/src/heuristic_ai.rs` / `search_ai.rs`）
- `search_ai.rs`：
  - `end_of_turn_penalty`：回合末现金 <£15 且收入无明显回升（Δ<+2.5 级）且跑道不足 → 惩罚，抑制"爽一回合瘫一整轮"。
  - `combo_alpha`：铁路后半 ALPHA 0.6→0.35，抑制贷款→建网类 combo 盖过 heuristic 自身债务风控。
- `heuristic_ai.rs`：
  - `score_build_candidate` 运河 Lv≥2 乘数按 `flip_prob` 分级（≥0.6 →×2.0，≥0.35 →×1.3，否则 ×1.0）。
  - `estimate_flip_probability`：可卖板块底值 0.35→0.12；商家加分必须有酒（无酒只 +0.1）；酒厂翻面按啤酒需求缩放；新增 `sell_saturation`（剩余啤酒容量，新板块排在已有待卖板块之后）。
  - `score_network_candidate`：建网惩罚按 `max(0, 链接数 - 2×生产板块 - 1)` 缩放，无板块时前 1 条免费之后每条 ×1.2。
  - `score_develop_plan`：研发行动数护栏（`develops_in_canal/rail` 计数，运河超 4 次、铁路超 1 次开始陡峭惩罚）；免费研发不计入。

### 5.3 需要避开的坑
- **不要全局压制某类动作**：曾给"无场上板块的研发"加惩罚，导致早期"先研发后建"被误伤，
  整体均值 73.5 → 59.1（建网飙到 51.2/局）。研发护栏应基于**行动次数**而非"有没有板块"。
- **参考实现与规则文档有分歧时不要自作主张**：如"研发是否需要场上板块"参考实现允许轨道推进、
  规则书要求移除场上板块，需列明双方依据交用户裁决（见 AGENTS.md 规则 7）。
- **改动必须跑整体基线**：单 seed 改善可能是噪声；均值、bottom-20、pass、income 一起看，
  并确认没有让另一类动作失衡。
- **插桩用完即删**：环境变量门控的调试代码不要留在提交里。

---

## 6. 本次调优的量化收益（2ply 500 局，累计）

| 阶段 | 改动 | 均值 | bottom-20 均值 | <40 分选手的局数 |
| --- | --- | --- | --- | --- |
| 起点 | — | 58.6 | ~21 | — |
| +2ply 流动性惩罚 / 末期 ALPHA | 60.1 | 25.9 | — |
| +运河乘数按翻面分级 / 翻面底值 | 68.1 | 33.9 | 130 |
| +啤酒饱和度 / 商家需酒 | 70.9 | 37.7 | 130 |
| +建网防滥铺 | 73.5 | 40.5 | 90 |
| +研发行动护栏 | 73.4 | 40.5 | 92 |

`heuristic` 同批从 ~46.6 提升到 ~60.8。均值与 bottom 同时改善，过牌大幅下降（2ply 7.7→2.5）。

---

## 7. 残余问题与下一步候选
- 仍有 ~90/500 局存在至少一个 <40 分选手（~44 局 <15 分），多为"早期不可逆崩盘"：
  1 个卖不掉的板块 + 大量建网/贷款进入铁路后空转。
- 单靠 heuristic 评分边际收益递减；更有效方向是换更强搜索（MCTS，见 `docs/reference-notes/mcts-stage3.md`）
  或降低崩盘方差（手牌/开局调度）。
