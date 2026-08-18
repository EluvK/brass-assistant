python src/ai/bootstrap_imitation.py --games 2000 --epochs 10 --workers 8 --min-
  avg-vp 80 --min-vp 58 --max-attempts 20000 --ckpt checkpoints/bootstrap.pt
  device: cuda
  [accepted imitation game] 2000/2000 elapsed: 140s |ETA:   0s                                       , samples: 250898
  generated 250898 imitation samples from 2000 heuristic games (140s)
  [train] 10/10 elapsed: 298s |ETA:   0s                                                             trained (313s): policy=1.416
  value=0.553 top1=59.8% top3=87.0% top5=96.3% type_top1=51.9% entropy=1.24 candidates=9.3/p95=13
  [bench sims=60] 20/20 elapsed:  12s |ETA:   0s
  MCTS(bootstrap net) vs heuristic: win_rate=15% (mcts_vp=79.7 vs heuristic_vp=82.9)
  checkpoint saved: checkpoints/bootstrap.pt


• 这次 bootstrap 是一个有效的第一阶段基线，结果比 200 局小样本明显更好，但还不足以进入长期 self-play。

  吞吐与规模

  2,000 games
  250,898 samples
  140 s generation
  313 s training

  平均每局约 125.4 个样本，和小规模运行一致，说明 teacher shortlist 路径稳定。

  采样吞吐约为 1,792 samples/s 墙钟时间。8 个 worker 下没有再次出现候选 replay 内存爆炸，说明有界 shortlist 修复生效。

  训练部分约 10 个 epoch、250 万 sample-updates，总耗时约 298 秒，约 8.4k sample-updates/s。这已经可以作为当前 GPU candidate-policy
  baseline。

  训练指标变化

  和之前 200 局、3 epoch 的结果相比：

   指标           小样本     当前    含义
  ━━━━━━━━━━━━━  ━━━━━━━━  ━━━━━━━  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   policy loss     1.649    1.416    更接近 teacher soft ranking
  ─────────────  ────────  ───────  ─────────────────────────────────────
   value MSE       0.758    0.553    终局标准化 VP 预测更好
  ─────────────  ────────  ───────  ─────────────────────────────────────
   top-1           51.7%    59.8%    teacher 最佳 shortlist 动作命中提升
  ─────────────  ────────  ───────  ─────────────────────────────────────
   top-3           82.0%    87.0%    teacher 候选排序显著改善
  ─────────────  ────────  ───────  ─────────────────────────────────────
   top-5           94.3%    96.3%    基本覆盖 teacher 前部偏好
  ─────────────  ────────  ───────  ─────────────────────────────────────
   type top-1      39.4%    51.9%    粗粒度动作类型学习更充分
  ─────────────  ────────  ───────  ─────────────────────────────────────
   entropy          1.49     1.24    policy 更有明确偏好

  exp(1.416) ≈ 4.1，可以粗略理解为：网络的概率质量仍分散在约 4 个等效候选上，但已经比先前的约 5.2 个更集中。

  平均 shortlist 大小仍是 9.3，p95 为 13。因此：

  - top-5=96.3% 是正向信号，但不能过度解读，因为候选集并不大；
  - top-1=59.8% 和 top-3=87.0% 更值得观察；
  - type_top1=51.9% 低于完整 candidate top-1，说明网络确实从 card、位置、产业和资源特征中学到了额外区分能力，而非只学动作类型。

  这些仍是训练 replay 上的指标，不能代表泛化。下一步必须有 held-out teacher validation。

  MCTS 对 heuristic

  win_rate = 15%
  MCTS mean VP = 79.7
  heuristic mean VP = 82.9

  这是比小样本明显改善的结果：

  小样本：55.8 vs 75.9，差 20.1 VP
  当前：  79.7 vs 82.9，差  3.2 VP

  说明 policy/value 已经能给 MCTS 提供有用 priors，不再是随机网络。

  但 15% 胜率仍偏低。四人局中，若四席强度相当，单席理论胜率约为 25%。不过这里只有 20 局，15% 实际是 3 胜，样本方差很大，不能据此下强结论。

  VP 接近但胜率偏低也符合当前目标函数：

  - value 学的是 normalized final VP，不是 win probability；
  - policy 学的是 heuristic shortlist 排序，不是直接最大化胜率；
  - 四人局中，提升自身 VP 不一定提升第一名概率；
  - 当前 benchmark 的网络 MCTS 还要面对 full legal candidates，其中许多动作没有出现在 imitation shortlist 训练中。

  当前结论

  这份 checkpoint 已足够作为后续候选 policy 研究的 baseline，但不适合作为 self-play 主模型的唯一依据。最重要的缺口不是扩大训练规模，而是验
  证和分布对齐。

  下一步建议按这个顺序做：

  1. 建立固定 held-out teacher validation
     独立 seeds 生成 teacher 样本，不参与训练。报告 loss、top-1/3/5、type accuracy、target entropy，避免把训练集 59.8% 当作泛化能力。

  2. 扩大 benchmark 样本
     用至少 100 局，最好 200 局，分别测试 60 和 200 simulations。20 局只能发现灾难性问题，不能可靠区分 15%、20% 或 25% 胜率。

  3. 加入 hard negatives
     当前 imitation 只看到高质量 teacher shortlist，而 MCTS 必须在完整合法动作中决策。应从全量 legal actions 中抽取少量结构不同的低分候
     选，加入每条训练样本，控制总候选数在 16-32，而不是重新持久化全量动作。

  4. 再评估 action feature v1
     对 benchmark 中网络高分但 heuristic 低分的动作做 feature collision 和错误案例分析，决定 action feature v2 优先补哪些资源来源和 Sell
     信息。

  5. Value 扩展
     在 held-out policy 指标稳定后，增加 win/rank auxiliary head。当前 VP value 足以启动搜索，但不够贴近“四人局拿第一”的最终目标。

  完成前两项后，才能有依据决定是否启动 train_mp.py 的最小 self-play smoke run。