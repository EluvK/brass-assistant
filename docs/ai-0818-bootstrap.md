python python/bootstrap_imitation.py --games 2000 --epochs 10 --workers 8 --min-
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

3. 不采用 hard negatives 作为正式方案
   少量 hard negatives 只能抽查 full-legal 分布，无法保证其余数百个未见动作不会获得异常高 logit；它不能从根本上消除训练/推理候选分布错位。除非作为临时诊断，不再把它列为正式演进步骤。

  4. 再评估 action feature v1
     对 benchmark 中网络高分但 heuristic 低分的动作做 feature collision 和错误案例分析，决定 action feature v2 优先补哪些资源来源和 Sell
     信息。

  5. Value 扩展
     在 held-out policy 指标稳定后，增加 win/rank auxiliary head。当前 VP value 足以启动搜索，但不够贴近“四人局拿第一”的最终目标。

  完成前两项后，才能有依据决定是否启动 train_mp.py 的最小 self-play smoke run。

--- 2026-08-22 full-legal 最终架构方向 ---

当前 `uint8` 压缩只解决 full-legal replay 的存储和 IPC 成本，属于过渡实验，不改变候选矩阵仍被完整持久化的事实。最终方案确定为：

```text
ReplaySample = 可恢复的 Rust GameState 快照 + teacher canonical action + value/econ target
训练时 = 恢复 GameState -> Rust 动态生成全部 legal candidates/features -> 网络训练
推理时 = 当前 GameState -> Rust 动态生成全部 legal candidates/features -> full-legal MCTS
```

候选特征是 `GameState + Move` 的确定性派生数据，不应作为 replay 的长期主数据。实现该方案前需要增加 Rust state snapshot/restore 契约；当前给网络的 board/hand tensor 不能保证无损恢复完整 GameState。训练阶段的候选生成成本可以通过 worker、micro-batch 和流式处理控制，推理阶段本来就必须生成合法动作，因此不会引入额外的候选语义步骤。

执行顺序固定为：先完成当前压缩实验验证，再实现 state snapshot/restore 和动态候选训练，最后恢复 `candidate_k=0` 做 full-legal benchmark；在此之前不启动大规模 self-play。


---  0819 修复 engine 一些问题后再次 bootstrap结果：
python python/bootstrap_imitation.py --games 2000 --epochs 10 --workers 8 --min-avg-vp 95 --min-vp 88 --max-attempts 20000 --ckpt checkpoints/bootstrap-test0819.pt  
device: cuda
[accepted imitation game] 2000/2000 elapsed: 390s |ETA:   0s                                       , samples: 250915  
generated 250915 imitation samples from 2000 heuristic games (390s)
[train] 10/10 elapsed: 275s |ETA:   0s                                                             trained (289s): policy=1.524 value=0.757 top1=57.4% top3=84.5% top5=95.7% type_top1=60.7% entropy=1.37 candidates=10.9/p95=13
[bench sims=60] 20/20 elapsed:  15s |ETA:   0s
MCTS(bootstrap net) vs heuristic: win_rate=0% (mcts_vp=33.3 vs heuristic_vp=106.1)
checkpoint saved: checkpoints/bootstrap-test0819.pt

--- 0822 修改动作空间以后，训练用的
python python/bootstrap_imitation.py --games 1000 --epochs 10 --workers 8 --min-avg-vp 90 --min-vp 78 --max-attempts 20000 --ckpt checkpoints/bootstrap-0822.pt
 policy=1.625 value=0.788 top1=55.3% top3=82.4% top5=94.8% type_top1=58.5% entropy=1.37 candidates=11.2/p95=13
[bench sims=60] 20/20 elapsed:  10s |ETA:   0s
MCTS(bootstrap net) vs heuristic: win_rate=5% (mcts_vp=82.7 vs heuristic_vp=102.2)
checkpoint saved: checkpoints/bootstrap-0822.pt

---  0822 候选分布错位分析与验证

今天沿 bootstrap -> policy network -> Rust NN-MCTS -> benchmark 的完整链路检查后，确认
0822 退化的首要原因不是 235 维 CARD 特征本身无法学习，而是训练候选集和推理候选集不一致：

- imitation 的 `heuristic_candidates()` 只保留 `candidate_actions_k(..., 4)` 的 teacher shortlist，平均约 11 个候选；policy loss 只在这个 shortlist 内归一化。
- NN-MCTS 原先在每个节点扩展完整 `legal_moves()`。初始局面实测约 626 个合法 concrete actions，其中绝大多数从未作为训练候选出现。
- 因此网络没有学会压低 shortlist 之外的合法动作。新增 CARD 语义后，未训练的 action-feature 组合更多，full-legal MCTS 更容易被随机偏高的候选 logit 和错误先验带偏。

为验证该结论，临时增加了 `candidate_k` 配置，让 NN-MCTS 也使用 `heuristic_ai::candidate_actions_k(..., 4)`。同样的 1000 局、10 epoch bootstrap 结果为：

```text
shortlist MCTS: mcts_vp=82.7, heuristic_vp=102.2, win_rate=5%
此前 full-legal MCTS: mcts_vp=32.9, heuristic_vp=97.9, win_rate=0%
```

该结果显著恢复了策略强度，证明 full-legal 推理中的未训练候选是主要故障来源。当前 `candidate_k` 改动属于诊断/临时基线，不应作为最终能力边界；它会让 MCTS 永久受 heuristic shortlist 限制，也可能错过 shortlist 之外的更优合法动作。

后续执行路径：

1. 保留当前 shortlist MCTS 作为可复现实验基线
2. 完成当前 `uint8` full-legal 过渡实验，记录体积、生成速度和内存边界。
3. 实现 Rust `GameState` snapshot/restore，改造 replay 为状态快照 + teacher action + value/econ target。
4. 训练加载时从快照动态生成全部 legal candidates，恢复 NN-MCTS 的 full `legal_moves()`，对比 shortlist MCTS 与 full-legal MCTS。
5. 在动态候选训练通过 held-out validation 和 full-legal benchmark 前，不启动大规模 self-play。
