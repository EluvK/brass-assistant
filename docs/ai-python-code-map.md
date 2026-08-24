# `python` Python 代码地图

## 阅读顺序

```text
bootstrap_imitation.py
  -> selfplay.py                 生成 heuristic imitation Sample
  -> hierarchical_policy.py       从 Rust 获取候选动作/teacher
  -> train.py                     组 batch、计算 loss、更新网络
  -> net.py                       对状态与候选动作评分
  -> rust_mcts.py + evaluate.py   搜索与结尾 benchmark
```

## 顶层入口

### `bootstrap_imitation.py`

1. 作用：生成或复用 heuristic imitation shard，训练 `PolicyValueNet`，保存可恢复 checkpoint，并以网络引导的 Rust MCTS 对 heuristic 做基准评测。
2. 主要函数：
   - `main()`：解析参数；生成/读取 `imitation-*.pkl`；逐 shard 训练；恢复或保存 checkpoint；可选统计 policy 指标；运行 benchmark。
   - `save_checkpoint()`（内部函数）：原子写入 checkpoint，避免中断后留下不完整文件。

## `brass_ai` 包

### `brass_ai/hierarchical_policy.py`

1. 作用：从 Rust 获得完整合法候选或 heuristic shortlist，校验动作特征 schema，并将不同长度的候选集 padding 为网络 batch。
2. 主要函数：
   - `_feature_width()`：检查 Rust action feature schema version，返回动作特征维度。
   - `encode_legal_candidates(state)`：返回全部合法动作的 canonical 字符串和 `(N, 235)` tensor。
   - `encode_teacher_candidates(state)`：返回 heuristic shortlist 的特征、分数、最终动作及其索引。
   - `compress_candidate_features(features)`：将以 0.25 为步长的特征无损压缩为 `uint8`。
   - `pad_candidate_features(rows, device)`：把变长候选行变为 `(B, max_N, D)` 和 boolean mask。

### `brass_ai/net.py`

1. 作用：定义 `PolicyValueNet`：编码状态和 Rust 候选动作，为每个候选产生 logit，并预测四玩家 value 与经济辅助目标。
2. 主要类/函数：
   - `NetConfig`：网络宽度和输入维度配置。
   - `PolicyValueNet.__init__()`：构建状态 encoder、共享 trunk、动作 encoder 和 policy/value/econ heads。
   - `PolicyValueNet.encode_state(batch)`：编码 board、links、全局信息和手牌为 state embedding。
   - `PolicyValueNet.forward(batch, action_features, candidate_mask)`：计算 masked candidate logits、类型 logits、value 和 econ。
   - `PolicyValueNet.policy_value(...)`：无梯度推理包装，供搜索调用。

### `brass_ai/rust_mcts.py`

1. 作用：把 Python 网络包装成 Rust `GameState.search_net()` 的批量推理 callback，并将 Rust 搜索结果转换为 Python 对象。
2. 主要类/函数：
   - `make_net_fn(net, device)`：建立 Rust 调用的函数，完成 numpy -> tensor -> 网络 -> numpy 的转换。
   - `RustMCTSConfig`：PUCT、深度、根噪声、批大小、候选集大小和设备配置。
   - `SearchResult`：最佳动作、candidate visit 数及 candidate 到 canonical action 的映射。
   - `RustISMCTS.__init__()`：配置网络设备和 callback。
   - `RustISMCTS.search(state, sims, add_root_noise)`：调用 Rust 搜索并返回 `SearchResult`。

### `brass_ai/selfplay.py`

1. 作用：定义 `Sample`，生成 MCTS self-play 或 heuristic imitation 数据；完整合法候选模式可通过 Rust snapshot 延迟物化候选集。
2. 主要类/函数：
   - `Sample`：一个决策点的状态、候选、policy、value、econ 及可选 snapshot/教师动作。
   - `materialize_sample(sample)` / `materialize_samples(samples)`：从 snapshot 恢复 Rust 状态并重建全合法候选与 one-hot target。
   - `SelfPlayConfig`：玩家数、模拟数、温度、最大步数与 seed。
   - `_candidate_policy(...)`：将 MCTS visit 对齐为 Rust 候选顺序上的 policy 分布。
   - `_sample_move(...)`：按 visit 分布和温度选择动作。
   - `play_game(...)`：同一 MCTS 控制四席的一局 self-play。
   - `play_game_with_roles(...)`：允许不同座位使用不同搜索角色并收集样本。
   - `generate_imitation_samples(...)`：并行生成并在内存中返回 heuristic imitation 样本，适合测试和小实验。
   - `generate_imitation_sample_shards(...)`：边生成边写 shard，供 bootstrap 使用。
   - `play_batch(...)`：顺序运行多局 self-play 并汇总结果。

### `brass_ai/train.py`

1. 作用：持有 optimizer/scheduler；组装变长候选 batch；计算 policy、value、动作类型、经济和 L2 损失；控制完整候选训练的 batch 内存。
2. 主要类/函数：
   - `TrainConfig`：训练超参数、设备、候选行预算和 snapshot materialize worker 数。
   - `Trainer.__init__()`：创建 AdamW、CosineAnnealingLR 并绑定网络。
   - `Trainer.train_on_samples(samples)`：按配置训练多 epoch，推进学习率并计算训练集指标。
   - `Trainer.train_one_epoch(samples, progress_label)`：训练一遍样本；snapshot 模式即时物化，并按候选行预算拆分 micro-batch。
   - `Trainer.train_steps(...)`：有放回随机采样的固定步训练 API，供后续训练入口复用。
   - `Trainer.state_dict()` / `load_state_dict()`：保存/恢复训练状态并检查 feature schema。
   - `compute_loss(...)`：计算各监督目标与正则项。
   - `train_on_batch(...)`：执行一批的前向、反向和参数更新。
   - `_to_batch(samples)`：堆叠状态、padding 候选和 policy。
   - `evaluate_policy(...)`：分批统计 top-k、类型命中率、熵与候选数。
   - `LoopConfig` / `run_loop(...)`：简化的单进程 self-play -> train 循环；当前没有顶层入口调用它。

### `brass_ai/evaluate.py`

1. 作用：运行完整对局，并评测网络引导 MCTS 与 Rust heuristic 的对战结果。
2. 主要函数：
   - `heuristic_policy(state)`：返回 Rust heuristic 动作。
   - `mcts_policy(mcts, sims)`：将 MCTS 包装为策略函数。
   - `play_game_with_policies(...)`：以指定的四席策略运行一局，并对失效动作做保底处理。
   - `benchmark_mcts_vs_heuristic(...)`：轮换 MCTS 座位和 seed，汇总胜率及 VP。
   - `benchmark_net_vs_heuristic(...)`：创建 Rust MCTS 后执行上述 benchmark。

### `brass_ai/mp_selfplay.py`

1. 作用：维护多进程 self-play worker，传递模型权重，收集纯 numpy 格式的 MCTS 样本；可支持历史模型 matchmaking。
2. 主要类/函数：
   - `_worker_fn(...)`：子进程主循环，构建网络/MCTS 并运行 self-play。
   - `_pack_samples(samples)` / `unpack_samples(packed)`：在 `Sample` 与可跨进程传输的 numpy 字典之间转换。
   - `SelfPlayPool.__init__()`：启动常驻 worker。
   - `SelfPlayPool.generate(...)`：广播权重、收集结果和进度。
   - `SelfPlayPool.close()`：通知 worker 退出并等待回收。

### `brass_ai/progress.py`

1. 作用：为生成、训练和 benchmark 打印带 ETA 的单行进度。
2. 主要类/函数：
   - `Progress.__init__(...)`：初始化进度状态。
   - `Progress.update(done, extra)`：按刷新频率输出进度与 ETA。
   - `Progress.done()`：输出完成状态并换行。
