# 完全移除原生 MCTS，保留并解耦 NN-MCTS 的 determinize

  ## Summary

  删除原生 mcts_ai 搜索器及其所有运行入口、配置、CLI、回放证据、测试和活动文
  档引用；保留 nn_mcts.rs、Python RustISMCTS 和 heuristic AI。

  由于 NN-MCTS 仍依赖隐藏手牌采样，将 determinize 提取为独立共享模块，保持
  Python GameState.determinize() 行为不变。

  ## Implementation Changes

  ### 1. 提取共享 determinize

  - 新建 engine/src/ai/determinize.rs。
  - 移入 mcts_ai.rs 中的：
      - is_wild
      - determinize
      - 牌池一致性逻辑及相关注释。

  - 对外提供稳定函数：

    pub fn determinize(
        state: &GameState,
        rng: &mut ChaCha12Rng,
    ) -> GameState

  - 在 ai/mod.rs 注册并导出 determinize 模块。
  - 更新 nn_mcts.rs 的两处调用，改为使用新模块。
  - 更新 bridge/pymod.rs 的 Python GameState.determinize() 实现。
  - train_bench.rs 改为直接基准测试新函数，不再依赖 MctsConfig 或
    determinize_for_test。

  - 将 engine/tests/engine_tests.rs 中的 determinize 测试改为调用新函数；删
    除旧的 MctsConfig 测试包装器。

  - 更新 game_state/state.rs、PyO3 注释和其他源码注释，不再声称 determinize
    属于 mcts_ai。

  ### 2. 删除原生 MCTS 搜索

  - 删除 engine/src/ai/mcts_ai.rs。
  - 从 engine/src/ai/mod.rs 和 engine/src/lib.rs 移除 mcts_ai 模块及其公开导
    出。

  - 删除原生 MCTS 专用类型和接口：
      - MctsConfig
      - LeafEval
      - choose_action_mcts
      - search_with_root_stats
      - RootChildStat
      - 原生 MCTS 的 MoveKey、PUCT、MaxN 搜索树和 evaluate_position 调用链。

  - 删除 engine/src/bin/mcts_lab.rs，不改造成 NN-MCTS 工具。
  - 保留 heuristic AI 的 choose_action、候选动作评分和所有相关模块。

  ### 3. 清理原生 MCTS 运行入口

  - engine/src/main.rs
      - 删除 mcts、mcts-vs-heur、mcts-vs-random 策略分支。
      - 删除 MctsConfig、simulation 参数和 MCTS 专用统计字段/输出。
      - 保留 heuristic/random 运行路径。

  - engine/src/ai/replay.rs
      - 删除 StrategySpec::Mcts。
      - 删除 MctsConfig 导入、原生 MCTS 分支、trace_for_mcts 和 root-search
        evidence。

      - StrategySpec::parse 移除 simulation 参数。
      - 保留 heuristic/random/Python worker 策略。

  - engine/src/bin/replay.rs
      - 删除原生 mcts、mcts-vs-heur、mcts-vs-random 模式。
      - 删除原生 MCTS 使用的 positional sims 参数。

  - engine/src/bin/replay_web.rs
      - 删除用于原生 MCTS 的顶层 --sims 参数。
      - 保留 Python worker 配置字符串内部的 --sims，因为那属于 NN-MCTS。

  - engine/src/bin/sweep_scores.rs
      - 删除 mcts policy、MCTS simulation 参数和相关分支。
      - 保留 heuristic sweep。

  - 删除原生 MCTS 专用 Rust 测试：
      - MCTS 动作/Pass fallback 测试；
      - 原生 mcts_ai 路径测试。

  - 保留并迁移 determinize 的牌池一致性、手牌数量、弃牌排除测试。

  ### 4. 保持 NN-MCTS 和训练链路不变

  - nn_mcts.rs 的搜索、网络 policy/value 回调、终局 ranking backup 保持现
    状。

  - Python RustISMCTS、replay_worker --mode mcts、NN self-play 和训练循环保
    持现状。

  - python 中的 GameState.determinize() API 保持不变。
  - --mcts-shortlist 保留，因为它表示 NN-MCTS 是否使用 heuristic 候选
    shortlist，不是原生 MCTS。

  - Bootstrap imitation 继续使用 heuristic teacher，不引入原生 MCTS。

  ### 5. 更新活动文档

  更新活动文档中的架构和命令：

  - docs/architecture.md
  - docs/engine-tools.md
  - docs/replay-design.md
  - docs/ai-action-encoding.md
  - docs/roadmap.md
  - docs/tutorial/09-evolution.md

  更新内容包括：

  - 删除 mcts_ai.rs、mcts_lab、原生 MCTS policy 和 OnePly/RootTwoPly 描述。
  - 将原生 MCTS benchmark 命令替换为 NN-MCTS benchmark 或 heuristic 工具。

  docs/archived/ 按已确认的选择保留历史内容，不删除其中的旧原生 MCTS 记录。

  - 这是有意的破坏性清理：
      - _engine::mcts_ai 不再存在；
      - 原生 MCTS 的 Rust API、CLI 策略和 replay 策略不再兼容。

  - 新增共享模块接口 ai::determinize::determinize。
  - Python GameState.determinize()、NN-MCTS Python 接口和训练数据格式保持兼
    容。

  - 不提供原生 MCTS 的兼容包装器，避免留下死代码和误导性 API。

  ## Test Plan

  - Rust 默认构建与测试：

    cargo test -p brass-engine --all-targets

  - Python feature 构建检查：

    cargo check -p brass-engine --features python --all-targets

  - Python 回归：

    .\.venv\Scripts\python.exe -m pytest python/tests -q

  - 重点验证：
      - determinize 保留己方手牌；
      - 对手手牌数量不变；
      - 牌池多重集合一致；
      - 已弃牌不会重新进入对手手牌或牌堆；
      - NN-MCTS self-play 和 replay worker 仍可运行；
      - heuristic replay 仍可运行；
      - 删除 mcts_lab 后 Cargo 不再尝试构建该 binary；
      - replay-web 的 Python worker --sims 仍有效。

  - 最终静态检查：

    活动源码中不得再出现：
    mcts_ai、MctsConfig、LeafEval、choose_action_mcts、mcts_lab、
    StrategySpec::Mcts、native mcts

    允许保留：

    nn_mcts、RustISMCTS、Python worker mode=mcts、
    ai::determinize、归档文档中的历史描述

  ## Assumptions

  - “完全清理原生 MCTS”包含删除原生 MCTS 的搜索实现、CLI、回放策略、实验
    binary、统计证据和活动文档，但不删除 NN-MCTS。

  - determinize 属于 AI 搜索共享基础设施，放在 engine/src/ai/
    determinize.rs，而不是继续挂在某个具体搜索器下面。

  - 旧的原生 MCTS Rust API 不需要兼容迁移层。