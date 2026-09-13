# Brass Assistant

An AI strategy analysis and simulation project for the board game *Brass: Birmingham*.

## Project Status: Indefinitely On Hold

Due to computational budget constraints and exploration bottlenecks in multi-agent self-play, the neural network training (AlphaZero-style Policy-Value network) **did not achieve meaningful or positive results**, and the policy consistently struggled to scale scores effectively. Active development on this project is therefore **indefinitely suspended** until more viable solutions or sufficient compute resources become available.

### ⚠️ Important Notice for Future Use
The latest commit contains an experimental ablation change that hardcodes a fixed merchant configuration in `engine/src/game_state/state.rs` (the `init_merchants` method), bypassing the standard random tile shuffle for 2–4 player setups.

**If you intend to use, fork, or build upon this project for standard gameplay or AI research, be sure to REVERT this latest commit (or restore the randomized merchant setup in `init_merchants`).**

---

## What Can Be Used: Game Simulation Engine

While the neural network exploration did not reach expected performance, the repository provides a complete, high-performance simulation engine that can be used for reference, benchmarking, or research:

- **Rust Simulation Engine (`engine/`)**:
  - High-speed, rule-complete implementation of *Brass: Birmingham* (2–4 players, Canal & Rail eras, market mechanics, loan systems, and full network connectivity).
  - Fast legal action generator, graph-based resource routing (coal, iron, brewery beer), and game-state determinization.
  - High-throughput rule-based heuristic agents capable of running ~70 games/second in pure Rust.
- **PyO3 Rust ↔ Python Bridge**:
  - Tensor encoding, state tokenization, and vectorized game environment interfaces connecting the Rust engine to PyTorch.
- **Python Module (`python/`)**:
  - Transformer-based Policy-Value network architectures, vectorized rollout environments, MCTS adapters, and diagnostic benchmarking suites.

---

## Compliance & Disclaimer

This repository is strictly for academic and personal research purposes. Game mechanics and rule implementations are independently authored. All intellectual property, trademarks, and copyrights for *Brass: Birmingham* belong to Roxley Games and the original designers (Martin Wallace, Gavan Brown, and Matt Tolman).
