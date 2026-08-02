"""Brass: Birmingham AI layer (Python).

Phase-3 research stack: a Policy-Value network (PyTorch) guiding an ISMCTS
search over the Rust game engine (`brass_engine` bindings). The MCTS runs in
Python for fast iteration; the engine (legal moves / step / tensor encoding /
determinization) stays in Rust.
"""
