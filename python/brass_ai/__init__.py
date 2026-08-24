"""Brass: Birmingham AI layer (Python).

Phase-3 training stack: a PyTorch Policy-Value network guiding Rust ISMCTS
through the `brass_ai._engine` bindings. Python owns data generation, training and
evaluation; Rust owns game rules, information-set search and tensor encoding.
"""
