"""Deterministic ProteinMPNN seeds for minibatch sampling.

When no base seed is given, diverse sampling keeps the historical formula
`n * 12345 + 42`. A provided base seed shifts every sequence so two
minibatches of the same backbone and temperature do not repeat sequences.
"""
from __future__ import annotations


def diverse_mpnn_seed(base_seed: int | None, index: int) -> int:
    if base_seed is None:
        return index * 12345 + 42
    return int(base_seed) + int(index)
