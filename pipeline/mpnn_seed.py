"""Binder identity and deterministic ProteinMPNN seeds.

Stored sequences are TARGET/BINDER. The binder is the tail after the last
'/'. ``binder_key_for_length`` is the same key, except when that tail is not
the length AlphaFold folds: then it uses the tail of the slash-stripped
sequence, which is what ``seq.replace('/', '')[-af_len:]`` predicts.
"""
from __future__ import annotations

def diverse_mpnn_seed(base_seed: int | None, index: int) -> int:
    if base_seed is None:
        return index * 12345 + 42
    return int(base_seed) + int(index)


def _normalize(seq: str) -> str:
    return "".join(str(seq).split()).upper()


def binder_key(seq: str | None) -> str:
    """Binder chain: text after the last '/', without whitespace, uppercased."""
    if seq is None:
        return ""
    text = _normalize(seq)
    if not text:
        return ""
    if "/" not in text:
        return text
    tail = text.rsplit("/", 1)[-1]
    return tail if tail else text.replace("/", "")


def binder_key_for_length(seq: str | None, expected_len: int | None) -> str:
    """``binder_key``, or the folded tail when that key is not ``expected_len``."""
    key = binder_key(seq)
    if expected_len is None or expected_len <= 0 or len(key) == expected_len:
        return key
    if seq is None:
        return key
    flat = _normalize(seq).replace("/", "")
    if len(flat) >= expected_len:
        return flat[-expected_len:]
    return key


def load_binder_keys(path: str | None) -> set[str]:
    if not path:
        return set()
    keys: set[str] = set()
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            key = binder_key(line)
            if key:
                keys.add(key)
    return keys
