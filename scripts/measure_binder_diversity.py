"""Cuenta binders distintos y muestra las semillas que usaría un minibatch.

El muestreo diverso de este repo (pipeline/mpnn_diverse_af.py) llama a
ProteinMPNN solo después de preparar AlphaFold (`get_af_inputs`). No hay una
función que devuelva secuencias sin cargar AlphaFold. Este script mide un CSV
ya producido y, aparte, imprime las semillas de la fórmula actual.

  python scripts/measure_binder_diversity.py --csv mpnn_results.csv
  python scripts/measure_binder_diversity.py --seeds --job-id 17 --n-offset 0 --num-seqs 100
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pipeline"))

from mpnn_seed import binder_key, diverse_mpnn_seed  # noqa: E402


def seed_for(job_id: int, n_offset: int) -> int:
    """Same mix as MpnnSampling.seedFor."""
    def splitmix64(z: int) -> int:
        mask = (1 << 64) - 1
        z &= mask
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & mask
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & mask
        return (z ^ (z >> 31)) & mask

    mixed = splitmix64(job_id)
    offset = max(n_offset, 0)
    mixed = splitmix64(mixed ^ ((offset * 0x9E3779B97F4A7C15) & ((1 << 64) - 1)))
    seed = mixed & 0x7FFFFFFF
    return 1 if seed == 0 else seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--seeds", action="store_true")
    parser.add_argument("--job-id", type=int)
    parser.add_argument("--n-offset", type=int, default=0)
    parser.add_argument("--num-seqs", type=int, default=100)
    args = parser.parse_args()

    if args.csv:
        import pandas as pd

        frame = pd.read_csv(args.csv)
        if "seq" not in frame.columns:
            raise SystemExit(f"{args.csv} no tiene columna seq")
        keys = [binder_key(s) for s in frame["seq"].astype(str)]
        keys = [k for k in keys if k]
        print(f"rows={len(frame)} with_binder={len(keys)} unique_binders={len(set(keys))}")

    if args.seeds:
        if args.job_id is None:
            raise SystemExit("--seeds requiere --job-id")
        base = seed_for(args.job_id, args.n_offset)
        values = [diverse_mpnn_seed(base, n) for n in range(args.num_seqs)]
        print(f"base={base} first={values[0]} last={values[-1]} unique={len(set(values))}")
        return

    if not args.csv:
        raise SystemExit("indica --csv y/o --seeds")


if __name__ == "__main__":
    main()
