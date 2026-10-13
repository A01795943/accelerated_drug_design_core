"""Mide un intento de ProteinMPNN y comprueba la semilla. No cambia el pipeline.

Hay que ejecutarlo dentro del contenedor del core, con ColabDesign y GPU:

  python scripts/measure_mpnn_attempts.py --pdb /workspace/outputs/backbone.pdb --contig "20-35/0 A19-127" --temperature 0.4

Imprime la media de segundos de K intentos (default 30) del mismo paso que el
bucle de generación: mk_mpnn_model(seed), get_af_inputs, sample(num=1, batch=1).
No llama a AlphaFold predict.
"""
from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pipeline"))


def _build_af(pdb: str, contig: str):
    from mpnn_diverse_af import alphabet_list, get_info
    from colabdesign.af import mk_af_model

    contigs = []
    for contig_str in contig.replace(" ", ":").replace(",", ":").split(":"):
        if not contig_str:
            continue
        parts = [x for x in contig_str.split("/") if x != "0"]
        if parts:
            contigs.append("/".join(parts))
    chains = alphabet_list[: len(contigs)]
    info = [get_info(x) for x in contigs]
    fixed_pos = []
    fixed_chains = []
    free_chains = []
    both_chains = []
    for pos, (fixed_chain, free_chain) in info:
        fixed_pos += pos
        fixed_chains.append(fixed_chain and not free_chain)
        free_chains.append(free_chain and not fixed_chain)
        both_chains.append(fixed_chain and free_chain)
    flags = {
        "initial_guess": False,
        "best_metric": "rmsd",
        "use_multimer": True,
        "model_names": ["model_1_multimer_v3"],
    }
    if sum(both_chains) == 0 and sum(fixed_chains) > 0 and sum(free_chains) > 0:
        target = [chains[n] for n, flag in enumerate(fixed_chains) if flag]
        binder = [chains[n] for n, flag in enumerate(fixed_chains) if not flag]
        af_model = mk_af_model(protocol="binder", **flags)
        prep = {"target_chain": ",".join(target), "binder_chain": ",".join(binder), "rm_aa": "C"}
    else:
        af_model = mk_af_model(protocol="fixbb", **flags)
        prep = {"chain": ",".join(chains), "copies": 1, "homooligomer": False, "rm_aa": "C"}
    af_model.prep_inputs(pdb, **prep)
    return af_model


def _one(af_model, seed: int, temperature: float) -> tuple[float, str]:
    from colabdesign.mpnn import mk_mpnn_model

    started = time.perf_counter()
    model = mk_mpnn_model(seed=int(seed))
    model.get_af_inputs(af_model)
    out = model.sample(num=1, batch=1, temperature=float(temperature))
    seq = out["seq"][0]
    seq = seq if hasattr(seq, "strip") else str(seq)
    return time.perf_counter() - started, str(seq)


def _probe(af_model, seed: int) -> str:
    try:
        _elapsed, seq = _one(af_model, seed, 1.0)
    except Exception as exc:
        return f"seed={seed} ERROR {type(exc).__name__}: {exc}"
    return f"seed={seed} OK len={len(seq)} seq={seq[:80]}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdb", required=True)
    parser.add_argument("--contig", required=True)
    parser.add_argument("--temperature", type=float, required=True)
    parser.add_argument("--attempts", type=int, default=30)
    args = parser.parse_args()
    if args.attempts < 1:
        raise SystemExit("--attempts debe ser >= 1")

    af_model = _build_af(args.pdb, args.contig)
    times = []
    for n in range(args.attempts):
        elapsed, _seq = _one(af_model, n + 1, args.temperature)
        times.append(elapsed)
        print(f"attempt={n} seed={n + 1} seconds={elapsed:.3f}")
    print(f"mean_seconds={sum(times) / len(times):.3f} attempts={len(times)} temperature={args.temperature}")

    print("--- check (a) T=1.0 seeds s and s+1 ---")
    _, seq_s = _one(af_model, 1000, 1.0)
    _, seq_s1 = _one(af_model, 1001, 1.0)
    print(f"distinct={seq_s != seq_s1}")
    print(f"s={seq_s}")
    print(f"s+1={seq_s1}")

    print("--- check (b) same seed twice at T=1.0 ---")
    _, first = _one(af_model, 1000, 1.0)
    _, second = _one(af_model, 1000, 1.0)
    print(f"identical={first == second}")

    print("--- check (c) large seeds at T=1.0 ---")
    reference = _probe(af_model, 5)
    print("reference", reference)
    for seed in (2**31 + 5, 2**32 + 5, 2**33):
        try:
            print(_probe(af_model, seed))
        except Exception:
            traceback.print_exc()


if __name__ == "__main__":
    main()
