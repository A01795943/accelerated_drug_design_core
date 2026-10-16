"""Mide por separado los pasos de ProteinMPNN. No cambia el pipeline.

Hay que ejecutarlo dentro del contenedor del core, con ColabDesign y GPU:

  python scripts/measure_mpnn_steps.py --pdb /workspace/outputs/backbone.pdb --contigs "20-35/0 A19-127"

La preparación del modelo AlphaFold es la de scripts/measure_mpnn_attempts.py
(_build_af). No llama a AlphaFold predict. El progreso va a stderr; la tabla
copiable va a stdout.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pipeline"))

BATCH_SIZES = (1, 8, 32, 64, 100, 200)


def _stats(samples: list[float]) -> tuple[float, float, float]:
    return sum(samples) / len(samples), min(samples), max(samples)


def _touch(out) -> None:
    seq = out["seq"][0]
    if hasattr(seq, "block_until_ready"):
        seq.block_until_ready()


def _mib(num_bytes: float | int | None, *, scale: float) -> float | None:
    if num_bytes is None:
        return None
    return float(num_bytes) / scale


def _gpu_memory_mib() -> tuple[float | None, float | None]:
    """Uso actual y pico del device JAX, en MiB. El pico es del proceso entero."""
    try:
        import jax

        devices = jax.devices()
        if not devices or not hasattr(devices[0], "memory_stats"):
            return None, None
        stats = devices[0].memory_stats() or {}
    except Exception:
        return None, None
    return (
        _mib(stats.get("bytes_in_use"), scale=1024**2),
        _mib(stats.get("peak_bytes_in_use"), scale=1024**2),
    )


def _nvml_used_mib() -> float | None:
    try:
        import pynvml

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        return _mib(pynvml.nvmlDeviceGetMemoryInfo(handle).used, scale=1024**2)
    except Exception:
        return None


def _cpu_maxrss_mib() -> float | None:
    """Máximo RSS del proceso. En Linux getrusage lo da en KiB y no baja."""
    try:
        import resource
    except ImportError:
        return None
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return _mib(rss, scale=1024**2)
    return _mib(rss, scale=1024)


def _gpu_names() -> str:
    try:
        import pynvml

        pynvml.nvmlInit()
        names = []
        for index in range(pynvml.nvmlDeviceGetCount()):
            handle = pynvml.nvmlDeviceGetHandleByIndex(index)
            name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode()
            names.append(str(name))
        if names:
            return ", ".join(names)
    except Exception:
        pass
    try:
        import jax

        described = [str(device) for device in jax.devices()]
        return ", ".join(described) if described else "none"
    except Exception as exc:
        return f"unavailable ({type(exc).__name__}: {exc})"


def _colabdesign_version() -> str:
    try:
        return importlib.metadata.version("colabdesign")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def _cell(value) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value).replace("\n", " ")


def _print_table(headers: list[str], rows: list[list]) -> None:
    rendered = [[_cell(value) for value in row] for row in rows]
    widths = [len(header) for header in headers]
    for row in rendered:
        for index, value in enumerate(row):
            widths[index] = max(widths[index], len(value))

    def fmt(row: list[str]) -> str:
        return "  ".join(value.ljust(widths[index]) for index, value in enumerate(row))

    print(fmt(headers), flush=True)
    print(fmt(["-" * width for width in widths]), flush=True)
    for row in rendered:
        print(fmt(row), flush=True)


def _time_call(fn) -> tuple[float, object]:
    started = time.perf_counter()
    value = fn()
    return time.perf_counter() - started, value


def _build_af(pdb: str, contig: str, rm_aa: str | None):
    """Igual que measure_mpnn_attempts._build_af, con rm_aa configurable.

    Devuelve el modelo y los segundos de mk_af_model + prep_inputs.
    """
    from colabdesign.af import mk_af_model
    from mpnn_diverse_af import alphabet_list, get_info

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
        protocol = "binder"
        prep = {
            "target_chain": ",".join(target),
            "binder_chain": ",".join(binder),
            "rm_aa": rm_aa,
        }
    else:
        protocol = "fixbb"
        prep = {
            "chain": ",".join(chains),
            "copies": 1,
            "homooligomer": False,
            "rm_aa": rm_aa,
        }

    def _make():
        model = mk_af_model(protocol=protocol, **flags)
        model.prep_inputs(pdb, **prep)
        return model

    elapsed, model = _time_call(_make)
    return model, elapsed


def _measure_reps(reps: int, label: str, fn) -> list[float]:
    samples: list[float] = []
    for index in range(reps):
        print(f"{label} {index + 1}/{reps}", file=sys.stderr, flush=True)
        elapsed, _value = _time_call(fn)
        samples.append(elapsed)
    return samples


def _run_timed(fn) -> tuple[float | None, str]:
    try:
        elapsed, _value = _time_call(fn)
    except Exception as exc:
        message = f"{type(exc).__name__}: {exc}".replace("\n", " ")
        print(f"error: {message}", file=sys.stderr, flush=True)
        return None, message
    return elapsed, ""


def main() -> None:
    parser = argparse.ArgumentParser(description="Mide pasos de ProteinMPNN por separado.")
    parser.add_argument("--pdb", required=True)
    parser.add_argument("--contigs", required=True, help="Mismo texto de contigs que el pipeline")
    parser.add_argument("--rm_aa", default="C", help="Aminoácidos prohibidos; cadena vacía no prohíbe ninguno")
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--reps", type=int, default=30)
    args = parser.parse_args()
    if args.reps < 1:
        raise SystemExit("--reps debe ser >= 1")
    rm_aa = args.rm_aa if args.rm_aa != "" else None

    print("import jax", file=sys.stderr, flush=True)
    jax_seconds, _jax = _time_call(lambda: __import__("jax"))
    print("import colabdesign", file=sys.stderr, flush=True)
    colab_seconds, _colab = _time_call(
        lambda: (
            __import__("colabdesign.af", fromlist=["mk_af_model"]),
            __import__("colabdesign.mpnn", fromlist=["mk_mpnn_model"]),
        )
    )

    print("## hardware", flush=True)
    _print_table(
        ["key", "value"],
        [
            ["cpu_cores", os.cpu_count()],
            ["gpu", _gpu_names()],
            ["colabdesign", _colabdesign_version()],
        ],
    )
    print(flush=True)
    print("gpu_peak_mib y cpu_maxrss_mib son máximos del proceso: no bajan entre filas.", flush=True)
    print("gpu_in_use_mib es el uso JAX justo después de la llamada (o NVML si JAX no informa).", flush=True)
    print(flush=True)

    print("mk_af_model + prep_inputs", file=sys.stderr, flush=True)
    af_model, prep_seconds = _build_af(args.pdb, args.contigs, rm_aa)

    print("## startup_once", flush=True)
    _print_table(
        ["what", "seconds"],
        [
            ["import_jax", jax_seconds],
            ["import_colabdesign", colab_seconds],
            ["mk_af_model_plus_prep_inputs", prep_seconds],
        ],
    )
    print(flush=True)

    from colabdesign.mpnn import mk_mpnn_model

    print("## steps", flush=True)
    step_rows: list[list] = []

    model = None

    def _new_model(seed_box: list[int]):
        nonlocal model
        seed_box[0] += 1
        model = mk_mpnn_model(seed=seed_box[0])

    seed_box = [0]
    create_times = _measure_reps(args.reps, "mk_mpnn_model", lambda: _new_model(seed_box))
    mean_s, min_s, max_s = _stats(create_times)
    step_rows.append(["1 mk_mpnn_model", args.reps, mean_s, min_s, max_s, ""])

    input_times = _measure_reps(
        args.reps,
        "get_af_inputs",
        lambda: model.get_af_inputs(af_model),
    )
    mean_s, min_s, max_s = _stats(input_times)
    step_rows.append(["2 get_af_inputs", args.reps, mean_s, min_s, max_s, ""])

    def _sample_one():
        _touch(model.sample(num=1, batch=1, temperature=args.temperature))

    sample_times = _measure_reps(args.reps, "sample(num=1, batch=1)", _sample_one)
    mean_s, min_s, max_s = _stats(sample_times)
    step_rows.append(
        ["3 sample(num=1, batch=1)", args.reps, mean_s, min_s, max_s, "la primera incluye la compilacion JAX"]
    )

    if not hasattr(model, "set_seed"):
        step_rows.append(
            [
                "4 set_seed + sample(num=1, batch=1)",
                0,
                None,
                None,
                None,
                "set_seed no existe en el ColabDesign importado (v1.1.1 si lo tiene)",
            ]
        )
    else:
        seed_box[0] = 20_000

        def _reseed_and_sample():
            seed_box[0] += 1
            model.set_seed(seed_box[0])
            _sample_one()

        reseed_times = _measure_reps(args.reps, "set_seed + sample", _reseed_and_sample)
        mean_s, min_s, max_s = _stats(reseed_times)
        step_rows.append(
            ["4 set_seed + sample(num=1, batch=1)", args.reps, mean_s, min_s, max_s, "sin reconstruir el modelo"]
        )

    _print_table(["what", "reps", "mean_s", "min_s", "max_s", "notes"], step_rows)
    print(flush=True)

    print("## batch_vs_loop", flush=True)
    print("una pasada por N, no repetida --reps veces", flush=True)
    batch_rows: list[list] = []
    for size in BATCH_SIZES:
        print(f"sample(num=1, batch={size})", file=sys.stderr, flush=True)
        elapsed, note = _run_timed(
            lambda size=size: _touch(model.sample(num=1, batch=size, temperature=args.temperature))
        )
        gpu_in_use, gpu_peak = _gpu_memory_mib()
        if gpu_in_use is None:
            gpu_in_use = _nvml_used_mib()
        batch_rows.append(
            [size, "sample(num=1, batch=N)", elapsed, gpu_in_use, gpu_peak, _cpu_maxrss_mib(), note]
        )

        print(f"{size} x sample(num=1, batch=1)", file=sys.stderr, flush=True)

        def _loop(size=size):
            for _ in range(size):
                _sample_one()

        elapsed, note = _run_timed(_loop)
        gpu_in_use, gpu_peak = _gpu_memory_mib()
        if gpu_in_use is None:
            gpu_in_use = _nvml_used_mib()
        batch_rows.append(
            [size, "N x sample(num=1, batch=1)", elapsed, gpu_in_use, gpu_peak, _cpu_maxrss_mib(), note]
        )

    _print_table(
        ["N", "mode", "seconds", "gpu_in_use_mib", "gpu_peak_mib", "cpu_maxrss_mib", "notes"],
        batch_rows,
    )


if __name__ == "__main__":
    main()
