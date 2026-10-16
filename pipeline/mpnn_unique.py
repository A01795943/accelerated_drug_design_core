"""Sample unique binder sequences, then score only the ones that were kept.

Nothing in this module imports ColabDesign. ``mpnn_diverse_af.py`` builds the
models and passes them in, and the tests pass fakes.

One ``mk_mpnn_model`` and one ``get_af_inputs`` serve the whole run. Each
round calls ``set_seed(diverse_mpnn_seed(base_seed, round_index))`` and
``sample(num=1, batch=round_size)``. Sequences are reproducible for the same
base seed and the same effective round size. Changing ``--round_size`` or
``--max_round_size`` changes the seeds and the batch shapes, so ProteinMPNN
draws different sequences. ``max_attempts`` is the total number of sequences
returned by those batches, not a count of rounds.
"""
from __future__ import annotations

import csv
import json
import os
import time
from pathlib import Path

from mpnn_seed import binder_key_for_length, diverse_mpnn_seed

# backbone_binder_keys.binder_key is VARCHAR(64). Longer binders are stored
# truncated, so a known key may be only the first 64 characters.
STORED_KEY_LIMIT = 64

# scripts/measure_mpnn_steps.py times sample(num=1, batch=N) for
# N in {1, 8, 32, 64, 100, 200}. 32 is the default memory cap: it stays in
# the lower half of that grid until a measured run raises --max_round_size.
DEFAULT_MAX_ROUND_SIZE = 32

# Draw up to twice the sequences still missing so one round can absorb a
# reject, and never more than the configured round size.
_PENDING_MARGIN = 2


def matches_known_key(key: str, known: set[str]) -> bool:
    if not key:
        return False
    if key in known:
        return True
    return len(key) > STORED_KEY_LIMIT and key[:STORED_KEY_LIMIT] in known


def write_progress(loc: str, phase: str, attempts: int, accepted: int, target: int) -> None:
    path = Path(loc) / "progress.json"
    path.write_text(
        json.dumps(
            {"phase": phase, "attempts": attempts, "accepted": accepted, "target": target}
        ),
        encoding="utf-8",
    )


def merge_run_stats(param_details: dict, output_dir: str) -> dict:
    path = os.path.join(output_dir, "run_stats.json")
    if not os.path.isfile(path):
        return param_details
    try:
        with open(path, encoding="utf-8") as handle:
            extra = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return param_details
    if isinstance(extra, dict):
        param_details.update(extra)
    return param_details


def round_batch_size(
    pending: int,
    remaining: int,
    round_size: int,
    max_round_size: int,
) -> int:
    """Sequences to draw this round.

    ``min(pending with margin, remaining attempt budget, memory cap)``, and
    the margin itself is also capped by the configured ``round_size``.
    """
    if pending <= 0 or remaining <= 0 or round_size <= 0 or max_round_size <= 0:
        return 0
    pending_with_margin = min(round_size, pending * _PENDING_MARGIN)
    return min(pending_with_margin, remaining, max_round_size)


def _score_at(scores, index: int) -> float:
    if isinstance(scores, (list, tuple)):
        return float(scores[index])
    shape = getattr(scores, "shape", None)
    if shape is not None and getattr(shape, "__len__", lambda: 0)():
        return float(scores[index])
    return float(scores)


def run_unique_design(
    *,
    loc: str,
    af_model,
    mk_mpnn_model,
    num_seqs: int,
    max_attempts: int,
    temperature: float,
    base_seed: int | None,
    num_recycles: int,
    af_terms: list[str],
    known_keys: set[str],
    logger=None,
    round_size: int | None = None,
    max_round_size: int = DEFAULT_MAX_ROUND_SIZE,
) -> dict:
    os.makedirs(loc, exist_ok=True)
    os.makedirs(f"{loc}/all_pdb", exist_ok=True)
    known = set(known_keys)
    seen: set[str] = set()
    accepted: list[tuple[str, float, str]] = []
    attempts = 0
    skipped_db = 0
    skipped_batch = 0
    configured_round = num_seqs if round_size is None else round_size
    if configured_round < 1:
        raise ValueError("round_size must be >= 1")
    if max_round_size < 1:
        raise ValueError("max_round_size must be >= 1")

    sampling_started = time.perf_counter()
    write_progress(loc, "sampling", 0, 0, num_seqs)
    print("running proteinMPNN (one model, seeded rounds)...")
    mpnn_model = None
    if num_seqs > 0 and max_attempts > 0:
        mpnn_model = mk_mpnn_model(seed=diverse_mpnn_seed(base_seed, 0))
        mpnn_model.get_af_inputs(af_model)
    round_index = 0
    while len(accepted) < num_seqs and attempts < max_attempts:
        batch_size = round_batch_size(
            num_seqs - len(accepted),
            max_attempts - attempts,
            configured_round,
            max_round_size,
        )
        if batch_size <= 0 or mpnn_model is None:
            break
        seed = diverse_mpnn_seed(base_seed, round_index)
        round_started = time.perf_counter()
        accepted_before = len(accepted)
        mpnn_model.set_seed(seed)
        out = mpnn_model.sample(num=1, batch=batch_size, temperature=temperature)
        seqs = out["seq"]
        produced = min(batch_size, len(seqs))
        if produced <= 0:
            attempts += batch_size
            break
        attempts += produced
        for index in range(produced):
            if len(accepted) >= num_seqs:
                break
            raw = seqs[index]
            seq = raw if hasattr(raw, "strip") else str(raw)
            score = _score_at(out["score"], index)
            key = binder_key_for_length(seq, getattr(af_model, "_len", None))
            if matches_known_key(key, known):
                skipped_db += 1
                continue
            if not key or key in seen:
                skipped_batch += 1
                continue
            seen.add(key)
            accepted.append((seq, score, key))
        write_progress(loc, "sampling", attempts, len(accepted), num_seqs)
        accepted_round = len(accepted) - accepted_before
        if logger is not None:
            logger.info(
                "round=%d sampled=%d accepted=%d skipped_db=%d skipped_batch=%d "
                "round_accept_rate=%.3f round_seconds=%.3f",
                round_index,
                attempts,
                len(accepted),
                skipped_db,
                skipped_batch,
                (accepted_round / produced) if produced else 0.0,
                time.perf_counter() - round_started,
            )
        round_index += 1
    sec_sampling = time.perf_counter() - sampling_started
    write_progress(loc, "sampling", attempts, len(accepted), num_seqs)
    if len(accepted) < num_seqs and logger is not None:
        logger.warning(
            "sampling hit max_attempts=%d with accepted=%d missing=%d",
            max_attempts,
            len(accepted),
            num_seqs - len(accepted),
        )

    af_started = time.perf_counter()
    write_progress(loc, "alphafold", attempts, len(accepted), num_seqs)
    print("running AlphaFold...")
    data = []
    best = {"rmsd": float("inf"), "n": None}
    available_terms = None
    with open(f"{loc}/design.fasta", "w", encoding="utf-8") as fasta:
        for n, (seq, score, _key) in enumerate(accepted):
            sub_seq = seq.replace("/", "")[-af_model._len :]
            af_model.predict(seq=sub_seq, num_recycles=num_recycles, verbose=False)
            row = {"design": 0, "n": n, "mpnn": score}
            for term in af_terms:
                if term not in af_model.aux.get("log", {}):
                    continue
                value = af_model.aux["log"][term]
                if term in ("i_pae", "pae"):
                    value = value * 31
                row[term] = value
            row["seq"] = seq
            if available_terms is None:
                available_terms = [term for term in af_terms if term in row]
            data.append(row)
            rmsd = row.get("rmsd", float("inf"))
            if rmsd < best["rmsd"]:
                best = {"rmsd": rmsd, "n": n}
            af_model.save_current_pdb(f"{loc}/all_pdb/design0_n{n}.pdb")
            af_model._save_results(save_best=True, verbose=False)
            af_model._k += 1
            terms_for_row = available_terms if available_terms is not None else [t for t in af_terms if t in row]
            score_line = [f"design:0 n:{n}", f"mpnn:{score:.3f}"]
            for term in terms_for_row:
                score_line.append(f"{term}:{row[term]:.3f}")
            print(" ".join(score_line) + " " + (seq[:80] + "..." if len(seq) > 80 else seq))
            fasta.write(f'>{"|".join(score_line)}\n{seq}\n')
            write_progress(loc, "alphafold", attempts, n + 1, num_seqs)
    sec_alphafold = time.perf_counter() - af_started

    if best["n"] is not None:
        with open(f"{loc}/best.pdb", "w", encoding="utf-8") as handle:
            handle.write(f"REMARK 001 design 0 N {best['n']} RMSD {best['rmsd']:.3f}\n")
            with open(f"{loc}/all_pdb/design0_n{best['n']}.pdb", encoding="utf-8") as source:
                handle.write(source.read())

    labels = ["design", "n", "mpnn"] + (available_terms or []) + ["seq"]
    with open(f"{loc}/mpnn_results.csv", "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=labels, extrasaction="ignore")
        writer.writeheader()
        for row in data:
            writer.writerow(row)
    stats = {
        "attempts_used": attempts,
        "accepted": len(accepted),
        "skipped_db_duplicates": skipped_db,
        "skipped_batch_duplicates": skipped_batch,
        "capped": len(accepted) < num_seqs,
        "sec_sampling": round(sec_sampling, 3),
        "sec_alphafold": round(sec_alphafold, 3),
    }
    (Path(loc) / "run_stats.json").write_text(json.dumps(stats), encoding="utf-8")
    if logger is not None:
        logger.info(
            "Saved %s/mpnn_results.csv with %d sequences (attempts=%d capped=%s)",
            loc,
            len(accepted),
            attempts,
            stats["capped"],
        )
    return stats
