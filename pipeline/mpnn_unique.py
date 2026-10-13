"""Sample unique binder sequences, then score only the ones that were kept.

Nothing in this module imports ColabDesign. ``mpnn_diverse_af.py`` builds the
models and passes them in, and the tests pass fakes.
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
) -> dict:
    os.makedirs(loc, exist_ok=True)
    os.makedirs(f"{loc}/all_pdb", exist_ok=True)
    known = set(known_keys)
    seen: set[str] = set()
    accepted: list[tuple[str, float, str]] = []
    attempts = 0
    skipped_db = 0
    skipped_batch = 0

    sampling_started = time.perf_counter()
    write_progress(loc, "sampling", 0, 0, num_seqs)
    print("running proteinMPNN (one model per seed for diversity)...")
    while len(accepted) < num_seqs and attempts < max_attempts:
        if attempts % 25 == 0:
            write_progress(loc, "sampling", attempts, len(accepted), num_seqs)
        seed = diverse_mpnn_seed(base_seed, attempts)
        attempts += 1
        mpnn_model = mk_mpnn_model(seed=seed)
        mpnn_model.get_af_inputs(af_model)
        out = mpnn_model.sample(num=1, batch=1, temperature=temperature)
        raw = out["seq"][0]
        seq = raw if hasattr(raw, "strip") else str(raw)
        score_raw = out["score"]
        score = float(score_raw[0] if isinstance(score_raw, (list, tuple)) else score_raw)
        key = binder_key_for_length(seq, getattr(af_model, "_len", None))
        if matches_known_key(key, known):
            skipped_db += 1
            continue
        if not key or key in seen:
            skipped_batch += 1
            continue
        seen.add(key)
        accepted.append((seq, score, key))
    sec_sampling = time.perf_counter() - sampling_started
    write_progress(loc, "sampling", attempts, len(accepted), num_seqs)

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
