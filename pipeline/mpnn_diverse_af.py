"""
MPNN with diverse sampling (one sequence per random seed) + AlphaFold evaluation.
Called by 2_run_mpnn_af.py when --use_alphafold to get different sequences and pLDDT/ptm/pae/rmsd.
Usage: python3 mpnn_diverse_af.py --pdb=... --loc=... --contig=... [--num_seqs=8] [--mpnn_sampling_temp=0.7] ...
"""
import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

_PIPELINE = Path(__file__).resolve().parent
_REPO_ROOT = _PIPELINE.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_PIPELINE) not in sys.path:
    sys.path.insert(0, str(_PIPELINE))

from common.logger import get_logger, resolve_run_id
from mpnn_seed import load_binder_keys
from mpnn_unique import run_unique_design

# ColabDesign must be on path (caller sets cwd=/workspace and path)
sys.path.insert(0, "/workspace/colabdesign")

from colabdesign.mpnn import mk_mpnn_model
from colabdesign.af import mk_af_model

from string import ascii_uppercase, ascii_lowercase
alphabet_list = list(ascii_uppercase + ascii_lowercase)


def get_info(contig):
    F = []
    free_chain = False
    fixed_chain = False
    sub_contigs = [x.split("-") for x in contig.split("/")]
    for n, (a, b) in enumerate(sub_contigs):
        if a[0].isalpha():
            L = int(b) - int(a[1:]) + 1
            F += [1] * L
            fixed_chain = True
        else:
            L = int(b)
            F += [0] * L
            free_chain = True
    return F, [fixed_chain, free_chain]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdb", required=True)
    parser.add_argument("--loc", required=True)
    parser.add_argument("--contig", required=True)
    parser.add_argument("--copies", type=int, default=1)
    parser.add_argument("--num_seqs", type=int, default=8)
    parser.add_argument("--num_recycles", type=int, default=3)
    parser.add_argument("--rm_aa", type=str, default="C")
    parser.add_argument("--mpnn_sampling_temp", type=float, default=0.3)
    parser.add_argument("--use_multimer", action="store_true", default=True)
    parser.add_argument("--initial_guess", action="store_true", default=False)
    parser.add_argument("--num_designs", type=int, default=1)
    parser.add_argument("--run_id", "--run-id", dest="run_id", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None, help="Base MPNN sampling seed; omit for historical seeds")
    parser.add_argument("--exclude_seqs_file", type=str, default=None, help="Binder sequences already stored for this backbone")
    parser.add_argument("--max_attempts", type=int, default=None, help="Stop sampling after this many attempts; default num_seqs*20")
    args = parser.parse_args()

    run_id = resolve_run_id(args.run_id)
    run_logger = get_logger(run_id, "mpnn_diverse_af.py")
    run_logger.info("mpnn_diverse_af.py started")
    run_logger.info(
        "Parameters: pdb=%s loc=%s contig=%s num_seqs=%s temp=%s",
        args.pdb, args.loc, args.contig, args.num_seqs, args.mpnn_sampling_temp,
    )
    t0 = time.time()

    if args.rm_aa == "":
        args.rm_aa = None

    # Parse contigs (same as designability_test)
    contigs = []
    for contig_str in args.contig.replace(" ", ":").replace(",", ":").split(":"):
        if len(contig_str) > 0:
            contig = []
            for x in contig_str.split("/"):
                if x != "0":
                    contig.append(x)
            contigs.append("/".join(contig))

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
        "initial_guess": args.initial_guess,
        "best_metric": "rmsd",
        "use_multimer": args.use_multimer,
        "model_names": ["model_1_multimer_v3" if args.use_multimer else "model_1_ptm"],
    }

    if sum(both_chains) == 0 and sum(fixed_chains) > 0 and sum(free_chains) > 0:
        protocol = "binder"
        print("protocol=binder")
        target_chains = [chains[n] for n, x in enumerate(fixed_chains) if x]
        binder_chains = [chains[n] for n, x in enumerate(fixed_chains) if not x]
        af_model = mk_af_model(protocol="binder", **flags)
        prep_flags = {
            "target_chain": ",".join(target_chains),
            "binder_chain": ",".join(binder_chains),
            "rm_aa": args.rm_aa,
        }
    elif sum(fixed_pos) > 0:
        protocol = "partial"
        print("protocol=partial")
        af_model = mk_af_model(protocol="fixbb", use_templates=True, **flags)
        rm_template = np.array(fixed_pos) == 0
        prep_flags = {
            "chain": ",".join(chains),
            "rm_template": rm_template,
            "rm_template_seq": rm_template,
            "copies": args.copies,
            "homooligomer": args.copies > 1,
            "rm_aa": args.rm_aa,
        }
    else:
        protocol = "fixbb"
        print("protocol=fixbb")
        af_model = mk_af_model(protocol="fixbb", **flags)
        prep_flags = {
            "chain": ",".join(chains),
            "copies": args.copies,
            "homooligomer": args.copies > 1,
            "rm_aa": args.rm_aa,
        }

    if args.use_multimer:
        af_terms = ["plddt", "ptm", "i_ptm", "pae", "i_pae", "rmsd"]
    else:
        af_terms = ["plddt", "ptm", "pae", "rmsd"]

    os.makedirs(args.loc, exist_ok=True)
    os.makedirs(f"{args.loc}/all_pdb", exist_ok=True)

    pdb_filename = args.pdb

    af_model.prep_inputs(pdb_filename, **prep_flags)
    if protocol == "partial":
        p = np.where(fixed_pos)[0]
        af_model.opt["fix_pos"] = p[p < af_model._len]

    max_attempts = args.max_attempts if args.max_attempts is not None else args.num_seqs * 20
    run_unique_design(
        loc=args.loc,
        af_model=af_model,
        mk_mpnn_model=mk_mpnn_model,
        num_seqs=args.num_seqs,
        max_attempts=max_attempts,
        temperature=args.mpnn_sampling_temp,
        base_seed=args.seed,
        num_recycles=args.num_recycles,
        af_terms=af_terms,
        known_keys=load_binder_keys(args.exclude_seqs_file),
        logger=run_logger,
    )
    run_logger.info("mpnn_diverse_af.py finished in %.2f seconds", time.time() - t0)


if __name__ == "__main__":
    main()
