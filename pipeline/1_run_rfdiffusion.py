"""
Step 1: RFdiffusion backbone generation.
Outputs: /workspace/outputs/{run_name}_0.pdb (and optionally _1.pdb, ...)
When run_id and run_status_db are provided, updates run_status table on completion (COMPLETED + output_pdbs or ERROR + error_details).
"""
import argparse
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests
import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from common.logger import get_logger, log_subprocess_result, resolve_run_id, run_env_for_child

sys.path.append("/workspace/RFdiffusion")

OUTPUTS_DIR = "/workspace/outputs"
RFDIFFUSION_SCRIPT = "/workspace/RFdiffusion/run_inference.py"
PROCESS_NAME = "1_run_rfdiffusion.py"
TASK_RD_DIFFUSION = "RD_DIFFUSION"
STATUS_COMPLETED = "COMPLETED"
STATUS_ERROR = "ERROR"


def update_run_status(
    run_status_db: str,
    run_id: str,
    status: str,
    error_details: str | None = None,
    output_pdbs: dict | None = None,
    logger=None,
) -> None:
    """Update run_status table: status, error_details, output_pdbs (JSON string), updated_at."""
    now = datetime.now(timezone.utc).isoformat()
    output_pdbs_str = json.dumps(output_pdbs) if output_pdbs else None
    with sqlite3.connect(run_status_db) as conn:
        conn.execute(
            "UPDATE run_status SET status = ?, error_details = ?, output_pdbs = ?, updated_at = ? WHERE run_id = ? AND task = ?",
            (status, error_details, output_pdbs_str, now, run_id, TASK_RD_DIFFUSION),
        )
        conn.commit()
    if logger:
        logger.info("run_status updated: status=%s run_id=%s", status, run_id)


def quitar_cadena(pdb_file, cadena_a_quitar, logger) -> str:
    """Remove a specific chain from the PDB file."""
    with open(pdb_file, "r", encoding="utf-8") as f:
        lineas = f.readlines()
    with open(pdb_file, "w", encoding="utf-8") as f:
        for linea in lineas:
            if linea.startswith(("ATOM", "HETATM")) and len(linea) > 21 and linea[21:22] == cadena_a_quitar:
                continue
            f.write(linea)
    logger.info("Chain %s removed from %s", cadena_a_quitar, pdb_file)
    return pdb_file


def download_pdb(pdb_id, remove_chain=None, logger=None):
    """Download PDB from RCSB; optionally remove a chain. Saves to OUTPUTS_DIR."""
    if not pdb_id or len(pdb_id) != 4:
        return pdb_id

    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    pdb_file = os.path.join(OUTPUTS_DIR, f"{pdb_id}.pdb")
    if os.path.exists(pdb_file):
        logger.info("Input PDB already exists: %s", pdb_file)
        return pdb_file

    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    logger.info("Downloading PDB from RCSB: %s", url)
    try:
        response = requests.get(url, timeout=30)
        if response.status_code == 200:
            with open(pdb_file, "w", encoding="utf-8") as f:
                f.write(response.text)
            logger.info("PDB downloaded: %s", pdb_file)
            if remove_chain is not None:
                original_file = os.path.join(OUTPUTS_DIR, f"{pdb_id}_ORIGINAL.pdb")
                shutil.copy2(pdb_file, original_file)
                logger.info("Original PDB copy saved: %s", original_file)
                pdb_file = quitar_cadena(pdb_file, remove_chain, logger)
            return pdb_file
        logger.error("RCSB download failed for %s status=%s", pdb_id, response.status_code)
        return None
    except Exception:
        logger.exception("RCSB download connection error for %s", pdb_id)
        return None


def run_rfdiffusion(
    run_name: str,
    logger,
    run_id: str,
    contigs: str = "12-15/0 R311-337",
    pdb: str = "4Z18",
    iterations: int = 30,
    num_designs: int = 1,
    hotspot: str = "R312,R313,R314,R315",
    chain_to_remove: str | None = "P",
    symmetry: str = "",
    symmetry_order: str = "",
    chains: str = "",
) -> bool:
    """Run RFdiffusion and write output to /workspace/outputs/{run_name}_0.pdb (and _1, ...)."""
    start_time_total = time.time()
    params = {
        "run_name": run_name,
        "contigs": contigs,
        "pdb": pdb,
        "iterations": iterations,
        "num_designs": num_designs,
        "hotspot": hotspot,
        "chain_to_remove": chain_to_remove,
        "symmetry": symmetry,
        "symmetry_order": symmetry_order,
        "chains": chains,
    }
    logger.info("run_rfdiffusion started with parameters: %s", params)

    logger.info("GPU check: CUDA available=%s", torch.cuda.is_available())
    if torch.cuda.is_available():
        logger.info("GPU: %s", torch.cuda.get_device_name(0))
        logger.info(
            "GPU memory: %.1f GB",
            torch.cuda.get_device_properties(0).total_memory / 1e9,
        )
    else:
        logger.warning("No GPU detected — inference will be very slow on CPU")

    pdb_file_actual = ""
    if pdb and len(pdb) == 4:
        pdb_file_actual = download_pdb(pdb, remove_chain=chain_to_remove, logger=logger) or ""
    else:
        pdb_file_actual = pdb or ""

    if pdb_file_actual:
        logger.info("Using input PDB: %s", pdb_file_actual)

    cmd = [
        "python3",
        RFDIFFUSION_SCRIPT,
        f"inference.output_prefix=outputs/{run_name}",
        f"contigmap.contigs=[{contigs}]",
        f"inference.num_designs={num_designs}",
        f"diffuser.T={iterations}",
        "inference.dump_pdb=True",
        "inference.dump_pdb_path=/dev/shm",
    ]
    if pdb_file_actual:
        cmd.append(f"inference.input_pdb={pdb_file_actual}")
    if hotspot:
        cmd.append(f"ppi.hotspot_res=[{hotspot}]")
    if symmetry:
        cmd.append(f"inference.symmetry={symmetry}")
    if symmetry_order:
        cmd.append(f"inference.symmetry_order={symmetry_order}")
    if chains:
        cmd.append(f"inference.chains={chains}")

    logger.info("Launching RFdiffusion subprocess")
    t0 = time.time()
    result = subprocess.run(
        cmd,
        cwd="/workspace",
        capture_output=True,
        text=True,
        env=run_env_for_child(run_id),
    )
    log_subprocess_result(logger, result, cmd, label="run_inference.py", elapsed_sec=time.time() - t0)

    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    generated: list[str] = []

    if result.returncode == 0:
        for i in range(num_designs):
            candidates = [
                f"/dev/shm/outputs/{run_name}_{i}.pdb",
                f"/dev/shm/outputs/{run_name}.pdb" if num_designs == 1 and i == 0 else None,
                f"/dev/shm/{run_name}_{i}.pdb",
                f"/dev/shm/{run_name}.pdb" if num_designs == 1 and i == 0 else None,
                f"/workspace/RFdiffusion/outputs/{run_name}_{i}.pdb",
                f"/workspace/RFdiffusion/outputs/{run_name}.pdb" if num_designs == 1 and i == 0 else None,
            ]
            src = None
            for c in candidates:
                if c and os.path.exists(c):
                    src = c
                    break
            if src:
                dst = f"{OUTPUTS_DIR}/{run_name}_{i}.pdb"
                shutil.copy2(src, dst)
                generated.append(dst)
                logger.info("Output PDB copied: %s -> %s", src, dst)
            else:
                logger.warning("No PDB candidate found for design index %s", i)

        if pdb and len(pdb) == 4:
            original_src = os.path.join(OUTPUTS_DIR, f"{pdb}_ORIGINAL.pdb")
            if not os.path.exists(original_src):
                original_src = f"/workspace/RFdiffusion/{pdb}_ORIGINAL.pdb"
            if os.path.exists(original_src) and not os.path.exists(
                os.path.join(OUTPUTS_DIR, f"{pdb}_ORIGINAL.pdb")
            ):
                dst = os.path.join(OUTPUTS_DIR, f"{pdb}_ORIGINAL.pdb")
                shutil.copy2(original_src, dst)
                logger.info("Reference copy: %s", dst)

            pdb_src = os.path.join(OUTPUTS_DIR, f"{pdb}.pdb")
            if not os.path.exists(pdb_src):
                pdb_src = f"/workspace/RFdiffusion/{pdb}.pdb"
            sin_dst = os.path.join(OUTPUTS_DIR, f"{pdb}_SIN_{chain_to_remove}.pdb")
            if os.path.exists(pdb_src) and not os.path.exists(sin_dst):
                shutil.copy2(pdb_src, sin_dst)
                logger.info("Reference copy: %s", sin_dst)
    else:
        logger.error("RFdiffusion subprocess failed with return code %s", result.returncode)

    elapsed = time.time() - start_time_total
    logger.info("Generated files: %s", generated or "(none)")
    logger.info("run_rfdiffusion finished in %.2f seconds success=%s", elapsed, result.returncode == 0)
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description="Step 1: RFdiffusion backbone generation")
    parser.add_argument("--run_id", "--run-id", dest="run_id", type=str, default=None, help="Run ID for centralized logging and DB")
    parser.add_argument("--run_status_db", type=str, default="/workspace/outputs/run_status.db", help="Path to run_status SQLite DB")
    parser.add_argument("--run_name", type=str, default="pipeline_run", help="Job/run name (used for outputs)")
    parser.add_argument("--contigs", type=str, default="20-35/0 A19-127")
    parser.add_argument("--pdb", type=str, default="4Z18")
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--num_designs", type=int, default=1)
    parser.add_argument("--hotspot", type=str, default="A54,A56,A58,A66,A113,A115,A123,A124,A125")
    parser.add_argument("--chain_to_remove", type=str, default="B", help="Chain to remove from PDB (empty = none)")
    parser.add_argument("--symmetry", type=str, default="")
    parser.add_argument("--symmetry_order", type=str, default="")
    parser.add_argument("--chains", type=str, default="")
    args = parser.parse_args()

    run_id = resolve_run_id(args.run_id)
    run_logger = get_logger(run_id, PROCESS_NAME)
    run_logger.info("%s started", PROCESS_NAME)
    run_logger.info(
        "CLI arguments: run_id=%s run_name=%s contigs=%s pdb=%s iterations=%s num_designs=%s "
        "hotspot=%s chain_to_remove=%s symmetry=%s symmetry_order=%s chains=%s run_status_db=%s",
        run_id,
        args.run_name,
        args.contigs,
        args.pdb,
        args.iterations,
        args.num_designs,
        args.hotspot,
        args.chain_to_remove,
        args.symmetry,
        args.symmetry_order,
        args.chains,
        args.run_status_db,
    )

    chain_to_remove = args.chain_to_remove if args.chain_to_remove else None
    success = False
    try:
        success = run_rfdiffusion(
            run_name=args.run_name,
            logger=run_logger,
            run_id=run_id,
            contigs=args.contigs,
            pdb=args.pdb,
            iterations=args.iterations,
            num_designs=args.num_designs,
            hotspot=args.hotspot,
            chain_to_remove=chain_to_remove,
            symmetry=args.symmetry,
            symmetry_order=args.symmetry_order,
            chains=args.chains,
        )
    except Exception as exc:
        run_logger.exception("%s failed with exception: %s", PROCESS_NAME, exc)
        if args.run_status_db and os.path.isfile(args.run_status_db):
            update_run_status(
                args.run_status_db,
                run_id,
                STATUS_ERROR,
                error_details=str(exc),
                logger=run_logger,
            )
        run_logger.info("%s finished with exit_code=1", PROCESS_NAME)
        sys.exit(1)

    if args.run_status_db and os.path.isfile(args.run_status_db):
        if success:
            output_pdbs = {}
            for i in range(args.num_designs):
                path = os.path.join(OUTPUTS_DIR, f"{args.run_name}_{i}.pdb")
                if os.path.exists(path):
                    output_pdbs[f"output_{i}"] = path
            update_run_status(
                args.run_status_db,
                run_id,
                STATUS_COMPLETED,
                output_pdbs=output_pdbs,
                logger=run_logger,
            )
            run_logger.info("run_status COMPLETED output_pdbs=%s", output_pdbs)
        else:
            update_run_status(
                args.run_status_db,
                run_id,
                STATUS_ERROR,
                error_details="RFdiffusion exited with non-zero return code",
                logger=run_logger,
            )

    exit_code = 0 if success else 1
    run_logger.info("%s finished with exit_code=%s", PROCESS_NAME, exit_code)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
