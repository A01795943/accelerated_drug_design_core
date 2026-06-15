"""
Step 3: Rosetta stability analysis.
Reads PDBs from:
  - /workspace/outputs/{run_name}/all_pdb/*.pdb, or
  - /workspace/outputs/{run_name}/*.pdb, or
  - /workspace/outputs/{run_name}_0.pdb (single backbone)
Writes: *_relaxed.pdb and report to stdout.
"""
import argparse
import glob
import os
import sys
import time
from pathlib import Path

import pyrosetta
from pyrosetta import *
from pyrosetta.rosetta.core.scoring import *
from pyrosetta.rosetta.protocols.relax import FastRelax
from pyrosetta.rosetta.core.scoring import CA_rmsd

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from common.logger import get_logger, resolve_run_id

pyrosetta.init("-mute all")

OUTPUTS_DIR = "/workspace/outputs"
PROCESS_NAME = "3_run_rosetta.py"


def analizar_estructura_completa(pdb_file, logger) -> dict:
    """Full Rosetta REF15 analysis for one PDB."""
    start_time = time.time()
    logger.info("Rosetta analysis started: %s", pdb_file)

    pose = pose_from_pdb(pdb_file)
    scorefxn = get_fa_scorefxn()
    total_energy = scorefxn(pose)
    energy_per_residue = total_energy / pose.total_residue()

    logger.info("Residues=%d energy=%.2f REU energy/res=%.2f", pose.total_residue(), total_energy, energy_per_residue)

    energies = pose.energies()
    score_terms = [
        fa_atr, fa_rep, fa_sol, fa_elec,
        hbond_sc, hbond_bb_sc, omega, rama_prepro, p_aa_pp,
    ]
    for score_type in score_terms:
        value = energies.total_energies()[score_type]
        logger.info("Score %s: %.2f REU", score_type.name, value)

    residue_energies = []
    for i in range(1, pose.total_residue() + 1):
        res_energy = pose.energies().residue_total_energy(i)
        residue_energies.append((i, pose.residue(i).name3(), res_energy))
    residue_energies.sort(key=lambda x: x[2], reverse=True)
    for res_num, res_name, energy in residue_energies[:10]:
        logger.info("Unstable residue %d %s: %.2f", res_num, res_name, energy)

    logger.info("Running FastRelax")
    relax = FastRelax()
    relax.set_scorefxn(scorefxn)
    relaxed_pose = pose.clone()
    relax.apply(relaxed_pose)
    relaxed_energy = scorefxn(relaxed_pose)
    energy_change = relaxed_energy - total_energy
    rmsd = CA_rmsd(pose, relaxed_pose)

    logger.info(
        "Post-relax energy=%.2f change=%.2f RMSD=%.2f A",
        relaxed_energy, energy_change, rmsd,
    )

    output_file = pdb_file.replace(".pdb", "_relaxed.pdb")
    relaxed_pose.dump_pdb(output_file)
    logger.info("Relaxed structure saved: %s", output_file)
    logger.info("Analysis finished in %.2f seconds", time.time() - start_time)

    return {
        "archivo": pdb_file,
        "residuos": pose.total_residue(),
        "energia_inicial": total_energy,
        "energia_relajada": relaxed_energy,
        "energia_por_residuo": energy_per_residue,
        "rmsd": rmsd,
        "energy_change": energy_change,
    }


def analizar_carpeta_completa(carpeta, logger) -> list:
    """Analyze all PDBs in a folder."""
    start_total = time.time()
    logger.info("Rosetta batch analysis started: %s", carpeta)
    pdb_files = glob.glob(f"{carpeta}/*.pdb")
    if not pdb_files:
        logger.error("No PDB files found in %s", carpeta)
        return []
    logger.info("Found %d PDB files", len(pdb_files))
    resultados = []
    for pdb_file in pdb_files:
        try:
            resultado = analizar_estructura_completa(pdb_file, logger)
            resultados.append(resultado)
        except Exception:
            logger.exception("Error analyzing %s", pdb_file)
    if resultados:
        resultados.sort(key=lambda x: x["energia_por_residuo"])
        for i, res in enumerate(resultados):
            energy_res = res["energia_por_residuo"]
            estado = "EXCELENTE" if energy_res < -2.0 else "BUENA" if energy_res < -1.5 else "ACEPTABLE" if energy_res < -1.0 else "PROBLEMA"
            logger.info(
                "Rank %d %s residues=%d energy/res=%.2f rmsd=%.2f status=%s",
                i + 1, os.path.basename(res["archivo"]), res["residuos"], energy_res, res["rmsd"], estado,
            )
    logger.info("Batch analysis finished in %.2f seconds", time.time() - start_total)
    return resultados


def get_pdb_folder(run_name: str) -> str | None:
    """Return folder path that contains PDBs for this run, or None."""
    all_pdb = f"{OUTPUTS_DIR}/{run_name}/all_pdb"
    if os.path.isdir(all_pdb) and glob.glob(f"{all_pdb}/*.pdb"):
        return all_pdb
    run_dir = f"{OUTPUTS_DIR}/{run_name}"
    if os.path.isdir(run_dir) and glob.glob(f"{run_dir}/*.pdb"):
        return run_dir
    single = f"{OUTPUTS_DIR}/{run_name}_0.pdb"
    if os.path.isfile(single):
        return single
    return None


def run_rosetta(run_name: str, logger) -> bool:
    """Run Rosetta analysis for run_name. Returns True on success."""
    location = get_pdb_folder(run_name)
    if location is None:
        logger.error(
            "No PDBs found for run_name=%s (checked all_pdb, run dir, single backbone)",
            run_name,
        )
        return False
    logger.info("Input location: %s", location)
    if os.path.isfile(location):
        analizar_estructura_completa(location, logger)
        return True
    analizar_carpeta_completa(location, logger)
    return True


def main():
    parser = argparse.ArgumentParser(description="Step 3: Rosetta stability analysis")
    parser.add_argument("--run_id", "--run-id", dest="run_id", type=str, default=None)
    parser.add_argument("--run_name", type=str, default="pipeline_run", help="Must match step 1 and 2 run_name")
    args = parser.parse_args()

    run_id = resolve_run_id(args.run_id)
    run_logger = get_logger(run_id, PROCESS_NAME)
    run_logger.info("%s started", PROCESS_NAME)
    run_logger.info("CLI arguments: run_id=%s run_name=%s", run_id, args.run_name)

    try:
        success = run_rosetta(args.run_name, run_logger)
    except Exception:
        run_logger.exception("%s failed", PROCESS_NAME)
        run_logger.info("%s finished with exit_code=1", PROCESS_NAME)
        sys.exit(1)

    exit_code = 0 if success else 1
    run_logger.info("%s finished with exit_code=%s", PROCESS_NAME, exit_code)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
