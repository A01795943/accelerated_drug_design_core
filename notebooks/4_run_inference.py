import argparse
import contextlib
import io
import json
import os
import sys
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import torch

warnings.filterwarnings("ignore")

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from common.logger import get_logger, resolve_run_id
from models.esm2_embedder import ESM2Embedder

base_path = _REPO_ROOT
PROCESS_NAME = "4_run_inference.py"

MODEL = None
EMBEDDER = None
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_resources(logger=None):
    """Load model and embedder once."""
    global MODEL, EMBEDDER

    if MODEL is None:
        model_path = os.path.join(base_path, "model.pkl")
        if logger:
            logger.info("Loading model from %s", model_path)
        MODEL = joblib.load(model_path)

    if EMBEDDER is None:
        if logger:
            logger.info("Loading ESM2 embedder on device=%s", DEVICE)
        with contextlib.redirect_stdout(io.StringIO()):
            EMBEDDER = ESM2Embedder(device=DEVICE, batch_size=1)


def run_inference(sequence: str, energy_score: float, logger=None) -> dict:
    """Return ptm and iptm predictions."""
    clean_sequence = sequence.replace("/", "")
    if logger:
        logger.info("Inference started seq_len=%d energy_score=%s", len(clean_sequence), energy_score)

    try:
        load_resources(logger)

        with contextlib.redirect_stdout(io.StringIO()):
            embeddings = EMBEDDER.embed([clean_sequence])
            if hasattr(embeddings, "numpy"):
                X_emb = embeddings.numpy()
            else:
                X_emb = np.asarray(embeddings)

        X_final = np.hstack([X_emb, [[float(energy_score)]]])
        preds = MODEL.predict(X_final)
        result = {
            "ptm": round(float(preds[0][0]), 4),
            "iptm": round(float(preds[0][1]), 4),
        }
        if logger:
            logger.info("Inference result: %s", result)
        return result
    except Exception as exc:
        if logger:
            logger.exception("Inference failed: %s", exc)
        return {"ptm": None, "iptm": None, "error": str(exc)}


def main():
    parser = argparse.ArgumentParser(description="Step 4: surrogate ptm/iptm inference")
    parser.add_argument("--run_id", "--run-id", dest="run_id", type=str, default=None)
    parser.add_argument("sequence", nargs="?", default=None)
    parser.add_argument("energy_score", nargs="?", default=None)
    args = parser.parse_args()

    run_id = resolve_run_id(args.run_id)
    run_logger = get_logger(run_id, PROCESS_NAME)
    run_logger.info("%s started", PROCESS_NAME)

    if args.sequence is None or args.energy_score is None:
        run_logger.error("Usage: python3 4_run_inference.py [--run_id ID] <sequence> <score>")
        sys.exit(1)

    run_logger.info("CLI arguments: sequence_len=%d energy_score=%s", len(args.sequence), args.energy_score)
    t0 = time.time()
    result = run_inference(args.sequence, float(args.energy_score), logger=run_logger)
    run_logger.info("Finished in %.2f seconds", time.time() - t0)
    print(json.dumps(result))
    sys.exit(0 if result.get("error") is None else 1)


if __name__ == "__main__":
    main()
