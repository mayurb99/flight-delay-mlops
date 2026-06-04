"""
src/evaluate.py
════════════════════════════════════════════════════════
Flight Delay Prediction — Model Evaluation + Champion Compare

Project Lecture 1: Foundation
Runs as Step 3 of the SageMaker Pipeline.

What this does:
  1. Load metrics.json written by train.py
  2. Fetch champion metrics from MLflow (model tagged 'champion')
  3. Compare challenger vs champion on key metrics
  4. Write comparison.json for ConditionStep to read
  5. Writes accuracy.json in the exact path ConditionStep reads:
       metrics.challenger_beats_champion.value (1.0=yes, 0.0=no)

SageMaker mounts:
  INPUT:  /opt/ml/processing/input/model/    (model.tar.gz)
          /opt/ml/processing/input/val/      (val.csv)
  OUTPUT: /opt/ml/processing/output/eval/   (comparison.json, accuracy.json)
════════════════════════════════════════════════════════
"""

import os
import sys
import json
import tarfile
import pickle
import logging
import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

import mlflow
from mlflow.tracking import MlflowClient

for _p in ["/opt/ml/processing/input/deps", os.path.dirname(os.path.abspath(__file__))]:
    if _p not in sys.path:
        sys.path.insert(0, _p)
from features import FEATURE_COLS, TARGET_COL

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ── SageMaker container paths ──────────────────────────
MODEL_INPUT  = "/opt/ml/processing/input/model"
VAL_INPUT    = "/opt/ml/processing/input/val"
OUTPUT_PATH  = "/opt/ml/processing/output/eval"

# ── Configuration ──────────────────────────────────────
MODEL_NAME           = "flight-delay-model"
IMPROVEMENT_THRESHOLD = 0.01   # challenger must beat champion by 1%
METRIC_TO_COMPARE     = "val_f1"  # primary comparison metric


def get_champion_metrics(client: MlflowClient) -> dict:
    """
    Fetch metrics of the current champion model from MLflow.
    Champion is identified by the 'champion' alias.
    Returns empty dict if no champion exists yet.
    """
    try:
        champion_version = client.get_model_version_by_alias(
            name=MODEL_NAME, alias="champion"
        )
        run = client.get_run(champion_version.run_id)
        metrics = {k: v for k, v in run.data.metrics.items()}
        logger.info(f"Champion: version={champion_version.version} "
                    f"F1={metrics.get('val_f1', 'N/A')}")
        return metrics
    except Exception as e:
        logger.info(f"No champion found ({e}) — challenger auto-wins first deployment")
        return {}


def load_challenger_model(model_dir: str):
    """Load model from directory (handles model.tar.gz or model.pkl directly)."""
    model_pkl = os.path.join(model_dir, "model.pkl")

    if os.path.exists(model_pkl):
        with open(model_pkl, "rb") as f:
            return pickle.load(f)

    # Try extracting from tar.gz
    tar_files = [f for f in os.listdir(model_dir) if f.endswith(".tar.gz")]
    if tar_files:
        tar_path = os.path.join(model_dir, tar_files[0])
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(model_dir)
        if os.path.exists(model_pkl):
            with open(model_pkl, "rb") as f:
                return pickle.load(f)

    raise FileNotFoundError(f"No model.pkl found in {model_dir}")


def main():
    logger.info("=" * 60)
    logger.info("evaluate.py — Champion vs Challenger")
    logger.info(f"Improvement threshold: {IMPROVEMENT_THRESHOLD}")
    logger.info(f"Comparison metric: {METRIC_TO_COMPARE}")
    logger.info("=" * 60)

    # ── Setup MLflow ──────────────────────────────────
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()

    os.makedirs(OUTPUT_PATH, exist_ok=True)

    # ── Load challenger model + val data ─────────────
    logger.info("Loading challenger model...")
    model = load_challenger_model(MODEL_INPUT)

    logger.info("Loading validation data...")
    df_val = pd.read_csv(os.path.join(VAL_INPUT, "val.csv"))
    X_val  = df_val[FEATURE_COLS]
    y_val  = df_val[TARGET_COL]

    # ── Compute challenger metrics ────────────────────
    preds  = model.predict(X_val)
    probas = model.predict_proba(X_val)[:, 1]

    challenger_metrics = {
        "val_accuracy": round(float(accuracy_score(y_val, preds)),           4),
        "val_f1":       round(float(f1_score(y_val, preds, zero_division=0)),4),
        "val_auc_roc":  round(float(roc_auc_score(y_val, probas)),           4),
    }
    logger.info(f"Challenger metrics: {challenger_metrics}")

    # ── Get champion metrics ──────────────────────────
    champion_metrics = get_champion_metrics(client)

    # ── Compare ───────────────────────────────────────
    challenger_score = challenger_metrics.get(METRIC_TO_COMPARE, 0.0)
    champion_score   = champion_metrics.get(METRIC_TO_COMPARE, 0.0)
    improvement      = challenger_score - champion_score
    beats_champion   = improvement >= IMPROVEMENT_THRESHOLD

    logger.info(f"Champion {METRIC_TO_COMPARE}  : {champion_score:.4f}")
    logger.info(f"Challenger {METRIC_TO_COMPARE}: {challenger_score:.4f}")
    logger.info(f"Improvement: {improvement:+.4f}")
    logger.info(f"Beats champion: {beats_champion}")

    # ── Write comparison.json ─────────────────────────
    comparison = {
        "challenger": challenger_metrics,
        "champion":   champion_metrics,
        "comparison": {
            "metric":       METRIC_TO_COMPARE,
            "challenger":   challenger_score,
            "champion":     champion_score,
            "improvement":  round(improvement, 4),
            "threshold":    IMPROVEMENT_THRESHOLD,
            "beats_champion": beats_champion,
        },
        "metrics": {
            # This is the path ConditionStep reads via JsonGet:
            # json_path = "metrics.challenger_beats_champion.value"
            "challenger_beats_champion": {
                "value": 1.0 if beats_champion else 0.0,
            },
            "challenger_f1": {
                "value": challenger_score,
            },
        }
    }

    comparison_path = os.path.join(OUTPUT_PATH, "comparison.json")
    with open(comparison_path, "w") as f:
        json.dump(comparison, f, indent=2)
    logger.info(f"Wrote: {comparison_path}")

    logger.info("=" * 60)
    logger.info("evaluate.py complete!")
    logger.info(f"  Decision: {'✓ REGISTER challenger' if beats_champion else '✗ KEEP champion'}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
