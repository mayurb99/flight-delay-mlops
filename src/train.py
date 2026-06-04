"""
src/train.py
════════════════════════════════════════════════════════
Flight Delay Prediction — SageMaker TrainingStep Script

Project Lecture 1: Foundation
Runs as Step 2 of the SageMaker Pipeline.

SageMaker mounts:
  INPUT:  /opt/ml/input/data/train/train.csv
          /opt/ml/input/data/val/val.csv
  OUTPUT: /opt/ml/model/         ← model saved here

MLflow tracking: all runs logged to DagsHub
  - Parameters: all hyperparameters + data info
  - Metrics: accuracy, f1, auc, precision, recall
  - Artifacts: model.pkl, feature_importance chart
  - Tags: git_sha, sagemaker_job_name, s3_data_key
  - Model signature: inferred from training data
════════════════════════════════════════════════════════
"""

import os
import sys
import json
import pickle
import tarfile
import logging
import argparse
import subprocess

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, roc_auc_score, confusion_matrix,
)

import mlflow
import mlflow.sklearn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features import FEATURE_COLS, TARGET_COL

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ── SageMaker paths — overridable via args for ProcessingStep mode ─
TRAIN_PATH  = "/opt/ml/input/data/train/train.csv"
VAL_PATH    = "/opt/ml/input/data/val/val.csv"
MODEL_DIR   = "/opt/ml/model"

# ── Hyperparameter defaults ────────────────────────────
N_ESTIMATORS  = int(os.environ.get("N_ESTIMATORS",  "200"))
MAX_DEPTH     = int(os.environ.get("MAX_DEPTH",     "5"))
LEARNING_RATE = float(os.environ.get("LEARNING_RATE","0.08"))
MIN_SAMPLES   = int(os.environ.get("MIN_SAMPLES",   "50"))
RANDOM_STATE  = int(os.environ.get("RANDOM_STATE",  "42"))


def get_git_sha() -> str:
    """Get current git SHA for MLflow tagging."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return os.environ.get("GITHUB_SHA", "unknown")[:8]


def plot_feature_importance(model, feature_names: list, output_path: str):
    """Save feature importance bar chart as PNG."""
    importances = model.feature_importances_
    indices     = np.argsort(importances)[::-1]

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("#0D1B2A")
    ax.set_facecolor("#1A2D42")

    bars = ax.bar(
        range(len(importances)),
        importances[indices],
        color="#00B4D8",
        edgecolor="white",
        linewidth=0.5,
    )
    ax.set_xticks(range(len(importances)))
    ax.set_xticklabels(
        [feature_names[i] for i in indices],
        rotation=45, ha="right", fontsize=9, color="white",
    )
    ax.set_title("Feature Importance — Flight Delay Model",
                 color="white", fontsize=13, fontweight="bold")
    ax.set_ylabel("Importance", color="white")
    ax.tick_params(colors="white")
    ax.spines["bottom"].set_color("white")
    ax.spines["left"].set_color("white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.label.set_color("white")

    plt.tight_layout()
    plt.savefig(output_path, dpi=120, facecolor="#0D1B2A", bbox_inches="tight")
    plt.close()
    logger.info(f"Feature importance chart saved: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-estimators",  type=int,   default=N_ESTIMATORS)
    parser.add_argument("--max-depth",     type=int,   default=MAX_DEPTH)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--min-samples",   type=int,   default=MIN_SAMPLES)
    parser.add_argument("--random-state",  type=int,   default=RANDOM_STATE)
    # Path overrides — used when running as ProcessingStep instead of TrainingStep
    parser.add_argument("--train-path", default=TRAIN_PATH)
    parser.add_argument("--val-path",   default=VAL_PATH)
    parser.add_argument("--model-dir",  default=MODEL_DIR)
    args, _ = parser.parse_known_args()

    logger.info("=" * 60)
    logger.info("train.py — SageMaker Pipeline Training")
    logger.info(f"n_estimators={args.n_estimators}  max_depth={args.max_depth}")
    logger.info(f"learning_rate={args.learning_rate}")
    logger.info(f"train_path={args.train_path}  model_dir={args.model_dir}")
    logger.info("=" * 60)

    # ── Configure MLflow ──────────────────────────────
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
        logger.info(f"MLflow tracking URI: {tracking_uri}")
    else:
        logger.warning("MLFLOW_TRACKING_URI not set — logging to local mlruns/")

    mlflow.set_experiment("flight-delay-prediction")

    # ── Load data ─────────────────────────────────────
    logger.info("Loading training data...")
    df_train = pd.read_csv(args.train_path)
    df_val   = pd.read_csv(args.val_path)

    X_train = df_train[FEATURE_COLS]
    y_train = df_train[TARGET_COL]
    X_val   = df_val[FEATURE_COLS]
    y_val   = df_val[TARGET_COL]

    logger.info(f"Train: {X_train.shape}  Val: {X_val.shape}")
    logger.info(f"Train delay rate: {y_train.mean():.3f}  Val: {y_val.mean():.3f}")

    # ── Train ─────────────────────────────────────────
    with mlflow.start_run() as run:
        logger.info(f"MLflow run ID: {run.info.run_id}")

        # Log parameters
        params = {
            "n_estimators":    args.n_estimators,
            "max_depth":       args.max_depth,
            "learning_rate":   args.learning_rate,
            "min_samples_leaf": args.min_samples,
            "random_state":    args.random_state,
            "train_rows":      len(X_train),
            "val_rows":        len(X_val),
            "n_features":      len(FEATURE_COLS),
        }
        mlflow.log_params(params)

        # Set tags
        mlflow.set_tags({
            "git_sha":             get_git_sha(),
            "sagemaker_job_name":  os.environ.get("TRAINING_JOB_NAME", "local"),
            "s3_data_key":         os.environ.get("S3_DATA_KEY", "unknown"),
            "model_type":          "GradientBoostingClassifier",
        })

        # Train model
        logger.info("Training GradientBoostingClassifier...")
        model = GradientBoostingClassifier(
            n_estimators    = args.n_estimators,
            max_depth       = args.max_depth,
            learning_rate   = args.learning_rate,
            min_samples_leaf= args.min_samples,
            random_state    = args.random_state,
        )
        model.fit(X_train, y_train)
        logger.info("✓ Training complete")

        # ── Evaluate ───────────────────────────────────
        preds_val  = model.predict(X_val)
        probas_val = model.predict_proba(X_val)[:, 1]

        metrics = {
            "val_accuracy":  round(float(accuracy_score(y_val, preds_val)),           4),
            "val_f1":        round(float(f1_score(y_val, preds_val, zero_division=0)),4),
            "val_precision": round(float(precision_score(y_val, preds_val, zero_division=0)), 4),
            "val_recall":    round(float(recall_score(y_val, preds_val, zero_division=0)),    4),
            "val_auc_roc":   round(float(roc_auc_score(y_val, probas_val)),           4),
        }
        mlflow.log_metrics(metrics)
        logger.info(f"Metrics: {json.dumps(metrics, indent=2)}")

        # ── Log feature importance chart ───────────────
        os.makedirs(args.model_dir, exist_ok=True)
        fi_path = os.path.join(args.model_dir, "feature_importance.png")
        plot_feature_importance(model, FEATURE_COLS, fi_path)
        mlflow.log_artifact(fi_path, artifact_path="charts")

        # ── Save + log model ───────────────────────────
        signature = mlflow.models.infer_signature(X_train, preds_val)
        mlflow.sklearn.log_model(
            sk_model        = model,
            artifact_path   = "model",
            signature       = signature,
            registered_model_name = None,
        )

        # Save model.pkl
        model_pkl_path = os.path.join(args.model_dir, "model.pkl")
        with open(model_pkl_path, "wb") as f:
            pickle.dump(model, f)

        # Create model.tar.gz — required for SageMaker Model Registry
        tar_path = os.path.join(args.model_dir, "model.tar.gz")
        with tarfile.open(tar_path, "w:gz") as tar:
            tar.add(model_pkl_path, arcname="model.pkl")
        logger.info(f"Created model.tar.gz: {tar_path}")

        # Save metrics.json for evaluate.py (next step)
        metrics_path = os.path.join(args.model_dir, "metrics.json")
        with open(metrics_path, "w") as f:
            json.dump({
                "metrics": metrics,
                "run_id":  run.info.run_id,
                "model_uri": f"runs:/{run.info.run_id}/model",
            }, f, indent=2)

        logger.info("=" * 60)
        logger.info("train.py complete!")
        logger.info(f"  Val accuracy  : {metrics['val_accuracy']}")
        logger.info(f"  Val F1        : {metrics['val_f1']}")
        logger.info(f"  Val AUC-ROC   : {metrics['val_auc_roc']}")
        logger.info(f"  MLflow run ID : {run.info.run_id}")
        logger.info("=" * 60)


if __name__ == "__main__":
    main()
