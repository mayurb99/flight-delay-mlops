"""
src/preprocessing.py
════════════════════════════════════════════════════════
Flight Delay Prediction — SageMaker ProcessingStep Script

Project Lecture 1: Foundation
Runs as Step 1 of the SageMaker Pipeline.

SageMaker mounts:
  INPUT:  /opt/ml/processing/input/raw/
  OUTPUT: /opt/ml/processing/output/train/
          /opt/ml/processing/output/val/
          /opt/ml/processing/output/test/
          /opt/ml/processing/output/reference/

What this script does:
  1. Reads raw flight CSV from SageMaker input path
  2. Validates schema with features.validate_raw_data()
  3. Creates binary target (delayed = ARR_DELAY >= 15)
  4. Engineers features with features.engineer_features()
  5. Splits into train/val/test (70/15/15)
  6. Saves reference stats for Evidently monitoring
  7. Writes all outputs to SageMaker output paths
════════════════════════════════════════════════════════
"""

import os
import sys
import json
import logging
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Add src to path so we can import features.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features import (
    validate_raw_data, create_target,
    engineer_features, get_feature_stats,
    normalize_columns,
    FEATURE_COLS, TARGET_COL,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ── SageMaker container paths ──────────────────────────
INPUT_PATH     = "/opt/ml/processing/input/raw"
OUTPUT_TRAIN   = "/opt/ml/processing/output/train"
OUTPUT_VAL     = "/opt/ml/processing/output/val"
OUTPUT_TEST    = "/opt/ml/processing/output/test"
OUTPUT_REF     = "/opt/ml/processing/output/reference"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--val-size",   type=float, default=0.15)
    parser.add_argument("--test-size",  type=float, default=0.15)
    parser.add_argument("--random-state", type=int, default=42)
    args, _ = parser.parse_known_args()

    logger.info("=" * 60)
    logger.info("preprocessing.py — SageMaker ProcessingStep")
    logger.info(f"val_size={args.val_size}  test_size={args.test_size}")
    logger.info("=" * 60)

    # ── Find input CSV ──────────────────────────────────
    csv_files = [
        f for f in os.listdir(INPUT_PATH) if f.endswith(".csv")
    ]
    if not csv_files:
        raise FileNotFoundError(f"No CSV found in {INPUT_PATH}")

    input_file = os.path.join(INPUT_PATH, csv_files[0])
    logger.info(f"Reading: {input_file}")

    df = pd.read_csv(input_file, low_memory=False)
    logger.info(f"Raw shape: {df.shape}")

    # ── Normalize column names ──────────────────────────
    df = normalize_columns(df)

    # ── Validate ────────────────────────────────────────
    logger.info("Validating raw data...")
    validate_raw_data(df)
    logger.info("✓ Validation passed")

    # ── Create target + drop cancelled flights ──────────
    df = create_target(df)
    logger.info(f"After dropping cancelled: {df.shape}")
    logger.info(f"Delay rate: {df[TARGET_COL].mean():.3f}")

    # ── Feature engineering ─────────────────────────────
    logger.info("Engineering features...")
    df_eng = engineer_features(df)
    logger.info(f"Engineered shape: {df_eng.shape}")
    logger.info(f"Features: {FEATURE_COLS}")

    # ── Train / val / test split ────────────────────────
    # First split off test set
    df_trainval, df_test = train_test_split(
        df_eng,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=df_eng[TARGET_COL],
    )
    # Then split train/val from remaining
    val_relative = args.val_size / (1 - args.test_size)
    df_train, df_val = train_test_split(
        df_trainval,
        test_size=val_relative,
        random_state=args.random_state,
        stratify=df_trainval[TARGET_COL],
    )

    logger.info(f"Train: {df_train.shape}  Val: {df_val.shape}  Test: {df_test.shape}")
    logger.info(f"Train delay rate: {df_train[TARGET_COL].mean():.3f}")
    logger.info(f"Val delay rate:   {df_val[TARGET_COL].mean():.3f}")

    # ── Compute reference stats for Evidently ──────────
    reference_stats = get_feature_stats(df_train[FEATURE_COLS])
    logger.info(f"Computed reference stats for {len(reference_stats)} features")

    # ── Write outputs ───────────────────────────────────
    for path in [OUTPUT_TRAIN, OUTPUT_VAL, OUTPUT_TEST, OUTPUT_REF]:
        os.makedirs(path, exist_ok=True)

    df_train.to_csv(os.path.join(OUTPUT_TRAIN, "train.csv"), index=False)
    df_val.to_csv(  os.path.join(OUTPUT_VAL,   "val.csv"),   index=False)
    df_test.to_csv( os.path.join(OUTPUT_TEST,  "test.csv"),  index=False)

    # Reference dataset for Evidently (a sample of train)
    df_reference = df_train.sample(
        n=min(5000, len(df_train)),
        random_state=args.random_state,
    )
    df_reference.to_csv(os.path.join(OUTPUT_REF, "reference.csv"), index=False)

    # Feature stats JSON
    with open(os.path.join(OUTPUT_REF, "feature_stats.json"), "w") as f:
        json.dump(reference_stats, f, indent=2)

    logger.info("=" * 60)
    logger.info("preprocessing.py complete!")
    logger.info(f"  Train rows   : {len(df_train):,}")
    logger.info(f"  Val rows     : {len(df_val):,}")
    logger.info(f"  Test rows    : {len(df_test):,}")
    logger.info(f"  Reference    : {len(df_reference):,} rows")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
