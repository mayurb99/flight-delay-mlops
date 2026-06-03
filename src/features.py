"""
src/features.py
════════════════════════════════════════════════════════
Flight Delay Prediction — Feature Engineering

Project Lecture 1: Foundation
Tested in: tests/test_features.py
Used by:   src/preprocessing.py (SageMaker ProcessingStep)
           src/train.py (training)
           src/inference/app.py (serving — must match exactly)

CRITICAL: Any change to these functions must be reflected
in src/inference/app.py. Divergence = training-serving skew.
════════════════════════════════════════════════════════
"""

import pandas as pd
import numpy as np
from typing import Tuple

# ── Column definitions ─────────────────────────────────
RAW_REQUIRED_COLS = [
    "FL_DATE", "OP_CARRIER", "ORIGIN", "DEST",
    "CRS_DEP_TIME", "CRS_ARR_TIME", "CRS_ELAPSED_TIME",
    "DISTANCE", "DEP_DELAY", "ARR_DELAY",
]

FEATURE_COLS = [
    "dep_hour", "dep_day_of_week", "dep_month",
    "is_weekend", "is_holiday_season",
    "carrier_encoded", "origin_encoded", "dest_encoded",
    "route_distance_km", "scheduled_elapsed_min",
    "dep_time_bucket",          # 0=red-eye,1=morning,2=afternoon,3=evening
    "is_hub_origin",            # 1 if major hub airport
    "is_hub_dest",
    "route_id_encoded",
]

TARGET_COL = "delayed"          # 1 if ARR_DELAY >= 15 else 0

# Major hub airports in US
HUB_AIRPORTS = {
    "ATL", "ORD", "LAX", "DFW", "DEN", "JFK", "SFO",
    "SEA", "LAS", "MCO", "EWR", "CLT", "PHX", "MIA",
}


def create_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create binary target: 1 if arrival delay >= 15 minutes.
    Flights that were cancelled (NaN delay) are excluded.
    """
    df = df.copy()
    df = df.dropna(subset=["ARR_DELAY"])
    df[TARGET_COL] = (df["ARR_DELAY"] >= 15).astype(int)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all feature engineering transforms.

    Input:  raw DataFrame with RAW_REQUIRED_COLS
    Output: DataFrame with FEATURE_COLS (+ TARGET_COL if present)

    This function is called:
      - During training in src/preprocessing.py
      - During serving in src/inference/app.py
      - During tests in tests/test_features.py

    Parameters
    ----------
    df : pd.DataFrame
        Raw flight data. Must have RAW_REQUIRED_COLS present.
        'ARR_DELAY' and 'DEP_DELAY' can be NaN (will be handled).

    Returns
    -------
    pd.DataFrame with FEATURE_COLS. If TARGET_COL exists in input
    it is also returned. Row count may decrease (cancelled flights dropped).
    """
    df = df.copy()

    # ── 1. Parse date / time features ─────────────────────
    df["FL_DATE"] = pd.to_datetime(df["FL_DATE"], errors="coerce")
    df["dep_hour"]        = (df["CRS_DEP_TIME"] // 100).clip(0, 23)
    df["dep_day_of_week"] = df["FL_DATE"].dt.dayofweek    # 0=Mon, 6=Sun
    df["dep_month"]       = df["FL_DATE"].dt.month
    df["is_weekend"]      = (df["dep_day_of_week"] >= 5).astype(int)

    # Holiday season: Nov-Dec + Jun-Jul (US travel peaks)
    df["is_holiday_season"] = df["dep_month"].isin([6, 7, 11, 12]).astype(int)

    # ── 2. Time bucket ─────────────────────────────────────
    # 0=red-eye(0-5), 1=morning(6-11), 2=afternoon(12-17), 3=evening(18-23)
    def time_bucket(hour):
        if hour < 6:   return 0
        if hour < 12:  return 1
        if hour < 18:  return 2
        return 3
    df["dep_time_bucket"] = df["dep_hour"].apply(time_bucket)

    # ── 3. Carrier encoding (frequency encoding) ──────────
    carrier_counts = df["OP_CARRIER"].value_counts()
    df["carrier_encoded"] = df["OP_CARRIER"].map(carrier_counts).fillna(0).astype(int)

    # ── 4. Airport encoding (frequency encoding) ──────────
    origin_counts = df["ORIGIN"].value_counts()
    dest_counts   = df["DEST"].value_counts()
    df["origin_encoded"] = df["ORIGIN"].map(origin_counts).fillna(0).astype(int)
    df["dest_encoded"]   = df["DEST"].map(dest_counts).fillna(0).astype(int)

    # ── 5. Route encoding ─────────────────────────────────
    df["route"] = df["ORIGIN"] + "_" + df["DEST"]
    route_counts = df["route"].value_counts()
    df["route_id_encoded"] = df["route"].map(route_counts).fillna(0).astype(int)

    # ── 6. Distance in km ─────────────────────────────────
    # BTS reports distance in miles; convert
    df["route_distance_km"] = (df["DISTANCE"] * 1.60934).round(1)

    # ── 7. Scheduled elapsed time ─────────────────────────
    df["scheduled_elapsed_min"] = df["CRS_ELAPSED_TIME"].fillna(
        df["route_distance_km"] / 12  # rough fallback: 720 km/h
    ).round(0).astype(int)

    # ── 8. Hub flags ──────────────────────────────────────
    df["is_hub_origin"] = df["ORIGIN"].isin(HUB_AIRPORTS).astype(int)
    df["is_hub_dest"]   = df["DEST"].isin(HUB_AIRPORTS).astype(int)

    # ── 9. Select output columns ──────────────────────────
    output_cols = FEATURE_COLS.copy()
    if TARGET_COL in df.columns:
        output_cols.append(TARGET_COL)

    # Ensure all feature columns exist
    for col in FEATURE_COLS:
        if col not in df.columns:
            df[col] = 0

    return df[output_cols].reset_index(drop=True)


def validate_raw_data(df: pd.DataFrame) -> None:
    """
    Validate raw flight data before feature engineering.

    Raises ValueError with descriptive message if:
      - Required columns are missing
      - Any required column is all-null
      - Dataset has fewer than 100 rows
      - CRS_DEP_TIME has values outside 0-2359

    Parameters
    ----------
    df : pd.DataFrame

    Raises
    ------
    ValueError with descriptive message.
    """
    # Check required columns
    missing = [c for c in RAW_REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Check row count
    if len(df) < 100:
        raise ValueError(
            f"Dataset too small: {len(df)} rows. Minimum 100 required."
        )

    # Check for completely null required columns
    for col in ["OP_CARRIER", "ORIGIN", "DEST", "CRS_DEP_TIME"]:
        if df[col].isnull().all():
            raise ValueError(f"Column '{col}' is entirely null.")

    # Check departure time range
    valid_dep = df["CRS_DEP_TIME"].dropna()
    if len(valid_dep) > 0:
        if valid_dep.min() < 0 or valid_dep.max() > 2359:
            raise ValueError(
                f"CRS_DEP_TIME out of range: "
                f"min={valid_dep.min()}, max={valid_dep.max()}. "
                f"Expected 0-2359."
            )


def get_feature_stats(df: pd.DataFrame) -> dict:
    """
    Compute reference statistics for Evidently monitoring.

    Called after feature engineering during training.
    Saved as monitoring/reference_stats.json in S3.
    Used as baseline for drift detection in P4.

    Parameters
    ----------
    df : pd.DataFrame with FEATURE_COLS

    Returns
    -------
    dict with mean, std, min, max per numeric feature
    """
    stats = {}
    for col in FEATURE_COLS:
        if col in df.columns:
            col_data = df[col].dropna()
            stats[col] = {
                "mean": round(float(col_data.mean()), 4),
                "std":  round(float(col_data.std()),  4),
                "min":  round(float(col_data.min()),  4),
                "max":  round(float(col_data.max()),  4),
                "null_pct": round(float(df[col].isnull().mean()), 4),
            }
    return stats
