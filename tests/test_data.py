"""
tests/test_data.py
════════════════════════════════════════════════════════
Flight Delay Prediction — Data Quality Tests

Runs in: CI + before every training run
Tests the actual flight CSV in S3.
════════════════════════════════════════════════════════
"""

import os
import sys
import pytest
import pandas as pd
import numpy as np
import boto3

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.features import RAW_REQUIRED_COLS, HUB_AIRPORTS, normalize_columns

S3_BUCKET = os.environ.get("S3_BUCKET", "")
S3_KEY    = os.environ.get("S3_KEY",    "data/raw/flights.csv")


@pytest.fixture(scope="module")
def flight_df():
    """Load flight CSV from S3 for data tests. Skip if not configured."""
    if not S3_BUCKET:
        pytest.skip("S3_BUCKET not set — skipping data tests")

    s3   = boto3.client("s3", region_name=os.environ.get("AWS_REGION", "us-east-1"))
    path = "/tmp/test_flights.csv"

    try:
        s3.download_file(S3_BUCKET, S3_KEY, path)
    except Exception as e:
        pytest.skip(f"Could not download s3://{S3_BUCKET}/{S3_KEY}: {e}")

    df = pd.read_csv(path, low_memory=False, nrows=50_000)
    df = normalize_columns(df)   # maps AIRLINE_CODE → OP_CARRIER etc.
    return df


class TestSchema:
    def test_required_columns_present(self, flight_df):
        missing = [c for c in RAW_REQUIRED_COLS if c not in flight_df.columns]
        assert missing == [], f"Missing columns: {missing}"

    def test_dep_time_numeric(self, flight_df):
        assert pd.api.types.is_numeric_dtype(flight_df["CRS_DEP_TIME"])

    def test_distance_numeric(self, flight_df):
        assert pd.api.types.is_numeric_dtype(flight_df["DISTANCE"])


class TestValueRanges:
    def test_dep_time_range(self, flight_df):
        valid = flight_df["CRS_DEP_TIME"].dropna()
        assert valid.min() >= 0
        assert valid.max() <= 2359

    def test_distance_positive(self, flight_df):
        assert flight_df["DISTANCE"].dropna().min() > 0

    def test_elapsed_time_positive(self, flight_df):
        valid = flight_df["CRS_ELAPSED_TIME"].dropna()
        assert valid.min() > 0

    def test_carrier_not_empty(self, flight_df):
        assert flight_df["OP_CARRIER"].dropna().str.len().min() >= 2


class TestDistributions:
    def test_delay_rate_reasonable(self, flight_df):
        df = flight_df.dropna(subset=["ARR_DELAY"])
        delay_rate = (df["ARR_DELAY"] >= 15).mean()
        assert 0.05 <= delay_rate <= 0.60, (
            f"Delay rate {delay_rate:.3f} outside expected 5-60% range"
        )

    def test_multiple_carriers_present(self, flight_df):
        n_carriers = flight_df["OP_CARRIER"].nunique()
        assert n_carriers >= 3, f"Only {n_carriers} carriers found"

    def test_multiple_origins_present(self, flight_df):
        assert flight_df["ORIGIN"].nunique() >= 10

    def test_dep_time_has_spread(self, flight_df):
        assert flight_df["CRS_DEP_TIME"].std() > 100


class TestVolume:
    def test_minimum_rows(self, flight_df):
        assert len(flight_df) >= 1000

    def test_not_all_same_date(self, flight_df):
        n_dates = flight_df["FL_DATE"].nunique()
        assert n_dates >= 2, "All flights on same date — suspicious"

    def test_no_excessive_duplicates(self, flight_df):
        dup_pct = flight_df.duplicated().mean()
        assert dup_pct < 0.05, f"Duplicate rate {dup_pct:.1%} too high"
