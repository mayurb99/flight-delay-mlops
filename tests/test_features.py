"""
tests/test_features.py
════════════════════════════════════════════════════════
Flight Delay Prediction — Feature Engineering Unit Tests

Runs in: CI on every PR (ci.yml)
No AWS needed. No MLflow needed. Pure Python.
════════════════════════════════════════════════════════
"""

import pytest
import pandas as pd
import numpy as np
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.features import (
    engineer_features, validate_raw_data,
    create_target, get_feature_stats,
    FEATURE_COLS, TARGET_COL, RAW_REQUIRED_COLS, HUB_AIRPORTS,
)


def make_raw_row(**kwargs) -> pd.DataFrame:
    """Build a single-row raw flight DataFrame with valid defaults."""
    defaults = {
        "FL_DATE":          "2024-02-15",
        "OP_CARRIER":       "AA",
        "ORIGIN":           "JFK",
        "DEST":             "LAX",
        "CRS_DEP_TIME":     900,     # 9:00am
        "CRS_ARR_TIME":     1230,
        "CRS_ELAPSED_TIME": 360,
        "DISTANCE":         2475,    # miles
        "DEP_DELAY":        5.0,
        "ARR_DELAY":        10.0,
    }
    defaults.update(kwargs)
    return pd.DataFrame([defaults])


def make_raw_df(n: int = 300, **overrides) -> pd.DataFrame:
    """Build a multi-row valid raw flight DataFrame."""
    np.random.seed(42)
    hours = np.random.choice([600, 900, 1200, 1500, 1800, 2100], n)
    df = pd.DataFrame({
        "FL_DATE":          pd.date_range("2024-01-01", periods=n, freq="1h").strftime("%Y-%m-%d"),
        "OP_CARRIER":       np.random.choice(["AA","UA","DL","SW","WN"], n),
        "ORIGIN":           np.random.choice(["JFK","LAX","ORD","ATL","DFW"], n),
        "DEST":             np.random.choice(["SFO","MIA","SEA","BOS","PHX"], n),
        "CRS_DEP_TIME":     hours,
        "CRS_ARR_TIME":     hours + 200,
        "CRS_ELAPSED_TIME": np.random.randint(60, 400, n),
        "DISTANCE":         np.random.randint(200, 3000, n),
        "DEP_DELAY":        np.random.uniform(-10, 120, n),
        "ARR_DELAY":        np.random.uniform(-15, 150, n),
    })
    for k, v in overrides.items():
        df[k] = v
    return df


# ════════════════════════════════════════════════════════
# TARGET CREATION
# ════════════════════════════════════════════════════════

class TestCreateTarget:
    def test_delay_15_or_more_is_1(self):
        df = make_raw_row(ARR_DELAY=15.0)
        result = create_target(df)
        assert result[TARGET_COL].iloc[0] == 1

    def test_delay_under_15_is_0(self):
        df = make_raw_row(ARR_DELAY=14.9)
        result = create_target(df)
        assert result[TARGET_COL].iloc[0] == 0

    def test_exactly_15_is_delayed(self):
        df = make_raw_row(ARR_DELAY=15.0)
        result = create_target(df)
        assert result[TARGET_COL].iloc[0] == 1

    def test_negative_delay_is_not_delayed(self):
        df = make_raw_row(ARR_DELAY=-20.0)
        result = create_target(df)
        assert result[TARGET_COL].iloc[0] == 0

    def test_cancelled_flights_dropped(self):
        df = make_raw_row(ARR_DELAY=None)
        result = create_target(df)
        assert len(result) == 0


# ════════════════════════════════════════════════════════
# DEPARTURE HOUR
# ════════════════════════════════════════════════════════

class TestDepHour:
    def test_0900_gives_hour_9(self):
        df = make_raw_row(CRS_DEP_TIME=900)
        result = engineer_features(df)
        assert result["dep_hour"].iloc[0] == 9

    def test_1430_gives_hour_14(self):
        df = make_raw_row(CRS_DEP_TIME=1430)
        result = engineer_features(df)
        assert result["dep_hour"].iloc[0] == 14

    def test_2359_gives_hour_23(self):
        df = make_raw_row(CRS_DEP_TIME=2359)
        result = engineer_features(df)
        assert result["dep_hour"].iloc[0] == 23

    def test_0000_gives_hour_0(self):
        df = make_raw_row(CRS_DEP_TIME=0)
        result = engineer_features(df)
        assert result["dep_hour"].iloc[0] == 0


# ════════════════════════════════════════════════════════
# TIME BUCKET
# ════════════════════════════════════════════════════════

class TestTimeBucket:
    @pytest.mark.parametrize("hour,expected", [
        (0,  0),   # red-eye
        (3,  0),   # red-eye
        (5,  0),   # red-eye boundary
        (6,  1),   # morning
        (11, 1),   # morning
        (12, 2),   # afternoon
        (17, 2),   # afternoon
        (18, 3),   # evening
        (23, 3),   # evening
    ])
    def test_time_bucket(self, hour, expected):
        df = make_raw_row(CRS_DEP_TIME=hour*100)
        result = engineer_features(df)
        assert result["dep_time_bucket"].iloc[0] == expected, (
            f"Hour {hour} should be bucket {expected}"
        )


# ════════════════════════════════════════════════════════
# WEEKEND FLAG
# ════════════════════════════════════════════════════════

class TestWeekendFlag:
    def test_saturday_is_weekend(self):
        # 2024-02-17 is Saturday
        df = make_raw_row(FL_DATE="2024-02-17")
        result = engineer_features(df)
        assert result["is_weekend"].iloc[0] == 1

    def test_monday_is_not_weekend(self):
        # 2024-02-19 is Monday
        df = make_raw_row(FL_DATE="2024-02-19")
        result = engineer_features(df)
        assert result["is_weekend"].iloc[0] == 0


# ════════════════════════════════════════════════════════
# HOLIDAY SEASON
# ════════════════════════════════════════════════════════

class TestHolidaySeason:
    @pytest.mark.parametrize("month,expected", [
        (1,  0), (2,  0), (3,  0), (4,  0), (5,  0),
        (6,  1), (7,  1), (8,  0), (9,  0), (10, 0),
        (11, 1), (12, 1),
    ])
    def test_holiday_season_months(self, month, expected):
        date = f"2024-{month:02d}-15"
        df = make_raw_row(FL_DATE=date)
        result = engineer_features(df)
        assert result["is_holiday_season"].iloc[0] == expected


# ════════════════════════════════════════════════════════
# HUB FLAGS
# ════════════════════════════════════════════════════════

class TestHubFlags:
    def test_known_hub_origin_is_1(self):
        for hub in ["ATL", "ORD", "JFK", "LAX"]:
            df = make_raw_row(ORIGIN=hub)
            result = engineer_features(df)
            assert result["is_hub_origin"].iloc[0] == 1, f"{hub} should be a hub"

    def test_non_hub_origin_is_0(self):
        df = make_raw_row(ORIGIN="BZN")   # Bozeman, MT — not a hub
        result = engineer_features(df)
        assert result["is_hub_origin"].iloc[0] == 0

    def test_hub_dest(self):
        df = make_raw_row(DEST="DFW")
        result = engineer_features(df)
        assert result["is_hub_dest"].iloc[0] == 1


# ════════════════════════════════════════════════════════
# DISTANCE CONVERSION
# ════════════════════════════════════════════════════════

class TestDistanceConversion:
    def test_1000_miles_to_km(self):
        df = make_raw_row(DISTANCE=1000)
        result = engineer_features(df)
        expected = round(1000 * 1.60934, 1)
        assert result["route_distance_km"].iloc[0] == pytest.approx(expected, abs=0.1)


# ════════════════════════════════════════════════════════
# OUTPUT CONTRACT
# ════════════════════════════════════════════════════════

class TestEngineerFeaturesContract:
    def test_all_feature_columns_present(self):
        df = make_raw_row()
        result = engineer_features(df)
        for col in FEATURE_COLS:
            assert col in result.columns, f"Missing: {col}"

    def test_no_nulls_in_features(self):
        df = make_raw_df(n=100)
        result = engineer_features(df)
        null_counts = result[FEATURE_COLS].isnull().sum()
        assert null_counts.sum() == 0, f"Nulls found: {null_counts[null_counts>0]}"

    def test_row_count_preserved(self):
        df = make_raw_df(n=50)
        result = engineer_features(df)
        assert len(result) == 50

    def test_input_not_modified(self):
        df = make_raw_row()
        original_cols = set(df.columns)
        _ = engineer_features(df)
        assert set(df.columns) == original_cols


# ════════════════════════════════════════════════════════
# VALIDATION
# ════════════════════════════════════════════════════════

class TestValidateRawData:
    def test_valid_data_passes(self):
        df = make_raw_df(n=200)
        validate_raw_data(df)  # should not raise

    def test_missing_column_raises(self):
        df = make_raw_df(n=200).drop(columns=["OP_CARRIER"])
        with pytest.raises(ValueError, match="Missing"):
            validate_raw_data(df)

    def test_too_few_rows_raises(self):
        df = make_raw_df(n=5)
        with pytest.raises(ValueError, match="too small"):
            validate_raw_data(df)

    def test_invalid_dep_time_raises(self):
        df = make_raw_df(n=200)
        df.loc[0, "CRS_DEP_TIME"] = 2500  # invalid: > 2359
        with pytest.raises(ValueError, match="CRS_DEP_TIME"):
            validate_raw_data(df)
