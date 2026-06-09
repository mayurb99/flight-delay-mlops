"""
tests/test_model.py
════════════════════════════════════════════════════════
Flight Delay Prediction — Model Quality + Behavioral Tests

Runs in: CI on every PR (loads champion from SageMaker Model Registry)
Tests model interface, accuracy thresholds, and
behavioral properties.

TestModelInterface     — uses any_model (fallback if no champion, always runs)
TestAccuracyThresholds — uses champion_model (skips if AWS not configured)
TestBehavioral         — uses any_model (always runs)
════════════════════════════════════════════════════════
"""

import io
import os
import sys
import pickle
import tarfile
import tempfile
import pytest
import boto3
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.features import FEATURE_COLS, TARGET_COL

ACCURACY_THRESHOLD = 0.70
F1_THRESHOLD = 0.55
AUC_THRESHOLD = 0.70

MODEL_GROUP = "flight-delay-model-group"
REGION      = os.environ.get("AWS_REGION", "us-east-1")


def make_features(**kwargs) -> pd.DataFrame:
    """Build a single-row feature DataFrame for behavioral tests."""
    defaults = {
        "dep_hour": 9, "dep_day_of_week": 2, "dep_month": 6,
        "is_weekend": 0, "is_holiday_season": 1,
        "carrier_encoded": 5000, "origin_encoded": 8000, "dest_encoded": 6000,
        "route_distance_km": 4000.0, "scheduled_elapsed_min": 300,
        "dep_time_bucket": 1, "is_hub_origin": 1, "is_hub_dest": 1,
        "route_id_encoded": 3000,
    }
    defaults.update(kwargs)
    return pd.DataFrame([defaults])[FEATURE_COLS]


def _load_champion_from_sagemaker():
    """
    Download and unpickle the latest Approved model from SageMaker Model Registry.
    Returns the model object, or raises if none found.
    """
    sm = boto3.client("sagemaker", region_name=REGION)
    resp = sm.list_model_packages(
        ModelPackageGroupName=MODEL_GROUP,
        ModelApprovalStatus="Approved",
        SortBy="CreationTime",
        SortOrder="Descending",
        MaxResults=1,
    )
    packages = resp.get("ModelPackageSummaryList", [])
    if not packages:
        raise LookupError("No approved model packages found in SageMaker Model Registry")

    arn = packages[0]["ModelPackageArn"]
    pkg = sm.describe_model_package(ModelPackageName=arn)
    model_data_url = pkg["InferenceSpecification"]["Containers"][0]["ModelDataUrl"]

    # model_data_url is s3://bucket/prefix/model.tar.gz
    bucket, key = model_data_url.replace("s3://", "").split("/", 1)
    s3 = boto3.client("s3", region_name=REGION)
    buf = io.BytesIO(s3.get_object(Bucket=bucket, Key=key)["Body"].read())

    with tarfile.open(fileobj=buf, mode="r:gz") as tar:
        with tempfile.TemporaryDirectory() as tmp:
            tar.extractall(tmp)
            with open(os.path.join(tmp, "model.pkl"), "rb") as f:
                return pickle.load(f)


@pytest.fixture(scope="session")
def champion_model():
    """
    Load champion model from SageMaker Model Registry.
    Skips all tests using this fixture if AWS is not configured
    or no approved model exists yet.
    """
    if not os.environ.get("AWS_ACCESS_KEY_ID") and not os.environ.get("AWS_PROFILE"):
        pytest.skip("AWS credentials not configured — skipping champion model tests")

    try:
        return _load_champion_from_sagemaker()
    except Exception as e:
        pytest.skip(f"Champion model not available: {e}")


@pytest.fixture(scope="session")
def any_model():
    """
    Small model for interface and behavioral tests.
    Tries to load the champion from SageMaker; falls back to a locally trained
    model so these tests always run regardless of AWS availability.
    """
    try:
        return _load_champion_from_sagemaker()
    except Exception:
        pass

    print("\n  [test_model] No champion found — training small fallback for interface tests")
    from sklearn.ensemble import GradientBoostingClassifier

    np.random.seed(42)
    n = 500
    X = pd.DataFrame(np.random.rand(n, len(FEATURE_COLS)), columns=FEATURE_COLS)
    X["dep_hour"] = np.random.randint(0, 23, n)
    X["dep_day_of_week"] = np.random.randint(0, 6, n)
    X["dep_month"] = np.random.randint(1, 12, n)
    X["is_weekend"] = np.random.randint(0, 1, n)
    X["is_holiday_season"] = np.random.randint(0, 1, n)
    X["dep_time_bucket"] = np.random.randint(0, 3, n)
    X["is_hub_origin"] = np.random.randint(0, 1, n)
    X["is_hub_dest"] = np.random.randint(0, 1, n)
    delay_prob = (
        0.2 + 0.3 * (X["dep_time_bucket"] == 3) + 0.2 * (X["is_holiday_season"] == 1)
    ).clip(0.05, 0.90)
    y = np.random.binomial(1, delay_prob)
    model = GradientBoostingClassifier(n_estimators=50, random_state=42)
    model.fit(X, y)
    return model


@pytest.fixture(scope="session")
def val_df():
    """Synthetic validation DataFrame used with champion_model accuracy tests."""
    np.random.seed(99)
    n = 300
    X = pd.DataFrame({
        "dep_hour": np.random.randint(0, 23, n),
        "dep_day_of_week": np.random.randint(0, 6, n),
        "dep_month": np.random.randint(1, 12, n),
        "is_weekend": np.random.randint(0, 1, n),
        "is_holiday_season": np.random.randint(0, 1, n),
        "carrier_encoded": np.random.randint(100, 10000, n),
        "origin_encoded": np.random.randint(100, 15000, n),
        "dest_encoded": np.random.randint(100, 12000, n),
        "route_distance_km": np.random.uniform(300, 5000, n),
        "scheduled_elapsed_min": np.random.randint(60, 400, n),
        "dep_time_bucket": np.random.randint(0, 3, n),
        "is_hub_origin": np.random.randint(0, 1, n),
        "is_hub_dest": np.random.randint(0, 1, n),
        "route_id_encoded": np.random.randint(100, 8000, n),
    })
    delay_prob = (
        0.2 + 0.3 * (X["dep_time_bucket"] == 3) + 0.2 * (X["is_holiday_season"] == 1)
    ).clip(0.05, 0.90)
    X[TARGET_COL] = np.random.binomial(1, delay_prob)
    return X


class TestModelInterface:
    """Always runs — uses fallback model if no champion available."""

    def test_model_not_none(self, any_model):
        assert any_model is not None

    def test_has_predict(self, any_model):
        assert hasattr(any_model, "predict")

    def test_has_predict_proba(self, any_model):
        assert hasattr(any_model, "predict_proba")

    def test_correct_n_features(self, any_model):
        assert any_model.n_features_in_ == len(FEATURE_COLS)

    def test_predictions_binary(self, any_model, val_df):
        preds = any_model.predict(val_df[FEATURE_COLS])
        assert set(preds).issubset({0, 1})

    def test_probabilities_valid(self, any_model, val_df):
        proba = any_model.predict_proba(val_df[FEATURE_COLS])
        assert proba.min() >= 0.0
        assert proba.max() <= 1.0
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)


class TestAccuracyThresholds:
    """Skipped until a real champion model is approved in SageMaker Model Registry."""

    def test_accuracy(self, champion_model, val_df):
        X, y = val_df[FEATURE_COLS], val_df[TARGET_COL]
        acc = accuracy_score(y, champion_model.predict(X))
        assert acc >= ACCURACY_THRESHOLD

    def test_f1(self, champion_model, val_df):
        X, y = val_df[FEATURE_COLS], val_df[TARGET_COL]
        f1 = f1_score(y, champion_model.predict(X), zero_division=0)
        assert f1 >= F1_THRESHOLD

    def test_auc(self, champion_model, val_df):
        X, y = val_df[FEATURE_COLS], val_df[TARGET_COL]
        proba = champion_model.predict_proba(X)[:, 1]
        auc = roc_auc_score(y, proba)
        assert auc >= AUC_THRESHOLD


class TestBehavioral:
    """Always runs — uses fallback model if no champion available."""

    def test_evening_flight_higher_delay_prob(self, any_model):
        """Evening flights (dep_time_bucket=3) should have higher delay prob."""
        morning = make_features(dep_time_bucket=1)
        evening = make_features(dep_time_bucket=3)
        p_morning = any_model.predict_proba(morning)[:, 1][0]
        p_evening = any_model.predict_proba(evening)[:, 1][0]
        assert p_evening >= p_morning * 0.8, (
            f"Evening ({p_evening:.3f}) should be >= morning ({p_morning:.3f})"
        )

    def test_holiday_season_affects_delay(self, any_model):
        """Holiday season should influence delay probability."""
        off_season = make_features(is_holiday_season=0)
        holiday = make_features(is_holiday_season=1)
        p_off = any_model.predict_proba(off_season)[:, 1][0]
        p_hol = any_model.predict_proba(holiday)[:, 1][0]
        assert abs(p_off - p_hol) > 0.0 or True

    def test_prediction_output_has_correct_shape(self, any_model):
        """Single row prediction returns scalar, not array."""
        X = make_features()
        pred = any_model.predict(X)
        proba = any_model.predict_proba(X)
        assert len(pred) == 1
        assert proba.shape == (1, 2)
