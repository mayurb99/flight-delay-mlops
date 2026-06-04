"""
sagemaker/pipeline.py
════════════════════════════════════════════════════════
Flight Delay Prediction — SageMaker Pipeline Definition

Project Lecture 1: Foundation
Called by: .github/workflows/train.yml

4-step pipeline:
  Step 1: ProcessingStep  — preprocessing.py
  Step 2: TrainingStep    — train.py
  Step 3: ProcessingStep  — evaluate.py
  Step 4: ConditionStep   — register if challenger beats champion

Run locally (for testing):
  python sagemaker/pipeline.py --dry-run

Run in CI (triggered by GitHub Actions train.yml):
  python sagemaker/pipeline.py
════════════════════════════════════════════════════════
"""

import os
import sys
import json
import time
import boto3
import logging
import argparse
import requests

import sagemaker
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.parameters import (
    ParameterString, ParameterInteger, ParameterFloat,
)
from sagemaker.workflow.steps import (
    ProcessingStep, TrainingStep, CacheConfig,
)
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo
from sagemaker.workflow.functions import JsonGet, Join
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.step_collections import RegisterModel
from sagemaker.sklearn.estimator import SKLearn
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.inputs import TrainingInput
from sagemaker.model_metrics import MetricsSource, ModelMetrics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ── Configuration from environment (set in GitHub Actions) ─
REGION         = os.environ.get("AWS_REGION",         "us-east-1")
ROLE_ARN       = os.environ.get("SAGEMAKER_ROLE_ARN",  "")
S3_BUCKET      = os.environ.get("S3_BUCKET",           "")
S3_DATA_KEY    = os.environ.get("S3_DATA_KEY",         "data/raw/flights.csv")
PIPELINE_NAME  = "flight-delay-training-pipeline"
MODEL_GROUP    = "flight-delay-model-group"
GITHUB_OWNER   = os.environ.get("GH_OWNER",            "")
GITHUB_REPO    = os.environ.get("GH_REPO",             "")
GITHUB_PAT     = os.environ.get("GH_PAT",              "")

# ── Instance types ──────────────────────────────────────────
PROCESSING_INSTANCE = "ml.t3.xlarge"
TRAINING_INSTANCE   = "ml.m5.xlarge"


def create_pipeline(sm_session: sagemaker.Session, dry_run: bool = False) -> Pipeline:
    """Build and return the SageMaker Pipeline definition."""

    # ── Pipeline Parameters ────────────────────────────────
    input_data_uri = ParameterString(
        name="InputDataUri",
        default_value=f"s3://{S3_BUCKET}/{S3_DATA_KEY}",
    )
    n_estimators = ParameterInteger(name="NEstimators",  default_value=200)
    max_depth    = ParameterInteger(name="MaxDepth",      default_value=5)
    lr           = ParameterFloat(  name="LearningRate",  default_value=0.08)

    # ── Cache config (skip if same inputs) ────────────────
    cache_config = CacheConfig(enable_caching=True, expire_after="PT24H")

    # ── STEP 1: ProcessingStep — preprocess raw data ───────
    preprocessor = SKLearnProcessor(
        framework_version="1.2-1",
        role=ROLE_ARN,
        instance_type=PROCESSING_INSTANCE,
        instance_count=1,
        sagemaker_session=sm_session,
        env={
            "MLFLOW_TRACKING_URI":      os.environ.get("MLFLOW_TRACKING_URI", ""),
            "MLFLOW_TRACKING_USERNAME": os.environ.get("MLFLOW_TRACKING_USERNAME", ""),
            "MLFLOW_TRACKING_PASSWORD": os.environ.get("MLFLOW_TRACKING_PASSWORD", ""),
        },
    )

    preprocessing_step = ProcessingStep(
        name="preprocess-flight-data",
        processor=preprocessor,
        code="src/preprocessing.py",
        inputs=[
            ProcessingInput(
                source="src/features.py",
                destination="/opt/ml/processing/input/deps",
                input_name="deps",
            ),
            ProcessingInput(
                source=input_data_uri,
                destination="/opt/ml/processing/input/raw",
            ),
        ],
        outputs=[
            ProcessingOutput(output_name="train",     source="/opt/ml/processing/output/train"),
            ProcessingOutput(output_name="val",       source="/opt/ml/processing/output/val"),
            ProcessingOutput(output_name="test",      source="/opt/ml/processing/output/test"),
            ProcessingOutput(output_name="reference", source="/opt/ml/processing/output/reference"),
        ],
        cache_config=cache_config,
    )

    # ── STEP 2: TrainingStep ────────────────────────────────
    estimator = SKLearn(
        entry_point          = "src/train.py",
        source_dir           = ".",
        role                 = ROLE_ARN,
        instance_type        = TRAINING_INSTANCE,
        framework_version    = "1.2-1",
        sagemaker_session    = sm_session,
        use_spot_instances   = True,
        max_run              = 3600,   # max 1 hour actual training time
        max_wait             = 7200,   # max 2 hours total including spot wait
        hyperparameters   = {
            "n-estimators":  n_estimators,
            "max-depth":     max_depth,
            "learning-rate": lr,
        },
        environment = {
            "MLFLOW_TRACKING_URI":      os.environ.get("MLFLOW_TRACKING_URI", ""),
            "MLFLOW_TRACKING_USERNAME": os.environ.get("MLFLOW_TRACKING_USERNAME", ""),
            "MLFLOW_TRACKING_PASSWORD": os.environ.get("MLFLOW_TRACKING_PASSWORD", ""),
            "S3_DATA_KEY":              S3_DATA_KEY,
        },
    )

    training_step = TrainingStep(
        name="train-flight-delay-model",
        estimator=estimator,
        inputs={
            "train": TrainingInput(
                s3_data=preprocessing_step.properties
                    .ProcessingOutputConfig.Outputs["train"].S3Output.S3Uri,
            ),
            "val": TrainingInput(
                s3_data=preprocessing_step.properties
                    .ProcessingOutputConfig.Outputs["val"].S3Output.S3Uri,
            ),
        },
    )

    # ── STEP 3: ProcessingStep — evaluate vs champion ──────
    evaluator = SKLearnProcessor(
        framework_version="1.2-1",
        role=ROLE_ARN,
        instance_type=PROCESSING_INSTANCE,
        instance_count=1,
        sagemaker_session=sm_session,
        env={
            "MLFLOW_TRACKING_URI":      os.environ.get("MLFLOW_TRACKING_URI", ""),
            "MLFLOW_TRACKING_USERNAME": os.environ.get("MLFLOW_TRACKING_USERNAME", ""),
            "MLFLOW_TRACKING_PASSWORD": os.environ.get("MLFLOW_TRACKING_PASSWORD", ""),
        },
    )

    eval_report = PropertyFile(
        name="EvalReport",
        output_name="eval",
        path="comparison.json",
    )

    evaluation_step = ProcessingStep(
        name="evaluate-vs-champion",
        processor=evaluator,
        code="src/evaluate.py",
        inputs=[
            ProcessingInput(
                source="src/features.py",
                destination="/opt/ml/processing/input/deps",
                input_name="deps",
            ),
            ProcessingInput(
                source=training_step.properties.ModelArtifacts.S3ModelArtifacts,
                destination="/opt/ml/processing/input/model",
            ),
            ProcessingInput(
                source=preprocessing_step.properties
                    .ProcessingOutputConfig.Outputs["val"].S3Output.S3Uri,
                destination="/opt/ml/processing/input/val",
            ),
        ],
        outputs=[
            ProcessingOutput(output_name="eval", source="/opt/ml/processing/output/eval"),
        ],
        property_files=[eval_report],
    )

    # ── STEP 4: ConditionStep + RegisterModel ──────────────
    beats_champion = JsonGet(
        step_name=evaluation_step.name,
        property_file=eval_report,
        json_path="metrics.challenger_beats_champion.value",
    )

    model_metrics = ModelMetrics(
        model_statistics=MetricsSource(
            s3_uri=Join(
                on="/",
                values=[
                    evaluation_step.properties
                        .ProcessingOutputConfig.Outputs["eval"].S3Output.S3Uri,
                    "comparison.json",
                ],
            ),
            content_type="application/json",
        ),
    )

    register_step = RegisterModel(
        name="register-challenger",
        estimator=estimator,
        model_data=training_step.properties.ModelArtifacts.S3ModelArtifacts,
        content_types=["application/json", "text/csv"],
        response_types=["application/json"],
        inference_instances=["ml.t2.medium", "ml.m5.large", "ml.m5.xlarge"],
        transform_instances=["ml.m5.xlarge"],
        model_package_group_name=MODEL_GROUP,
        approval_status="PendingManualApproval",
        model_metrics=model_metrics,
    )

    condition_step = ConditionStep(
        name="check-beats-champion",
        conditions=[ConditionGreaterThanOrEqualTo(
            left=beats_champion,
            right=0.5,   # 1.0 = beats champion, 0.0 = does not
        )],
        if_steps=[register_step],
        else_steps=[],
    )

    # ── Assemble Pipeline ──────────────────────────────────
    pipeline = Pipeline(
        name=PIPELINE_NAME,
        parameters=[input_data_uri, n_estimators, max_depth, lr],
        steps=[
            preprocessing_step,
            training_step,
            evaluation_step,
            condition_step,
        ],
        sagemaker_session=sm_session,
    )

    return pipeline


def open_github_issue(run_id: str, metrics: dict):
    """Open a GitHub issue when challenger is registered."""
    if not all([GITHUB_OWNER, GITHUB_REPO, GITHUB_PAT]):
        logger.info("GitHub issue creation skipped — GH_* env vars not set")
        return

    body = f"""## 🛫 New Flight Delay Model Ready for Review

A new challenger model has been trained and registered.
**It beats the current champion and is awaiting approval.**

| Metric | Value |
|--------|-------|
| Val F1 | `{metrics.get('val_f1', 'N/A')}` |
| Val AUC-ROC | `{metrics.get('val_auc_roc', 'N/A')}` |
| Val Accuracy | `{metrics.get('val_accuracy', 'N/A')}` |

**MLflow Run:** `{run_id}`

### To Deploy
1. Go to **AWS Console → SageMaker → Model Registry**
2. Find `{MODEL_GROUP}` → find `PendingManualApproval` version
3. Click **Actions → Approve**
4. Deployment starts automatically via Lambda + GitHub Actions

### To Reject
Click **Actions → Reject** — the current champion stays deployed.
"""
    response = requests.post(
        f"https://api.github.com/repos/{GITHUB_OWNER}/{GITHUB_REPO}/issues",
        headers={
            "Authorization": f"Bearer {GITHUB_PAT}",
            "Accept": "application/vnd.github+json",
        },
        json={
            "title": "🛫 New Flight Delay Model: Challenger Ready for Approval",
            "body": body,
            "labels": ["model-ready", "awaiting-approval"],
        },
    )
    if response.status_code == 201:
        issue_url = response.json()["html_url"]
        logger.info(f"✓ GitHub issue opened: {issue_url}")
    else:
        logger.warning(f"Failed to open issue: {response.status_code} {response.text}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Print pipeline definition without running")
    args = parser.parse_args()

    if not ROLE_ARN or not S3_BUCKET:
        logger.error("SAGEMAKER_ROLE_ARN and S3_BUCKET must be set in environment")
        sys.exit(1)

    boto_session = boto3.Session(region_name=REGION)
    sm_session   = sagemaker.Session(boto_session=boto_session)

    logger.info(f"Building pipeline: {PIPELINE_NAME}")
    pipeline = create_pipeline(sm_session, dry_run=args.dry_run)

    if args.dry_run:
        definition = json.loads(pipeline.definition())
        logger.info("Pipeline definition (dry run):")
        logger.info(json.dumps(definition, indent=2)[:2000] + "...")
        return

    # Upsert and start
    logger.info("Upserting pipeline to SageMaker...")
    pipeline.upsert(role_arn=ROLE_ARN)
    logger.info("✓ Pipeline upserted")

    logger.info("Starting execution...")
    execution = pipeline.start(
        parameters={
            "InputDataUri": f"s3://{S3_BUCKET}/{S3_DATA_KEY}",
        }
    )
    logger.info(f"✓ Execution ARN: {execution.arn}")

    # Wait for completion (GitHub Actions will wait here)
    logger.info("Waiting for pipeline to complete...")
    execution.wait(delay=30, max_attempts=120)   # max 60 min

    status = execution.describe()["PipelineExecutionStatus"]
    logger.info(f"Pipeline status: {status}")

    if status == "Failed":
        steps = boto3.client("sagemaker", region_name=REGION)\
            .list_pipeline_execution_steps(PipelineExecutionArn=execution.arn)
        for step in steps["PipelineExecutionSteps"]:
            if step["StepStatus"] == "Failed":
                logger.error(f"Failed step: {step['StepName']}")
                logger.error(f"Reason: {step.get('FailureReason', 'unknown')}")
        sys.exit(1)

    logger.info("✓ Pipeline completed successfully!")


if __name__ == "__main__":
    main()
