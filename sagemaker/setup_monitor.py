"""
sagemaker/setup_monitor.py
════════════════════════════════════════════════════════
Flight Delay Prediction — SageMaker Model Monitor Setup

Project Lecture 2: Approval + Blue/Green Deploy
Called by: .github/workflows/deploy.yml after successful deployment

What this does:
  1. Baseline job  — profiles reference.csv to create statistics.json
                     and constraints.json (the definition of "normal")
  2. DataCapture   — updates endpoint to capture all predictions to S3
  3. Monitor schedule — daily job comparing captured traffic vs baseline
  4. CloudWatch alarm — fires when violations exceed threshold

Run from deploy.yml:
  python sagemaker/setup_monitor.py \
    --endpoint-name flight-delay-endpoint \
    --reference-uri s3://bucket/reference/reference.csv
════════════════════════════════════════════════════════
"""

import os
import sys
import boto3
import logging
import argparse
import sagemaker
from sagemaker.model_monitor import DefaultModelMonitor
from sagemaker.model_monitor.dataset_format import DatasetFormat
from sagemaker.inputs import CreateModelInput
from sagemaker.predictor import Predictor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ── Configuration ──────────────────────────────────────
REGION         = os.environ.get("AWS_REGION",        "us-east-1")
ROLE_ARN       = os.environ.get("SAGEMAKER_ROLE_ARN", "")
S3_BUCKET      = os.environ.get("S3_BUCKET",          "")
ENDPOINT_NAME  = os.environ.get("ENDPOINT_NAME",      "flight-delay-endpoint")


def baseline_exists(s3_client, baseline_output_uri: str) -> bool:
    """Check if baseline statistics.json already exists in S3."""
    path = baseline_output_uri.replace("s3://", "").split("/", 1)
    bucket, prefix = path[0], path[1].rstrip("/") + "/statistics.json"
    try:
        s3_client.head_object(Bucket=bucket, Key=prefix)
        return True
    except Exception:
        return False


def run_baseline_job(
    monitor: DefaultModelMonitor,
    reference_s3_uri: str,
    baseline_output_uri: str,
    s3_client=None,
) -> None:
    """
    Phase 1: Run a baseline job once on the training reference data.

    This profiles reference.csv and creates two files in S3:
      statistics.json  — mean, std, quantiles per feature column
      constraints.json — the rules: age must be 18-100, charge must be non-null

    These files define what "normal" looks like for this model.
    All future monitoring runs compare against this baseline.

    Skips if baseline already exists. Submits async (wait=False) so CI
    doesn't block — the job runs in SageMaker in the background.
    """
    if s3_client and baseline_exists(s3_client, baseline_output_uri):
        logger.info("✓ Baseline already exists — skipping baseline job")
        logger.info(f"  {baseline_output_uri}/statistics.json")
        return

    logger.info("Phase 1: Submitting baseline job (async)...")
    logger.info(f"  Reference data: {reference_s3_uri}")
    logger.info(f"  Baseline output: {baseline_output_uri}")

    monitor.suggest_baseline(
        baseline_dataset=reference_s3_uri,
        dataset_format=DatasetFormat.csv(header=True),
        output_s3_uri=baseline_output_uri,
        wait=False,
        logs=False,
    )
    logger.info("✓ Baseline job submitted (running in background)")
    logger.info(f"  Results will appear at: {baseline_output_uri}/statistics.json")


def update_endpoint_data_capture(
    sm_client,
    endpoint_name: str,
    capture_s3_uri: str,
) -> None:
    """
    Phase 2: Enable DataCapture on the existing endpoint.

    Every prediction request and response is captured to S3
    automatically. 100% sampling. No compute cost — only S3 storage.

    This updates the existing endpoint config by creating a new one
    with DataCaptureConfig enabled and updating the endpoint.

    Parameters
    ----------
    sm_client       : boto3 sagemaker client
    endpoint_name   : name of the running SageMaker endpoint
    capture_s3_uri  : S3 URI where captured data is written
    """
    logger.info("Phase 2: Enabling DataCapture on endpoint...")

    # Get current endpoint config name
    endpoint = sm_client.describe_endpoint(EndpointName=endpoint_name)
    current_config_name = endpoint["EndpointConfigName"]
    logger.info(f"  Current config: {current_config_name}")

    # Get current config details
    current_config = sm_client.describe_endpoint_config(
        EndpointConfigName=current_config_name
    )
    production_variants = current_config["ProductionVariants"]

    # Create new config name with -monitored suffix
    new_config_name = f"{current_config_name}-monitored"

    # Create new endpoint config with DataCapture enabled
    sm_client.create_endpoint_config(
        EndpointConfigName=new_config_name,
        ProductionVariants=production_variants,
        DataCaptureConfig={
            "EnableCapture":          True,
            "InitialSamplingPercentage": 100,
            "DestinationS3Uri":       capture_s3_uri,
            "CaptureOptions": [
                {"CaptureMode": "Input"},
                {"CaptureMode": "Output"},
            ],
            "CaptureContentTypeHeader": {
                "JsonContentTypes":  ["application/json"],
                "CsvContentTypes":   ["text/csv"],
            },
        },
    )

    # Update endpoint to use new config (in-place update, no downtime)
    sm_client.update_endpoint(
        EndpointName=endpoint_name,
        EndpointConfigName=new_config_name,
    )

    # Wait for update to complete
    logger.info("  Waiting for endpoint update...")
    waiter = sm_client.get_waiter("endpoint_in_service")
    waiter.wait(
        EndpointName=endpoint_name,
        WaiterConfig={"Delay": 15, "MaxAttempts": 40},
    )
    logger.info("✓ DataCapture enabled on endpoint")
    logger.info(f"  Captured data → {capture_s3_uri}")


def create_monitoring_schedule(
    monitor: DefaultModelMonitor,
    endpoint_name: str,
    baseline_output_uri: str,
    reports_s3_uri: str,
) -> None:
    """
    Phase 3: Create a daily monitoring schedule.

    SageMaker runs a processing job every day at midnight UTC.
    The job downloads captured traffic from DataCapture S3 output,
    compares it against the baseline statistics and constraints,
    and writes a violations report to S3.

    CloudWatch metrics are published automatically:
      - feature_baseline_drift_*   per feature column
      - missing_values_*           per column
      - data_type_errors_*         per column

    Parameters
    ----------
    monitor             : DefaultModelMonitor instance
    endpoint_name       : endpoint to monitor
    baseline_output_uri : S3 URI containing statistics.json and constraints.json
    reports_s3_uri      : S3 URI where violations reports are written
    """
    logger.info("Phase 3: Creating daily monitoring schedule...")

    # Delete existing schedule if it exists (idempotent)
    schedule_name = f"{endpoint_name}-data-quality-monitor"
    try:
        monitor.delete_monitoring_schedule()
        logger.info("  Deleted existing monitoring schedule")
    except Exception:
        pass  # No existing schedule — fine

    monitor.create_monitoring_schedule(
        monitor_schedule_name=schedule_name,
        endpoint_input=endpoint_name,
        output_s3_uri=reports_s3_uri,
        statistics=f"{baseline_output_uri}/statistics.json",
        constraints=f"{baseline_output_uri}/constraints.json",
        schedule_cron_expression="cron(0 0 ? * * *)",  # daily at midnight UTC
        enable_cloudwatch_metrics=True,
    )
    logger.info("✓ Daily monitoring schedule created")
    logger.info(f"  Schedule name : {schedule_name}")
    logger.info(f"  Reports → {reports_s3_uri}")


def create_cloudwatch_alarm(
    cw_client,
    endpoint_name: str,
) -> None:
    """
    Create a CloudWatch alarm that fires when the monitor detects violations.

    The alarm publishes to an SNS topic which is wired to
    lambda/cloudwatch_to_github.py in P4 (triggers monitor.yml).

    For now: alarm just fires to SNS → email notification.
    P4 adds the Lambda trigger to automate retraining.

    Parameters
    ----------
    cw_client     : boto3 cloudwatch client
    endpoint_name : endpoint being monitored
    """
    logger.info("Creating CloudWatch alarm for monitoring violations...")

    alarm_name = f"{endpoint_name}-monitor-violations"
    metric_name = "feature_baseline_drift_count"

    try:
        cw_client.put_metric_alarm(
            AlarmName=alarm_name,
            AlarmDescription=(
                f"SageMaker Model Monitor detected baseline drift "
                f"violations on endpoint {endpoint_name}"
            ),
            MetricName=metric_name,
            Namespace="aws/sagemaker/Endpoints/data-metrics",
            Dimensions=[
                {"Name": "Endpoint", "Value": endpoint_name},
                {"Name": "MonitoringSchedule", "Value": f"{endpoint_name}-data-quality-monitor"},
            ],
            Statistic="Sum",
            Period=86400,        # 24 hours
            EvaluationPeriods=1,
            Threshold=1,
            ComparisonOperator="GreaterThanOrEqualToThreshold",
            TreatMissingData="notBreaching",
        )
        logger.info(f"✓ CloudWatch alarm created: {alarm_name}")
        logger.info(f"  Fires when: baseline_drift_count >= 1 over 24h")
        logger.info(f"  Note: Wire to SNS + Lambda in P4 for auto-retraining")
    except Exception as e:
        logger.warning(f"CloudWatch alarm creation warning (non-fatal): {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--endpoint-name",
        default=ENDPOINT_NAME,
        help="SageMaker endpoint name to setup monitoring for",
    )
    parser.add_argument(
        "--reference-uri",
        required=True,
        help="S3 URI of reference.csv from preprocessing step",
    )
    args = parser.parse_args()

    if not ROLE_ARN or not S3_BUCKET:
        logger.error("SAGEMAKER_ROLE_ARN and S3_BUCKET must be set")
        sys.exit(1)

    endpoint_name = args.endpoint_name
    reference_uri = args.reference_uri

    # S3 paths for monitor outputs
    baseline_output_uri = f"s3://{S3_BUCKET}/monitoring/baseline/{endpoint_name}"
    capture_s3_uri      = f"s3://{S3_BUCKET}/monitoring/datacapture/{endpoint_name}"
    reports_s3_uri      = f"s3://{S3_BUCKET}/monitoring/reports/{endpoint_name}"

    logger.info("=" * 60)
    logger.info("setup_monitor.py — SageMaker Model Monitor Setup")
    logger.info(f"  Endpoint  : {endpoint_name}")
    logger.info(f"  Reference : {reference_uri}")
    logger.info(f"  Baseline  : {baseline_output_uri}")
    logger.info(f"  Capture   : {capture_s3_uri}")
    logger.info("=" * 60)

    boto_session = boto3.Session(region_name=REGION)
    sm_session   = sagemaker.Session(boto_session=boto_session)
    sm_client    = boto3.client("sagemaker",    region_name=REGION)
    s3_client    = boto3.client("s3",           region_name=REGION)
    cw_client    = boto3.client("cloudwatch",   region_name=REGION)

    monitor = DefaultModelMonitor(
        role=ROLE_ARN,
        instance_type="ml.m5.xlarge",
        instance_count=1,
        max_runtime_in_seconds=1800,
        sagemaker_session=sm_session,
    )

    # Phase 1: Baseline (async — skips if already exists)
    run_baseline_job(monitor, reference_uri, baseline_output_uri, s3_client=s3_client)

    # Phase 2: DataCapture
    update_endpoint_data_capture(sm_client, endpoint_name, capture_s3_uri)

    # Phase 3: Schedule
    create_monitoring_schedule(
        monitor, endpoint_name, baseline_output_uri, reports_s3_uri
    )

    # CloudWatch alarm
    create_cloudwatch_alarm(cw_client, endpoint_name)

    logger.info("=" * 60)
    logger.info("✓ SageMaker Model Monitor setup complete!")
    logger.info(f"  Baseline stored at    : {baseline_output_uri}")
    logger.info(f"  Predictions captured  : {capture_s3_uri}")
    logger.info(f"  Daily reports at      : {reports_s3_uri}")
    logger.info(f"  CloudWatch alarm      : {endpoint_name}-monitor-violations")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
