"""
lambda/s3_to_github.py
════════════════════════════════════════════════════════
Flight Delay Prediction — S3 Upload → GitHub Actions Trigger

Project Lecture 1: Foundation
Deployed as: AWS Lambda function
Triggered by: EventBridge rule watching S3 object creation

What it does:
  1. EventBridge fires when new file lands in s3://bucket/data/raw/
  2. This Lambda receives the event
  3. Extracts bucket name and object key
  4. Calls GitHub Actions REST API to trigger train.yml
  5. Passes s3_bucket and s3_key as workflow inputs

Environment variables (set in Lambda config):
  GH_OWNER  — GitHub repo owner (your username or org)
  GH_REPO   — GitHub repo name (flight-delay-mlops)
  GH_PAT    — GitHub Personal Access Token (from Secrets Manager)
  GH_BRANCH — branch to trigger on (default: main)

EventBridge rule pattern:
  {
    "source": ["aws.s3"],
    "detail-type": ["Object Created"],
    "detail": {
      "bucket": {"name": ["YOUR_BUCKET_NAME"]},
      "object": {"key": [{"prefix": "data/raw/"}]}
    }
  }
════════════════════════════════════════════════════════
"""

import json
import logging
import os
import urllib.request
import urllib.error
import boto3

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# ── Configuration ──────────────────────────────────────
GH_OWNER  = os.environ.get("GH_OWNER",  "")
GH_REPO   = os.environ.get("GH_REPO",   "")
GH_BRANCH = os.environ.get("GH_BRANCH", "main")

# PAT stored in Secrets Manager (not as plain env var)
SECRETS_MANAGER_SECRET = os.environ.get(
    "GH_PAT_SECRET_NAME", "flight-delay/github-pat"
)


def get_github_pat() -> str:
    """Fetch GitHub PAT from AWS Secrets Manager."""
    client = boto3.client("secretsmanager")
    response = client.get_secret_value(SecretId=SECRETS_MANAGER_SECRET)
    secret = json.loads(response["SecretString"])
    return secret["token"]


def trigger_github_workflow(
    s3_bucket: str,
    s3_key: str,
    github_pat: str,
) -> dict:
    """
    Call GitHub Actions API to trigger train.yml.

    Uses workflow_dispatch event with inputs:
      s3_bucket: the S3 bucket containing the new data
      s3_key:    the S3 object key of the new data file
    """
    url = (
        f"https://api.github.com/repos/{GH_OWNER}/{GH_REPO}"
        f"/actions/workflows/train.yml/dispatches"
    )

    payload = json.dumps({
        "ref": GH_BRANCH,
        "inputs": {
            "s3_bucket": s3_bucket,
            "s3_key":    s3_key,
        },
    }).encode("utf-8")

    req = urllib.request.Request(
        url,
        data=payload,
        headers={
            "Authorization":        f"Bearer {github_pat}",
            "Accept":               "application/vnd.github+json",
            "Content-Type":         "application/json",
            "X-GitHub-Api-Version": "2022-11-28",
        },
        method="POST",
    )

    with urllib.request.urlopen(req) as response:
        status_code = response.getcode()
        # 204 = success (no content returned)
        return {"status_code": status_code}


def handler(event: dict, context) -> dict:
    """
    Lambda handler — called by EventBridge when S3 object created.

    EventBridge S3 event shape:
    {
      "source": "aws.s3",
      "detail-type": "Object Created",
      "detail": {
        "bucket": {"name": "my-bucket"},
        "object": {"key": "data/raw/flights_2024_02.csv", "size": 12345}
      }
    }
    """
    logger.info(f"Received event: {json.dumps(event)}")

    try:
        detail = event.get("detail", {})
        bucket = detail.get("bucket", {}).get("name", "")
        key    = detail.get("object", {}).get("key", "")

        if not bucket or not key:
            logger.error(f"Missing bucket or key: bucket={bucket} key={key}")
            return {"statusCode": 400, "body": "Missing bucket or key"}

        # Only trigger for CSV files in data/raw/ prefix
        if not key.startswith("data/raw/") or not key.endswith(".csv"):
            logger.info(f"Skipping non-data file: {key}")
            return {"statusCode": 200, "body": f"Skipped: {key}"}

        logger.info(f"New data file detected: s3://{bucket}/{key}")

        # Get GitHub PAT from Secrets Manager
        github_pat = get_github_pat()

        # Trigger the training workflow
        result = trigger_github_workflow(
            s3_bucket=bucket,
            s3_key=key,
            github_pat=github_pat,
        )

        logger.info(
            f"✓ Triggered train.yml for s3://{bucket}/{key} "
            f"— HTTP {result['status_code']}"
        )
        return {
            "statusCode": 200,
            "body": json.dumps({
                "message": "Training pipeline triggered",
                "s3_bucket": bucket,
                "s3_key": key,
                "workflow": "train.yml",
            }),
        }

    except urllib.error.HTTPError as e:
        logger.error(f"GitHub API error: {e.code} {e.read().decode()}")
        return {"statusCode": 500, "body": f"GitHub API error: {e.code}"}

    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        return {"statusCode": 500, "body": str(e)}
