"""
lambda/approval_trigger.py
════════════════════════════════════════════════════════
Flight Delay Prediction — Model Registry Approval → Deploy

Project Lecture 2: Approval + Blue/Green Deploy
Deployed as: AWS Lambda function
Triggered by: EventBridge rule watching SageMaker Model Registry

What it does:
  1. EventBridge fires when a model package status changes to Approved
  2. This Lambda receives the event
  3. Extracts the ModelPackageArn from the event
  4. Calls GitHub Actions API to trigger deploy.yml
  5. Passes model_package_arn as workflow input

EventBridge event pattern (set in infra/eventbridge_approval.json):
  {
    "source": ["aws.sagemaker"],
    "detail-type": ["SageMaker Model Package State Change"],
    "detail": {
      "ModelApprovalStatus": ["Approved"],
      "ModelPackageGroupName": ["flight-delay-model-group"]
    }
  }

Environment variables (set in Lambda config):
  GH_OWNER            — GitHub repo owner username
  GH_REPO             — GitHub repo name
  GH_BRANCH           — branch to trigger on (default: main)
  GH_PAT_SECRET_NAME  — Secrets Manager secret name for PAT
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

SECRETS_MANAGER_SECRET = os.environ.get(
    "GH_PAT_SECRET_NAME", "flight-delay/github-pat"
)


def get_github_pat() -> str:
    """Fetch GitHub PAT from AWS Secrets Manager."""
    client = boto3.client("secretsmanager")
    response = client.get_secret_value(SecretId=SECRETS_MANAGER_SECRET)
    secret = json.loads(response["SecretString"])
    return secret["token"]


def trigger_deploy_workflow(
    model_package_arn: str,
    github_pat: str,
) -> dict:
    """
    Call GitHub Actions API to trigger deploy.yml.

    Passes model_package_arn as workflow input so deploy.yml
    knows exactly which approved model version to deploy.
    """
    url = (
        f"https://api.github.com/repos/{GH_OWNER}/{GH_REPO}"
        f"/actions/workflows/deploy.yml/dispatches"
    )

    payload = json.dumps({
        "ref": GH_BRANCH,
        "inputs": {
            "model_package_arn": model_package_arn,
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
        return {"status_code": response.getcode()}


def handler(event: dict, context) -> dict:
    """
    Lambda handler — called by EventBridge on model package approval.

    EventBridge SageMaker Model Package event shape:
    {
      "source": "aws.sagemaker",
      "detail-type": "SageMaker Model Package State Change",
      "detail": {
        "ModelPackageArn": "arn:aws:sagemaker:...:model-package/...",
        "ModelPackageGroupName": "flight-delay-model-group",
        "ModelApprovalStatus": "Approved",
        "ModelPackageStatus": "Completed"
      }
    }
    """
    logger.info(f"Received event: {json.dumps(event)}")

    try:
        detail = event.get("detail", {})
        model_package_arn    = detail.get("ModelPackageArn", "")
        model_package_group  = detail.get("ModelPackageGroupName", "")
        approval_status      = detail.get("ModelApprovalStatus", "")

        if not model_package_arn:
            logger.error("No ModelPackageArn in event detail")
            return {"statusCode": 400, "body": "Missing ModelPackageArn"}

        # Safety check — only deploy approvals for our model group
        if model_package_group != "flight-delay-model-group":
            logger.info(f"Skipping — different model group: {model_package_group}")
            return {"statusCode": 200, "body": "Skipped — different model group"}

        if approval_status != "Approved":
            logger.info(f"Skipping — status is {approval_status}, not Approved")
            return {"statusCode": 200, "body": f"Skipped — status: {approval_status}"}

        logger.info(f"Model approved: {model_package_arn}")

        # Get GitHub PAT from Secrets Manager
        github_pat = get_github_pat()

        # Trigger deployment workflow
        result = trigger_deploy_workflow(
            model_package_arn=model_package_arn,
            github_pat=github_pat,
        )

        logger.info(
            f"✓ Triggered deploy.yml — HTTP {result['status_code']}"
        )
        return {
            "statusCode": 200,
            "body": json.dumps({
                "message":           "Deployment triggered",
                "model_package_arn": model_package_arn,
                "workflow":          "deploy.yml",
            }),
        }

    except urllib.error.HTTPError as e:
        logger.error(f"GitHub API error: {e.code} {e.read().decode()}")
        return {"statusCode": 500, "body": f"GitHub API error: {e.code}"}

    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        return {"statusCode": 500, "body": str(e)}
