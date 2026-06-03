#!/bin/bash
# ════════════════════════════════════════════════════════
# infra/setup_aws.sh
# Flight Delay MLOps — One-Time AWS Infrastructure Setup
# Project Lecture 1: Foundation
#
# Run this ONCE from your local machine after cloning the repo.
# Prerequisites: AWS CLI configured (aws configure)
#
# What it creates:
#   1. S3 bucket for data + artifacts
#   2. IAM role for SageMaker
#   3. S3 event notification → EventBridge
#   4. EventBridge rule watching S3
#   5. Lambda function (s3_to_github)
#   6. Lambda permission for EventBridge
#   7. SageMaker Model Package Group
#   8. Secrets Manager secret for GitHub PAT
#
# Usage:
#   export AWS_REGION=us-east-1
#   export S3_BUCKET=flight-delay-mlops-yourname
#   export GH_OWNER=your-github-username
#   export GH_REPO=flight-delay-mlops
#   export GH_PAT=ghp_your_personal_access_token
#   bash infra/setup_aws.sh
# ════════════════════════════════════════════════════════

set -euo pipefail

# ── Required environment variables ─────────────────────
: "${AWS_REGION:?Set AWS_REGION}"
: "${S3_BUCKET:?Set S3_BUCKET}"
: "${GH_OWNER:?Set GH_OWNER}"
: "${GH_REPO:?Set GH_REPO}"
: "${GH_PAT:?Set GH_PAT}"

ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
echo ""
echo "════════════════════════════════════════════════════"
echo "  Flight Delay MLOps — AWS Setup"
echo "  Account: $ACCOUNT_ID"
echo "  Region:  $AWS_REGION"
echo "  Bucket:  $S3_BUCKET"
echo "════════════════════════════════════════════════════"
echo ""

# ── 1. Create S3 bucket ─────────────────────────────────
echo "Step 1: Creating S3 bucket..."
if aws s3 ls "s3://$S3_BUCKET" 2>/dev/null; then
    echo "  ✓ Bucket already exists: $S3_BUCKET"
else
    if [ "$AWS_REGION" = "us-east-1" ]; then
        aws s3api create-bucket \
            --bucket "$S3_BUCKET" \
            --region "$AWS_REGION"
    else
        aws s3api create-bucket \
            --bucket "$S3_BUCKET" \
            --region "$AWS_REGION" \
            --create-bucket-configuration LocationConstraint="$AWS_REGION"
    fi
    echo "  ✓ Created bucket: $S3_BUCKET"
fi

# Enable versioning
aws s3api put-bucket-versioning \
    --bucket "$S3_BUCKET" \
    --versioning-configuration Status=Enabled
echo "  ✓ Bucket versioning enabled"

# Enable EventBridge notifications
aws s3api put-bucket-notification-configuration \
    --bucket "$S3_BUCKET" \
    --notification-configuration '{"EventBridgeConfiguration": {}}'
echo "  ✓ EventBridge notifications enabled on bucket"

# ── 2. Create IAM Role for SageMaker ───────────────────
echo ""
echo "Step 2: Creating SageMaker IAM role..."
ROLE_NAME="SageMakerExecutionRole-FlightDelay"

TRUST_POLICY=$(cat <<EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {"Service": "sagemaker.amazonaws.com"},
      "Action": "sts:AssumeRole"
    }
  ]
}
EOF
)

if aws iam get-role --role-name "$ROLE_NAME" 2>/dev/null; then
    echo "  ✓ IAM role already exists: $ROLE_NAME"
else
    aws iam create-role \
        --role-name "$ROLE_NAME" \
        --assume-role-policy-document "$TRUST_POLICY" > /dev/null
    aws iam attach-role-policy \
        --role-name "$ROLE_NAME" \
        --policy-arn "arn:aws:iam::aws:policy/AmazonSageMakerFullAccess"
    aws iam attach-role-policy \
        --role-name "$ROLE_NAME" \
        --policy-arn "arn:aws:iam::aws:policy/AmazonS3FullAccess"
    echo "  ✓ Created role: $ROLE_NAME"
fi

ROLE_ARN="arn:aws:iam::${ACCOUNT_ID}:role/${ROLE_NAME}"
echo "  ARN: $ROLE_ARN"

# ── 3. Create Lambda IAM Role ───────────────────────────
echo ""
echo "Step 3: Creating Lambda IAM role..."
LAMBDA_ROLE_NAME="LambdaS3GitHubTrigger-FlightDelay"

LAMBDA_TRUST=$(cat <<EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {"Service": "lambda.amazonaws.com"},
      "Action": "sts:AssumeRole"
    }
  ]
}
EOF
)

if aws iam get-role --role-name "$LAMBDA_ROLE_NAME" 2>/dev/null; then
    echo "  ✓ Lambda role already exists"
else
    aws iam create-role \
        --role-name "$LAMBDA_ROLE_NAME" \
        --assume-role-policy-document "$LAMBDA_TRUST" > /dev/null
    aws iam attach-role-policy \
        --role-name "$LAMBDA_ROLE_NAME" \
        --policy-arn "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
    aws iam attach-role-policy \
        --role-name "$LAMBDA_ROLE_NAME" \
        --policy-arn "arn:aws:iam::aws:policy/SecretsManagerReadWrite"
    echo "  ✓ Created Lambda role: $LAMBDA_ROLE_NAME"
fi

LAMBDA_ROLE_ARN="arn:aws:iam::${ACCOUNT_ID}:role/${LAMBDA_ROLE_NAME}"

# ── 4. Store GitHub PAT in Secrets Manager ─────────────
echo ""
echo "Step 4: Storing GitHub PAT in Secrets Manager..."
SECRET_NAME="flight-delay/github-pat"

aws secretsmanager create-secret \
    --name "$SECRET_NAME" \
    --description "GitHub PAT for Lambda → GitHub Actions trigger" \
    --secret-string "{\"token\": \"$GH_PAT\"}" > /dev/null \
    2>/dev/null \
    && echo "  ✓ Created secret: $SECRET_NAME" \
    || echo "  ✓ Secret already exists: $SECRET_NAME (skipping)"

# ── 5. Deploy Lambda function ───────────────────────────
echo ""
echo "Step 5: Deploying Lambda function..."
LAMBDA_NAME="flight-delay-s3-trigger"

# Package the lambda (use python zipfile to avoid requiring zip on Windows/Git Bash)
cd lambda
python -c "import zipfile; zipfile.ZipFile('s3_to_github.zip', 'w', zipfile.ZIP_DEFLATED).write('s3_to_github.py')"
cd ..

if aws lambda get-function --function-name "$LAMBDA_NAME" 2>/dev/null; then
    aws lambda update-function-code \
        --function-name "$LAMBDA_NAME" \
        --zip-file fileb://lambda/s3_to_github.zip > /dev/null
    echo "  ✓ Updated Lambda function"
else
    sleep 10  # wait for role to propagate
    aws lambda create-function \
        --function-name "$LAMBDA_NAME" \
        --runtime python3.11 \
        --role "$LAMBDA_ROLE_ARN" \
        --handler s3_to_github.handler \
        --zip-file fileb://lambda/s3_to_github.zip \
        --timeout 30 \
        --environment "Variables={GH_OWNER=$GH_OWNER,GH_REPO=$GH_REPO,GH_BRANCH=main,GH_PAT_SECRET_NAME=$SECRET_NAME}" > /dev/null
    echo "  ✓ Created Lambda function: $LAMBDA_NAME"
fi

LAMBDA_ARN="arn:aws:lambda:${AWS_REGION}:${ACCOUNT_ID}:function:${LAMBDA_NAME}"

# ── 6. Create EventBridge Rule ─────────────────────────
echo ""
echo "Step 6: Creating EventBridge rule..."
RULE_NAME="flight-delay-s3-data-upload"

EVENT_PATTERN=$(cat <<EOF
{
  "source": ["aws.s3"],
  "detail-type": ["Object Created"],
  "detail": {
    "bucket": {"name": ["$S3_BUCKET"]},
    "object": {"key": [{"prefix": "data/raw/"}]}
  }
}
EOF
)

aws events put-rule \
    --name "$RULE_NAME" \
    --event-pattern "$EVENT_PATTERN" \
    --state ENABLED \
    --description "Triggers training when new flight data CSV uploaded" > /dev/null

aws events put-targets \
    --rule "$RULE_NAME" \
    --targets "Id=LambdaTarget,Arn=$LAMBDA_ARN" > /dev/null

# Allow EventBridge to invoke Lambda
aws lambda add-permission \
    --function-name "$LAMBDA_NAME" \
    --statement-id "EventBridgeS3Trigger" \
    --action lambda:InvokeFunction \
    --principal events.amazonaws.com \
    --source-arn "arn:aws:events:${AWS_REGION}:${ACCOUNT_ID}:rule/${RULE_NAME}" \
    2>/dev/null || echo "  (Permission already exists)"

echo "  ✓ EventBridge rule created and connected to Lambda"

# ── 7. Create SageMaker Model Package Group ─────────────
echo ""
echo "Step 7: Creating SageMaker Model Package Group..."
aws sagemaker create-model-package-group \
    --model-package-group-name "flight-delay-model-group" \
    --model-package-group-description "Flight delay prediction model versions" \
    2>/dev/null && echo "  ✓ Created model package group" \
    || echo "  ✓ Model package group already exists"

# ── 8. Print summary ───────────────────────────────────
echo ""
echo "════════════════════════════════════════════════════"
echo "  SETUP COMPLETE!"
echo ""
echo "  Now add these to GitHub Secrets:"
echo "  ┌─────────────────────────────────────────────────"
echo "  │ AWS_ACCESS_KEY_ID     = (your IAM user key)"
echo "  │ AWS_SECRET_ACCESS_KEY = (your IAM user secret)"
echo "  │ AWS_REGION            = $AWS_REGION"
echo "  │ SAGEMAKER_ROLE_ARN    = $ROLE_ARN"
echo "  │ S3_BUCKET             = $S3_BUCKET"
echo "  │ MLFLOW_TRACKING_URI   = (from DagsHub)"
echo "  │ MLFLOW_TRACKING_USERNAME = (DagsHub username)"
echo "  │ MLFLOW_TRACKING_PASSWORD = (DagsHub token)"
echo "  │ GH_PAT                = $GH_PAT"
echo "  └─────────────────────────────────────────────────"
echo ""
echo "  To test the S3 trigger:"
echo "  aws s3 cp your_flights.csv s3://$S3_BUCKET/data/raw/flights_test.csv"
echo "  Then check GitHub Actions → train.yml should be running"
echo "════════════════════════════════════════════════════"
