#!/bin/bash
# ════════════════════════════════════════════════════════
# infra/setup_p2_aws.sh
# Flight Delay MLOps — P2 One-Time AWS Infrastructure Setup
#
# Run this ONCE after completing P1 setup.
# Prerequisites: P1 setup_aws.sh already run.
#
# What it creates:
#   1. ECR repository for inference Docker images
#   2. Lambda function (approval_trigger.py)
#   3. EventBridge rule watching SageMaker Model Registry approvals
#   4. Lambda permission for EventBridge
#   5. SSM Parameter Store entries for image tag tracking
#   6. IAM permissions for GitHub Actions to push to ECR
#
# Usage:
#   export AWS_REGION=us-east-1
#   export S3_BUCKET=flight-delay-mlops-yourname
#   export GH_OWNER=your-github-username
#   export GH_REPO=flight-delay-mlops
#   bash infra/setup_p2_aws.sh
# ════════════════════════════════════════════════════════

set -euo pipefail

: "${AWS_REGION:?Set AWS_REGION}"
: "${S3_BUCKET:?Set S3_BUCKET}"
: "${GH_OWNER:?Set GH_OWNER}"
: "${GH_REPO:?Set GH_REPO}"

ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
echo ""
echo "════════════════════════════════════════════════════"
echo "  Flight Delay MLOps — P2 AWS Setup"
echo "  Account: $ACCOUNT_ID"
echo "  Region:  $AWS_REGION"
echo "════════════════════════════════════════════════════"
echo ""

# ── 1. Create ECR Repository ────────────────────────────
echo "Step 1: Creating ECR repository..."
ECR_REPO_NAME="flight-delay-inference"

if aws ecr describe-repositories --repository-names "$ECR_REPO_NAME" \
    --region "$AWS_REGION" 2>/dev/null; then
    echo "  ✓ ECR repository already exists: $ECR_REPO_NAME"
else
    aws ecr create-repository \
        --repository-name "$ECR_REPO_NAME" \
        --region "$AWS_REGION" \
        --image-scanning-configuration scanOnPush=true \
        --image-tag-mutability MUTABLE > /dev/null
    echo "  ✓ Created ECR repository: $ECR_REPO_NAME"
fi

ECR_URI="${ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPO_NAME}"
echo "  URI: $ECR_URI"

# ── 2. Add ECR permissions to existing GitHub Actions IAM user ──
echo ""
echo "Step 2: Adding ECR permissions to IAM user/role..."

# Create ECR policy for GitHub Actions
ECR_POLICY=$(cat <<EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "ecr:GetAuthorizationToken"
      ],
      "Resource": "*"
    },
    {
      "Effect": "Allow",
      "Action": [
        "ecr:BatchCheckLayerAvailability",
        "ecr:GetDownloadUrlForLayer",
        "ecr:BatchGetImage",
        "ecr:InitiateLayerUpload",
        "ecr:UploadLayerPart",
        "ecr:CompleteLayerUpload",
        "ecr:PutImage"
      ],
      "Resource": "arn:aws:ecr:${AWS_REGION}:${ACCOUNT_ID}:repository/${ECR_REPO_NAME}"
    }
  ]
}
EOF
)

POLICY_NAME="GitHubActionsECRAccess"
if aws iam get-policy --policy-arn "arn:aws:iam::${ACCOUNT_ID}:policy/${POLICY_NAME}" \
    2>/dev/null; then
    aws iam create-policy-version \
        --policy-arn "arn:aws:iam::${ACCOUNT_ID}:policy/${POLICY_NAME}" \
        --policy-document "$ECR_POLICY" \
        --set-as-default > /dev/null
    echo "  ✓ Updated ECR policy: $POLICY_NAME"
else
    aws iam create-policy \
        --policy-name "$POLICY_NAME" \
        --policy-document "$ECR_POLICY" > /dev/null
    echo "  ✓ Created ECR policy: $POLICY_NAME"
fi

# ── 3. Add SSM permissions to SageMaker role ────────────
echo ""
echo "Step 3: Adding SSM + SageMaker Model Registry permissions..."

SAGEMAKER_ROLE="SageMakerExecutionRole-FlightDelay"
LAMBDA_ROLE="LambdaS3GitHubTrigger-FlightDelay"

# SageMaker role needs SSM read/write for image tag tracking
aws iam attach-role-policy \
    --role-name "$SAGEMAKER_ROLE" \
    --policy-arn "arn:aws:iam::aws:policy/AmazonSSMFullAccess" \
    2>/dev/null || echo "  (SSM policy already attached to SageMaker role)"

# Lambda role needs SSM read/write (deploy.yml uses it)
aws iam attach-role-policy \
    --role-name "$LAMBDA_ROLE" \
    --policy-arn "arn:aws:iam::aws:policy/AmazonSSMFullAccess" \
    2>/dev/null || echo "  (SSM policy already attached to Lambda role)"

# The identity running this script also needs SSM write access to initialise
# the parameters in Step 4. Detect whether it's an IAM user or assumed role.
CALLER_ARN=$(aws sts get-caller-identity --query Arn --output text)
if echo "$CALLER_ARN" | grep -q ":user/"; then
    CALLER_NAME=$(echo "$CALLER_ARN" | cut -d'/' -f2)
    aws iam attach-user-policy \
        --user-name "$CALLER_NAME" \
        --policy-arn "arn:aws:iam::aws:policy/AmazonSSMFullAccess" \
        2>/dev/null || echo "  (SSM policy already attached to calling user)"
    echo "  ✓ SSM permissions attached to calling user: $CALLER_NAME"
elif echo "$CALLER_ARN" | grep -q ":assumed-role/"; then
    CALLER_NAME=$(echo "$CALLER_ARN" | cut -d'/' -f2)
    aws iam attach-role-policy \
        --role-name "$CALLER_NAME" \
        --policy-arn "arn:aws:iam::aws:policy/AmazonSSMFullAccess" \
        2>/dev/null || echo "  (SSM policy already attached to calling role)"
    echo "  ✓ SSM permissions attached to calling role: $CALLER_NAME"
fi

echo "  ✓ SSM permissions attached"

# ── 4. Initialize SSM Parameter Store entries ───────────
echo ""
echo "Step 4: Initializing SSM Parameter Store..."
# Brief pause — IAM policy attachments can take a few seconds to propagate
sleep 5

aws ssm put-parameter \
    --name "flight-delay-current-image-tag" \
    --value "none" \
    --type "String" \
    --overwrite \
    --region "$AWS_REGION" > /dev/null

aws ssm put-parameter \
    --name "flight-delay-previous-image-tag" \
    --value "none" \
    --type "String" \
    --overwrite \
    --region "$AWS_REGION" > /dev/null

echo "  ✓ SSM parameters initialized:"
echo "    flight-delay-current-image-tag  = none"
echo "    flight-delay-previous-image-tag = none"

# ── 5. Create Lambda function for approval trigger ──────
echo ""
echo "Step 5: Deploying approval_trigger Lambda..."

LAMBDA_ROLE_ARN="arn:aws:iam::${ACCOUNT_ID}:role/${LAMBDA_ROLE}"
APPROVAL_LAMBDA_NAME="flight-delay-approval-trigger"
SECRET_NAME="flight-delay/github-pat"

# Package the lambda (use python zipfile to avoid requiring zip on Windows/Git Bash)
cd lambda
python -c "import zipfile; zipfile.ZipFile('approval_trigger.zip', 'w', zipfile.ZIP_DEFLATED).write('approval_trigger.py')"
cd ..

if aws lambda get-function --function-name "$APPROVAL_LAMBDA_NAME" \
    --region "$AWS_REGION" 2>/dev/null; then
    aws lambda update-function-code \
        --function-name "$APPROVAL_LAMBDA_NAME" \
        --zip-file fileb://lambda/approval_trigger.zip \
        --region "$AWS_REGION" > /dev/null
    echo "  ✓ Updated Lambda: $APPROVAL_LAMBDA_NAME"
else
    sleep 5
    aws lambda create-function \
        --function-name "$APPROVAL_LAMBDA_NAME" \
        --runtime python3.11 \
        --role "$LAMBDA_ROLE_ARN" \
        --handler approval_trigger.handler \
        --zip-file fileb://lambda/approval_trigger.zip \
        --timeout 30 \
        --region "$AWS_REGION" \
        --environment "Variables={GH_OWNER=${GH_OWNER},GH_REPO=${GH_REPO},GH_BRANCH=main,GH_PAT_SECRET_NAME=${SECRET_NAME}}" > /dev/null
    echo "  ✓ Created Lambda: $APPROVAL_LAMBDA_NAME"
fi

APPROVAL_LAMBDA_ARN="arn:aws:lambda:${AWS_REGION}:${ACCOUNT_ID}:function:${APPROVAL_LAMBDA_NAME}"

# ── 6. Create EventBridge rule for Model Registry ───────
echo ""
echo "Step 6: Creating EventBridge rule for model approval..."

APPROVAL_RULE_NAME="flight-delay-model-approval"

EVENT_PATTERN=$(cat <<EOF
{
  "source": ["aws.sagemaker"],
  "detail-type": ["SageMaker Model Package State Change"],
  "detail": {
    "ModelApprovalStatus": ["Approved"],
    "ModelPackageGroupName": ["flight-delay-model-group"]
  }
}
EOF
)

aws events put-rule \
    --name "$APPROVAL_RULE_NAME" \
    --event-pattern "$EVENT_PATTERN" \
    --state ENABLED \
    --description "Triggers deployment when flight delay model is approved" \
    --region "$AWS_REGION" > /dev/null

aws events put-targets \
    --rule "$APPROVAL_RULE_NAME" \
    --targets "Id=ApprovalLambdaTarget,Arn=$APPROVAL_LAMBDA_ARN" \
    --region "$AWS_REGION" > /dev/null

# Allow EventBridge to invoke the approval Lambda
aws lambda add-permission \
    --function-name "$APPROVAL_LAMBDA_NAME" \
    --statement-id "EventBridgeApprovalTrigger" \
    --action lambda:InvokeFunction \
    --principal events.amazonaws.com \
    --source-arn "arn:aws:events:${AWS_REGION}:${ACCOUNT_ID}:rule/${APPROVAL_RULE_NAME}" \
    --region "$AWS_REGION" \
    2>/dev/null || echo "  (Permission already exists)"

echo "  ✓ EventBridge rule created and connected to approval Lambda"

# ── 7. Print summary ─────────────────────────────────────
echo ""
echo "════════════════════════════════════════════════════"
echo "  P2 SETUP COMPLETE!"
echo ""
echo "  New GitHub Secret to add:"
echo "  ┌─────────────────────────────────────────────────"
echo "  │ ECR_REPOSITORY = $ECR_REPO_NAME"
echo "  └─────────────────────────────────────────────────"
echo ""
echo "  What is now wired up:"
echo "  AWS Console approval → EventBridge → Lambda → deploy.yml"
echo ""
echo "  To test: go to SageMaker → Model Registry → flight-delay-model-group"
echo "  Find the PendingManualApproval model → Actions → Approve"
echo "  Then watch GitHub Actions → deploy.yml starts automatically"
echo "════════════════════════════════════════════════════"
