# Flight Delay MLOps — Project Lecture 1: Foundation

## What This Lecture Builds

By the end of P1, uploading a new flight CSV to S3 automatically triggers a full SageMaker training pipeline. You never run a training command manually again.

---

## Prerequisites

You need:
- An AWS account (not brand new — quota for SageMaker instances)
- A GitHub account with an empty repository named `flight-delay-mlops`
- AWS CLI installed and configured (`aws configure`)
- Python 3.11 installed on your local machine
- Git installed
- The flight delay dataset downloaded (from Kaggle: BTS 2019-2023)

---

## Step-by-Step Setup From Scratch

### Step 1 — Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/flight-delay-mlops.git
cd flight-delay-mlops
```

Copy all files from the P1 zip into this directory. Then:

```bash
git add .
git commit -m "Initial P1 foundation setup"
```

Do NOT push yet — set up secrets first.

---

### Step 2 — Set up DagsHub (MLflow tracking)

1. Go to https://dagshub.com and create a free account
2. Click **Create → New Repository** → name it `flight-delay-mlops`
3. On your repo page, click the **MLflow** tab
4. Copy the **Tracking URI** — it looks like:
   `https://dagshub.com/YOUR_USERNAME/flight-delay-mlops.mlflow`
5. Go to **User Settings** (top-right) → **Tokens** → **New Token**
6. Name it `flight-delay-mlops` → copy the token

Save these — you will need them for GitHub Secrets.

---

### Step 3 — Install dependencies locally

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install uv
uv pip install -r requirements.txt
```

Verify:
```bash
python -c "import mlflow, sagemaker; print('All imports OK')"
```

---

### Step 4 — Run the AWS infrastructure setup

This script creates everything in AWS: S3 bucket, IAM roles, Lambda, EventBridge rule, Secrets Manager secret.

```bash
export AWS_REGION=us-east-1
export S3_BUCKET=flight-delay-mlops-mayur    # must be globally unique
export GH_OWNER=mayurb99
export GH_REPO=flight-delay-mlops
export GH_PAT=PAT    # create at github.com/settings/tokens

bash infra/setup_aws.sh
```

**Creating the GitHub PAT:**
1. Go to https://github.com/settings/tokens
2. Click **Generate new token (classic)**
3. Name: `flight-delay-mlops-lambda`
4. Select scopes: `workflow` + `repo`
5. Click **Generate token** → copy immediately

> **Security:** Never commit the PAT to git. The setup script stores it in AWS Secrets Manager automatically.

The setup script takes about 2 minutes and prints exactly what to add to GitHub Secrets at the end.

---

### Step 5 — Add GitHub Secrets

Go to your GitHub repo → **Settings** → **Secrets and variables** → **Actions** → **New repository secret**

Add these one by one:

| Secret Name | Value | Where to find it |
|---|---|---|
| `AWS_ACCESS_KEY_ID` | Your IAM access key | AWS Console → IAM → Users → Security credentials |
| `AWS_SECRET_ACCESS_KEY` | Your IAM secret key | Same place as above |
| `AWS_REGION` | `us-east-1` | Same region as your S3 bucket |
| `SAGEMAKER_ROLE_ARN` | `arn:aws:iam::ACCOUNT:role/SageMakerExecutionRole-FlightDelay` | setup_aws.sh printed this |
| `S3_BUCKET` | Your bucket name (no s3://) | What you set in Step 4 |
| `MLFLOW_TRACKING_URI` | `https://dagshub.com/USER/flight-delay-mlops.mlflow` | DagsHub MLflow tab |
| `MLFLOW_TRACKING_USERNAME` | Your DagsHub username | DagsHub profile |
| `MLFLOW_TRACKING_PASSWORD` | Your DagsHub token | DagsHub User Settings → Tokens |
| `GH_PAT` | Your GitHub PAT | Created in Step 4 |

---

### Step 6 — Set up branch protection

Go to GitHub repo → **Settings** → **Branches** → **Add rule**

- Branch name pattern: `main`
- ✅ Require a pull request before merging
- ✅ Require status checks to pass before merging
  - Add: `CI — Lint, Test, Validate`
- ✅ Require branches to be up to date before merging
- ✅ Do not allow bypassing the above settings

Now nobody (including you) can push directly to main. Every change goes through a PR.

---

### Step 7 — Upload the flight data to S3

Put your flight CSV in the local `data/raw/` folder (gitignored — never committed).

```bash
mkdir -p data/raw
cp ~/Downloads/flights.csv data/raw/flights.csv
```

Upload to S3 to make it available for training:

```bash
aws s3 cp data/raw/flights.csv \
    s3://YOUR_BUCKET/data/raw/flights_2024_01.csv
```

Your S3 bucket has versioning enabled — every file you upload is automatically versioned. You can always retrieve any previous version from the AWS console.

---

### Step 8 — Run unit tests locally

```bash
pytest tests/test_features.py -v
```

Expected: all tests pass. If not, check your Python path.

---

### Step 9 — Push to main and trigger CI

```bash
git push origin main
```

Go to GitHub → **Actions** → you will see CI running on the push.

Wait for it to complete (green ✓).

---

### Step 10 — Trigger the training pipeline

Upload the flight data to S3 at the path that EventBridge watches:

```bash
aws s3 cp data/raw/flights.csv \
    s3://flight-delay-mlops-mayur/data/raw/flights_2024_01.csv
```

**This is the trigger.** Within 30 seconds:
1. S3 creates an ObjectCreated event
2. EventBridge rule matches it
3. Lambda `s3_to_github.py` is invoked
4. Lambda calls GitHub Actions API
5. `train.yml` workflow starts with `s3_key=data/raw/flights_2024_01.csv`

Watch it in GitHub → **Actions** → `Train Pipeline — SageMaker`

The workflow will:
1. Run data validation tests against the uploaded file
2. Submit the SageMaker Pipeline using that exact file
3. Wait for all 4 steps to complete (~25-40 minutes)
4. If challenger beats champion → open a GitHub issue

---

### Step 11 — Verify in MLflow (DagsHub)

1. Go to https://dagshub.com/YOUR_USERNAME/flight-delay-mlops
2. Click the **MLflow** tab
3. You should see a new experiment run: `flight-delay-prediction`
4. Click it to see: parameters, metrics, feature importance chart, model artifact

---

### Step 12 — Verify in SageMaker

1. Go to **AWS Console → SageMaker → Pipelines**
2. Find `flight-delay-training-pipeline`
3. Click the latest execution — see the 4-step DAG
4. After completion: **SageMaker → Model Registry → flight-delay-model-group**
5. You should see a model version with status `PendingManualApproval`

---

## File Structure

```
flight-delay-mlops/
├── .github/workflows/
│   ├── ci.yml           ← runs on every PR
│   └── train.yml        ← triggered by Lambda via workflow_dispatch
├── src/
│   ├── features.py      ← feature engineering + column normalization
│   ├── preprocessing.py ← SageMaker ProcessingStep
│   ├── train.py         ← training + MLflow logging
│   └── evaluate.py      ← champion vs challenger
├── sagemaker/
│   └── pipeline.py      ← 4-step pipeline definition
├── lambda/
│   └── s3_to_github.py  ← S3 event → train.yml
├── tests/
│   ├── test_features.py
│   ├── test_data.py
│   └── test_model.py
├── infra/
│   ├── setup_aws.sh
│   └── eventbridge_s3.json
├── data/raw/            ← gitignored, stored in S3 (versioning enabled)
├── requirements.txt
└── README.md
```

---

## Data Versioning

Data is versioned automatically by S3 bucket versioning (enabled in `setup_aws.sh`). Every upload creates a new version. To list versions:

```bash
aws s3api list-object-versions \
    --bucket YOUR_BUCKET \
    --prefix data/raw/flights.csv
```

To download a specific version:

```bash
aws s3api get-object \
    --bucket YOUR_BUCKET \
    --key data/raw/flights.csv \
    --version-id VERSION_ID \
    output.csv
```

---

## Common Issues

| Problem | Fix |
|---|---|
| `boto3.exceptions.NoCredentialsError` | Run `aws configure` or export AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY |
| Lambda not triggering | Check S3 bucket has EventBridge notifications enabled; verify EventBridge rule has Lambda as target (`aws events list-targets-by-rule --rule flight-delay-s3-data-upload`) |
| SageMaker quota error | Request ml.t3.medium quota for processing jobs at AWS Service Quotas |
| MLflow tracking fails | Verify all 3 MLFLOW_* GitHub Secrets are set correctly |
| CI fails on test_model.py | Normal if no champion exists yet — accuracy tests skip gracefully until first model is registered |
| PAT rotated | Update Secrets Manager: `aws secretsmanager update-secret --secret-id flight-delay/github-pat --secret-string '{"token":"ghp_NEW"}'` and update GitHub Secret `GH_PAT` |

---

## What Happens in P2

After P1 is complete, P2 adds:
- `lambda/approval_trigger.py` — approval in console → deploy.yml
- `.github/workflows/deploy.yml` — Blue/Green deployment
- SageMaker Model Monitor setup
- Automatic endpoint creation after approval
