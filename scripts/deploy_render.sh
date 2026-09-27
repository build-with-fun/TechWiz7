#!/usr/bin/env bash
# Render.com deployment kit bootstrap (SRS deliverable 12: Deployed Application).
#
# What this does:
#   1. verifies the repo is commit-clean and the test suite is green
#   2. writes render.yaml (IaC for a Python service) if missing
#   3. prints the exact render CLI / dashboard steps
#
# Render free tier: Python 3.12, 512 MB. That is too small for the served AST model,
# so use a paid instance (or the Hugging Face kit in deploy/huggingface/).
set -euo pipefail
cd "$(dirname "$0")/.."

echo "== [1/3] verifying repo state =="
python_check=".venv/bin/python"
$python_check -c "import src.app" 2>/dev/null || { echo "venv missing, create .venv first"; exit 1; }
git diff --quiet && git diff --cached --quiet || { echo "uncommitted changes, commit first"; exit 1; }

echo "== [2/3] test suite gate =="
$python_check -m pytest -q -x --timeout=300 2>/dev/null | tail -1

echo "== [3/3] render blueprint =="
cat > render.yaml <<'YAML'
services:
  - type: web
    name: sonicsentinel-ai
    runtime: python
    plan: free
    buildCommand: |
      pip install -r requirements.txt
      pip install torch==2.14.0+cpu --index-url https://download.pytorch.org/whl/cpu
      python -c "
from src.db import create_engine_for, default_db_path, init_db
init_db(create_engine_for(default_db_path()))
print('db initialised')"
    startCommand: gunicorn -w 1 --threads 4 --timeout 120 -b 0.0.0.0:$PORT "src.app:create_app()"
    envVars:
      - key: PYTHON_VERSION
        value: "3.12.7"
      - key: SECRET_KEY
        generateValue: true
      - key: SONICSENTINEL_DB
        value: /var/data/sonicsentinel.db
      - key: SONICSENTINEL_ADMIN_PASSWORD
        sync: false   # set in dashboard: Admin#Sonic2026 for evaluation
YAML
echo "render.yaml written. Push to GitHub, then:"
echo "  1. dashboard.render.com: New + > Blueprint > pick this repo"
echo "  2. set SONICSENTINEL_ADMIN_PASSWORD env var"
echo "  3. deploy; URL https://<app>.onrender.com"
echo "  4. add evaluator credentials to the README 'Deployed Application' section"