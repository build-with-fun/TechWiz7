#!/bin/sh
# Container entry point: create and seed the database on first start, then serve.
# One worker keeps a single copy of both models in memory; its threads share them.
set -e
mkdir -p "$(dirname "$SST_DB_PATH")" "$SST_STORAGE_DIR"
python database/init_db.py
exec gunicorn -w 1 --threads 4 --timeout 300 -b 0.0.0.0:"${PORT:-7860}" "src.app:create_app()"
