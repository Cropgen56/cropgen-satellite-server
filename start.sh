#!/usr/bin/env bash
set -euo pipefail

# load conda
source ~/miniconda3/etc/profile.d/conda.sh

# activate env
conda activate cropgen-env

# go to project dir
cd ~/cropgen-satellite-apis

# OPTIONAL: load .env (if you use one)
# if [ -f .env ]; then export $(grep -v '^#' .env | xargs); fi

# run uvicorn (change main:app or port if needed)
exec uvicorn main:app --host 127.0.0.1 --port 8001 --workers 4 --proxy-headers
