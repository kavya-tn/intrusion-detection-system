#!/bin/bash

# Set the MLflow tracking URI to a local directory, but respect if already set
export MLFLOW_TRACKING_URI=${MLFLOW_TRACKING_URI:-./mlruns}

# Navigate to the root directory of the project if the script is not run from there
# This assumes the script is in project_root/scripts/
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
PROJECT_ROOT=$(dirname "$SCRIPT_DIR")

cd "$PROJECT_ROOT" || exit

echo "Running ML model training..."
echo "MLflow Tracking URI: $MLFLOW_TRACKING_URI"

# Ensure Python can find modules in src
export PYTHONPATH="${PROJECT_ROOT}/src:$PYTHONPATH"

python src/training/train_models.py

echo "Training script finished."
