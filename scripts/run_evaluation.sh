#!/bin/bash

# Set the MLflow tracking URI to a local directory
export MLFLOW_TRACKING_URI=${MLFLOW_TRACKING_URI:-./mlruns} # Use existing or default to ./mlruns

# Navigate to the root directory of the project if the script is not run from there
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
PROJECT_ROOT=$(dirname "$SCRIPT_DIR")

cd "$PROJECT_ROOT" || exit

echo "Running ML model evaluation..."
echo "MLflow Tracking URI: $MLFLOW_TRACKING_URI"

# Ensure Python can find modules in src
export PYTHONPATH="${PROJECT_ROOT}/src:$PYTHONPATH"

python src/training/evaluate_models.py

echo "Evaluation script finished."
