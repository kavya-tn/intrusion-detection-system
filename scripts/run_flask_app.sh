#!/bin/bash

# Navigate to the project root directory (assuming this script is in project_root/scripts)
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
PROJECT_ROOT=$(dirname "$SCRIPT_DIR")
cd "$PROJECT_ROOT" || exit

export FLASK_APP=src/deployment/app.py
export FLASK_ENV=${FLASK_ENV:-development} # Default to development if not set
export MLFLOW_TRACKING_URI=${MLFLOW_TRACKING_URI:-./mlruns} # Default to local mlruns if not set
export PYTHONPATH="${PROJECT_ROOT}/src:$PYTHONPATH"

echo "Starting Flask application..."
echo "FLASK_APP: $FLASK_APP"
echo "FLASK_ENV: $FLASK_ENV"
echo "MLFLOW_TRACKING_URI: $MLFLOW_TRACKING_URI"
echo "PYTHONPATH: $PYTHONPATH"
echo "Flask server will run on http://0.0.0.0:5001"

# Use gunicorn for a more robust server, or flask run for development
# If gunicorn is available, use it. Otherwise, fall back to flask run.
if command -v gunicorn &> /dev/null
then
    echo "Using gunicorn to run the Flask app..."
    gunicorn --bind 0.0.0.0:5001 --workers 2 --log-level info src.deployment.app:app
else
    echo "gunicorn not found. Using 'flask run' (development server)..."
    flask run --host=0.0.0.0 --port=5001
fi
