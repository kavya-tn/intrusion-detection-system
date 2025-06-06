"""
This script evaluates models logged to MLflow from previous training runs.
It fetches all runs from a specified MLflow experiment (defaults to '0'),
extracts relevant metrics and parameters, and identifies the best model based
on the highest F1-score.

The script then:
1. Prints a summary table of all models and their key metrics/hyperparameters.
2. Prints detailed information about the best model.
3. Saves the Run ID, model name, Model URI (for loading), F1-score, ROC AUC,
   accuracy, and best parameters of this best model to `data/best_model_info.json`.
   This JSON file is used by the deployment script (`app.py`) to load the selected model.

It includes logic to automatically determine the model artifact path within an MLflow run
by looking for a directory containing an 'MLmodel' file, or falls back to a convention
based on the model's name tag if direct discovery fails.
"""
import mlflow
import pandas as pd
import json
import os
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define output path for best model info
BEST_MODEL_INFO_PATH = "data/best_model_info.json"
# MLflow experiment configuration (assuming default experiment '0' if not specified)
# If you used a named experiment in train_models.py, set EXPERIMENT_NAME accordingly.
EXPERIMENT_NAME = None # Set to your experiment name if you used one, else None for default '0'

def get_experiment_id_by_name(client, name):
    """Helper to get experiment ID from name."""
    if name is None: # Default experiment
        return "0"
    exp = client.get_experiment_by_name(name)
    if exp:
        return exp.experiment_id
    else:
        logging.warning(f"Experiment '{name}' not found. Falling back to default experiment '0'.")
        return "0" # Fallback or handle as an error

def evaluate_models():
    """Fetches runs, selects the best model, and saves its info."""
    logging.info("Starting model evaluation...")

    client = mlflow.tracking.MlflowClient()

    if EXPERIMENT_NAME:
        experiment_id = get_experiment_id_by_name(client, EXPERIMENT_NAME)
        if experiment_id is None: # Should be handled by get_experiment_id_by_name for default
             logging.error(f"Experiment '{EXPERIMENT_NAME}' not found. Exiting.")
             return
    else: # Use default experiment
        experiment_id = "0"
        logging.info("Using default MLflow experiment '0'.")


    # Search runs in the specified experiment
    # Ensure all necessary columns are fetched. MLflow search_runs returns a list of Run objects.
    # We will convert this to a more usable format.
    try:
        runs = client.search_runs(experiment_ids=[experiment_id])
    except mlflow.exceptions.MlflowException as e:
        logging.error(f"Error fetching runs from MLflow: {e}")
        logging.error("Ensure MLflow tracking URI is set (e.g. export MLFLOW_TRACKING_URI=./mlruns) and 'mlruns' directory exists with data.")
        return

    if not runs:
        logging.warning(f"No runs found in experiment ID '{experiment_id}'. Please run train_models.py first.")
        return

    runs_data = []
    for run in runs:
        run_info = {
            "run_id": run.info.run_id,
            "model_name": run.data.tags.get("model_type", "N/A"),
            "artifact_uri": run.info.artifact_uri,
            # Metrics
            "accuracy": run.data.metrics.get("accuracy"),
            "f1_score": run.data.metrics.get("f1_score"),
            "precision": run.data.metrics.get("precision"),
            "recall": run.data.metrics.get("recall"),
            "roc_auc": run.data.metrics.get("roc_auc"),
        }
        # Add best hyperparameters (look for params starting with 'best_')
        for key, value in run.data.params.items():
            if key.startswith("best_"):
                run_info[key] = value
        runs_data.append(run_info)

    runs_df = pd.DataFrame(runs_data)

    if runs_df.empty:
        logging.warning("No run data could be processed into DataFrame. Check MLflow logs.")
        return

    # Print summary table
    logging.info("Model Performance Summary:")
    # Define columns to display, ensure they exist
    display_columns = ["run_id", "model_name", "accuracy", "f1_score", "precision", "recall", "roc_auc"]
    # Add any best_param columns found to display_columns if they exist in the df
    param_cols = [col for col in runs_df.columns if col.startswith("best_")]
    display_columns.extend(param_cols)

    # Filter out columns that might not exist in all runs to avoid KeyError
    existing_display_columns = [col for col in display_columns if col in runs_df.columns]
    print(runs_df[existing_display_columns].to_string())


    # Model selection based on F1-score
    primary_metric = "f1_score"
    if primary_metric not in runs_df.columns or runs_df[primary_metric].isnull().all():
        logging.error(f"Primary metric '{primary_metric}' not found in any runs or all values are NaN. Cannot select best model.")
        return

    logging.info(f"\nSelecting best model based on highest '{primary_metric}'...")
    best_run_df = runs_df.loc[runs_df[primary_metric].idxmax()]

    logging.info("\nBest Model Details:")
    print(best_run_df)

    # Determine the model artifact path (logged during training)
    # The artifact path in log_model was like "logisticregression-model"
    # We need to find this from the logged artifacts for the best run.

    # The model name tag (e.g., "LogisticRegression") should correspond to the artifact path
    # used during `mlflow.sklearn.log_model(model, artifact_path="<artifact_path_name>")` in train_models.py.
    # The convention used there was: f"{model_name.lower().replace(' ', '-')}-model"

    best_model_name_tag = best_run_df.get("model_name", "N/A")
    if best_model_name_tag == "N/A":
        logging.error("Could not determine model_name tag for the best run. Cannot construct model URI accurately.")
        return # Or raise an error

    # Attempt to discover the model artifact path by listing artifacts.
    # MLflow models are logged as directories containing an MLmodel file.
    model_artifact_path = None
    logging.info(f"Attempting to discover model artifact path for run_id: {best_run_df['run_id']} (model name: {best_model_name_tag})")
    try:
        artifacts = client.list_artifacts(best_run_df["run_id"])
        for artifact_info in artifacts:
            if artifact_info.is_dir:
                # Check if this directory (artifact_info.path) contains an MLmodel file
                try:
                    dir_artifacts = client.list_artifacts(best_run_df["run_id"], path=artifact_info.path)
                    if any(f.path.endswith("MLmodel") for f in dir_artifacts):
                        model_artifact_path = artifact_info.path
                        logging.info(f"Discovered model artifact path: '{model_artifact_path}'")
                        break
                except Exception as e_inner: # nosemgrep
                    # This can happen if artifact_info.path is not actually a directory recognizable by the backend store
                    logging.debug(f"Could not list sub-artifacts for {artifact_info.path}: {e_inner}")
                    pass

        if model_artifact_path is None:
            # Fallback to the convention used in train_models.py if discovery fails
            conventional_path = f"{best_model_name_tag.lower().replace(' ', '-')}-model"
            logging.warning(f"Could not automatically discover model artifact path for run {best_run_df['run_id']}. "
                            f"Falling back to conventional path: '{conventional_path}'")
            model_artifact_path = conventional_path

    except Exception as e:
        # Broader error during artifact listing for the run
        conventional_path = f"{best_model_name_tag.lower().replace(' ', '-')}-model"
        logging.error(f"Error listing artifacts for run {best_run_df['run_id']}: {e}. "
                      f"Falling back to conventional path: '{conventional_path}'")
        model_artifact_path = conventional_path


    best_model_info = {
        "run_id": best_run_df["run_id"],
        "model_name": best_model_name_tag,
        "model_uri": f"runs:/{best_run_df['run_id']}/{model_artifact_path}",
        "f1_score": best_run_df["f1_score"],
        "roc_auc": best_run_df["roc_auc"],
        "accuracy": best_run_df["accuracy"],
        "params": {k: v for k, v in best_run_df.items() if k.startswith("best_")}
    }

    logging.info(f"\nBest model URI: {best_model_info['model_uri']}")

    # Ensure data directory exists
    if not os.path.exists("data"):
        os.makedirs("data")

    # Save best model info
    with open(BEST_MODEL_INFO_PATH, 'w') as f:
        json.dump(best_model_info, f, indent=4)
    logging.info(f"Best model information saved to {BEST_MODEL_INFO_PATH}")

if __name__ == "__main__":
    # Check if MLFLOW_TRACKING_URI is set, if not, default to local ./mlruns
    # This is important for running the script directly if the environment variable isn't set by a wrapper script.
    if "MLFLOW_TRACKING_URI" not in os.environ:
        logging.info("MLFLOW_TRACKING_URI not set in environment.")
        default_mlruns_path = os.path.abspath(os.path.join(os.getcwd(), "mlruns"))
        if os.path.exists(default_mlruns_path) and os.path.isdir(default_mlruns_path):
            # Use file URI scheme for local filesystem paths
            os.environ["MLFLOW_TRACKING_URI"] = f"file:{default_mlruns_path}"
            logging.info(f"Defaulted MLFLOW_TRACKING_URI to local: {os.environ['MLFLOW_TRACKING_URI']}")
        else:
            logging.error(
                "MLFLOW_TRACKING_URI is not set and default './mlruns' directory not found or not a directory."
            )
            logging.error(
                "Please set MLFLOW_TRACKING_URI (e.g., 'export MLFLOW_TRACKING_URI=./mlruns' or 'file:/path/to/mlruns') "
                "or ensure './mlruns' exists and is populated by running train_models.py."
            )
            # Optionally exit if MLflow setup is critical and not found
            # exit(1)

    logging.info("Starting model evaluation script...")
    evaluate_models()
    logging.info("Model evaluation script finished.")
