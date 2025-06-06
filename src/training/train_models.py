"""
Loads preprocessed training and testing data, defines a set of machine learning models
with their hyperparameter grids, and then for each model:
1. Starts an MLflow run.
2. Performs hyperparameter tuning using GridSearchCV.
3. Logs all searched hyperparameters and the best parameters found to MLflow.
4. Trains the final model using the best parameters on the full training data.
5. Calculates and logs evaluation metrics (accuracy, precision, recall, F1-score, ROC AUC) on the test set.
6. Logs the trained model artifact to MLflow.

The script includes a check for the existence of data files and attempts to run
preprocessing scripts if data is missing.
"""
import os
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define data paths
TRAIN_X_PATH = "data/train_X.csv"
TRAIN_Y_PATH = "data/train_y.csv"
TEST_X_PATH = "data/test_X.csv"
TEST_Y_PATH = "data/test_y.csv"

# Define models and their hyperparameter grids
MODELS_GRID = {
    "LogisticRegression": {
        "model": LogisticRegression(max_iter=1000), # Increased max_iter for convergence
        "params": {'C': [0.1, 1.0, 10], 'solver': ['liblinear']}
    },
    "SVC": {
        "model": SVC(probability=True), # probability=True for roc_auc
        "params": {'C': [0.1, 1.0], 'kernel': ['linear', 'rbf']}
    },
    "RandomForestClassifier": {
        "model": RandomForestClassifier(random_state=42),
        "params": {'n_estimators': [50, 100], 'max_depth': [5, 10]}
    },
    "GradientBoostingClassifier": {
        "model": GradientBoostingClassifier(random_state=42),
        "params": {'n_estimators': [50, 100], 'learning_rate': [0.05, 0.1]}
    },
    "KNeighborsClassifier": {
        "model": KNeighborsClassifier(),
        "params": {'n_neighbors': [3, 5, 7]}
    },
    "GaussianNB": {
        "model": GaussianNB(),
        "params": {'var_smoothing': [1e-9, 1e-8, 1e-7]} # Added some basic params
    }
}

def load_data():
    """Loads preprocessed training and testing data."""
    logging.info("Loading preprocessed data...")
    X_train = pd.read_csv(TRAIN_X_PATH)
    y_train = pd.read_csv(TRAIN_Y_PATH).squeeze() # Use squeeze to make it a Series
    X_test = pd.read_csv(TEST_X_PATH)
    y_test = pd.read_csv(TEST_Y_PATH).squeeze() # Use squeeze to make it a Series
    logging.info("Data loaded successfully.")
    return X_train, y_train, X_test, y_test

def train_and_log_models():
    """Trains models, performs hyperparameter tuning, and logs with MLflow."""
    X_train, y_train, X_test, y_test = load_data()

    logging.info("Starting model training loop...")
    for model_name, config in MODELS_GRID.items():
        logging.info(f"--- Training Model: {model_name} ---")
        with mlflow.start_run(run_name=model_name) as run:
            run_id = run.info.run_id
            logging.info(f"MLflow Run ID: {run_id}")
            mlflow.set_tag("model_type", model_name)

            logging.info(f"Performing GridSearchCV for {model_name} (CV=3, scoring='accuracy')...")
            # GridSearchCV will refit the best model on the whole training data by default (refit=True)
            grid_search = GridSearchCV(config["model"], config["params"], cv=3, scoring="accuracy", verbose=1)
            grid_search.fit(X_train, y_train)

            best_params = grid_search.best_params_
            search_grid_params = config["params"]

            logging.info(f"Best parameters found for {model_name}: {best_params}")

            # Log all hyperparameters from the search grid
            # MLflow UI can struggle with deeply nested params, so flatten or use JSON string if complex.
            # For this grid, direct logging is fine.
            mlflow.log_params({f"search_{k}": str(v) for k, v in search_grid_params.items()})

            # Log the best parameters found by GridSearchCV
            mlflow.log_params({f"best_{k}": v for k, v in best_params.items()})

            # The best model is available as grid_search.best_estimator_
            model = grid_search.best_estimator_
            # No need to re-train model = config["model"].set_params(**best_params) and model.fit()
            # as GridSearchCV does this if refit=True (default).
            model.fit(X_train, y_train)

            # Make predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

            # Calculate and log metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)

            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1_score", f1)

            if y_pred_proba is not None:
                roc_auc = roc_auc_score(y_test, y_pred_proba)
                mlflow.log_metric("roc_auc", roc_auc)
                logging.info(f"Metrics for {model_name}: Acc={accuracy:.4f}, Prec={precision:.4f}, Rec={recall:.4f}, F1={f1:.4f}, ROC_AUC={roc_auc:.4f}")
            else:
                logging.info(f"Metrics for {model_name}: Acc={accuracy:.4f}, Prec={precision:.4f}, Rec={recall:.4f}, F1={f1:.4f} (ROC AUC not available)")

            # Log the trained model to MLflow
            # The artifact_path determines the subdirectory name within the MLflow run's artifacts.
            artifact_path_name = f"{model_name.lower().replace(' ', '-')}-model"
            mlflow.sklearn.log_model(model, artifact_path=artifact_path_name)
            logging.info(f"Model {model_name} logged to MLflow artifact path: {artifact_path_name}")
        logging.info(f"--- Finished training for {model_name} ---")

    logging.info("All models have been trained and logged to MLflow.")

if __name__ == "__main__":
    logging.info("Starting model training script...")

    # Check if data exists, if not, guide user or attempt to run preprocessing
    required_files = [TRAIN_X_PATH, TRAIN_Y_PATH, TEST_X_PATH, TEST_Y_PATH]
    missing_files = [f for f in required_files if not os.path.exists(f)]

    if missing_files:
        logging.error(f"Error: Missing preprocessed data files: {', '.join(missing_files)}")
        logging.warning("Attempting to run preprocessing scripts automatically...")
        try:
            # Assuming this script is run from the project root or PYTHONPATH is set correctly
            from preprocessing import load_data as ld, preprocess_data as ppd

            # Create data directory if it doesn't exist (load_data.py also does this)
            if not os.path.exists("data"):
                os.makedirs("data")

            # Run load_data if its primary output (processed_heart_disease.csv) is missing
            # load_data.py's PROCESSED_DATA_PATH is "data/processed_heart_disease.csv"
            # This path is used by preprocess_data.py
            path_for_load_data_output = os.path.join("data", "processed_heart_disease.csv") # More robust
            if not os.path.exists(path_for_load_data_output):
                logging.info("Running load_data.py...")
                ld.load_and_process_data()

            logging.info("Running preprocess_data.py...")
            ppd.preprocess_data()
            logging.info("Preprocessing scripts executed successfully. Please re-run training script.")
            # Exit here as user needs to re-run training to use the newly processed data.
            exit(0)
        except ImportError:
            logging.error("Critical Error: Could not import preprocessing modules. Ensure PYTHONPATH is correct or run them manually.")
            exit(1)
        except Exception as e:
            logging.error(f"Critical Error: Failed to run preprocessing scripts automatically: {e}")
            exit(1)
    else:
        logging.info("Preprocessed data found. Proceeding with model training.")
        train_and_log_models()

    logging.info("Model training script finished.")
