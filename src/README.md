# Source Code (`src`)

This directory contains all the Python source code for the ML pipeline.

## Subdirectories:

-   **`preprocessing/`**: Contains scripts related to data loading, cleaning, transformation, and feature engineering.
    -   `load_data.py`: Downloads the raw dataset and performs initial cleaning (e.g., handling '?' values, assigning column names).
    -   `preprocess_data.py`: Takes the output of `load_data.py`, performs feature encoding (one-hot) and scaling, splits data into training and testing sets, and saves the preprocessor object.

-   **`training/`**: Contains scripts for model training, hyperparameter tuning, experiment tracking with MLflow, and model evaluation.
    -   `train_models.py`: Trains multiple classification models, logs experiments to MLflow.
    -   `evaluate_models.py`: Fetches run information from MLflow, selects the best model based on specified metrics, and saves its details.

-   **`deployment/`**: Contains the Flask application for serving the trained model as an API.
    -   `app.py`: Loads the best model and the preprocessor, and exposes a `/predict` endpoint to make predictions on new data.
