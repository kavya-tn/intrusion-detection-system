# Data Directory (`data`)

This directory stores all data artifacts generated and used by the ML pipeline.

## Files:

The pipeline generates the following files in this directory:

-   **`processed.cleveland.data`**: The raw dataset downloaded from UCI, after very minimal initial touch-ups if any by `load_data.py` (though typically it's saved as is).
-   **`processed_heart_disease.csv`**: The cleaned and processed version of the data, with missing values handled and the target variable made binary. This is the direct input to the main preprocessing script.
-   **`preprocessor.joblib`**: The scikit-learn `ColumnTransformer` object (fitted on the training data) saved using `joblib`. This is used by the Flask API to preprocess incoming prediction requests consistently.
-   **`train_X.csv`**: Features for the training set.
-   **`train_y.csv`**: Target variable for the training set.
-   **`test_X.csv`**: Features for the test set.
-   **`test_y.csv`**: Target variable for the test set.
-   **`best_model_info.json`**: A JSON file containing information about the best model selected during the evaluation phase (e.g., its MLflow Run ID, Model URI, and key metrics). This file is used by the deployment script to load the correct model.

## Git Policy:

Generated data files (like `.csv`, `.joblib`, `.json` containing run-specific info) in this directory are generally **not meant to be committed to Git**. They are artifacts produced by running the pipeline scripts. The `.gitignore` file is configured to exclude them.

Only commit files to this directory if they are static assets that are part of the project's source, not generated artifacts.
