# Heart Disease Prediction ML System

This project implements an end-to-end machine learning system for predicting the presence of heart disease in patients using the UCI Heart Disease dataset. It includes stages for data loading and preprocessing, training multiple classification models, hyperparameter tuning, comprehensive experiment tracking with MLflow, model evaluation and selection, and deploying the best model as a RESTful API using Flask.

## Features

*   **Data Preprocessing:** Handles missing values, performs categorical feature encoding (One-Hot Encoding), and numerical feature scaling (StandardScaler).
*   **Multiple Model Training:** Trains and evaluates several common classification algorithms:
    *   Logistic Regression
    *   Support Vector Machine (SVM)
    *   Random Forest Classifier
    *   Gradient Boosting Classifier
    *   K-Nearest Neighbors (KNN)
    *   Gaussian Naive Bayes
*   **Hyperparameter Tuning:** Uses GridSearchCV for basic hyperparameter optimization for each model.
*   **MLflow Integration:**
    *   Logs experiment parameters, metrics (Accuracy, Precision, Recall, F1-score, ROC AUC), and trained models for each run.
    *   Facilitates model comparison and selection.
*   **Model Evaluation:** Programmatically selects the best model based on F1-score from MLflow runs.
*   **API Deployment:** Deploys the best model as a Flask API with a `/predict` endpoint.
*   **Input Preprocessing in API:** The API preprocesses raw JSON input using the same preprocessor object saved during the training pipeline, ensuring consistency.
*   **Unit & Integration Testing:** Includes tests for preprocessing, training, evaluation, and the Flask API.
*   **Modularity:** Code is organized into separate scripts for different stages of the ML lifecycle.
*   **Conceptual Model Monitoring:** Outlines strategies for monitoring data drift, concept drift, and model performance in a production environment.

## Directory Structure

```
.
├── data/                  # Stores raw data, processed data, train/test splits,
│                          # the saved preprocessor (preprocessor.joblib), and
│                          # information about the best model (best_model_info.json).
│                          # Generated data files are gitignored.
├── mlruns/                # MLflow tracking data (experiments, runs, models). Gitignored.
├── notebooks/             # (Optional) Jupyter notebooks for initial exploration, analysis.
├── scripts/               # Shell scripts to run various stages of the pipeline
│                          # (e.g., run_training.sh, run_flask_app.sh).
├── src/                   # Python source code for the project.
│   ├── deployment/        # Flask application (app.py) for serving the model.
│   ├── preprocessing/     # Scripts for data loading (load_data.py) and
│   │                      # preprocessing (preprocess_data.py).
│   └── training/          # Scripts for model training (train_models.py) and
│                          # evaluation (evaluate_models.py).
├── tests/                 # Test scripts for different components of the project
│                          # (e.g., test_preprocessing.py, test_flask_api.py).
├── .gitignore             # Specifies intentionally untracked files by Git.
├── requirements.txt       # Lists Python dependencies for the project.
└── README.md              # This file: project overview, setup, and usage instructions.
```

## Setup and Installation

### Prerequisites
*   Python 3.8+
*   `pip` (Python package installer)
*   `git` (for cloning the repository)

### Steps
1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd heart-disease-prediction-ml-system
    ```
    (Replace `<your-repository-url>` with the actual URL of this repository)

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```
    On Windows, use: `venv\Scripts\activate`

3.  **Install dependencies:**
    Install the required Python libraries using the `requirements.txt` file:
    ```bash
    pip install -r requirements.txt
    ```

## Running the ML Pipeline

Follow these steps sequentially to preprocess data, train models, evaluate them, and run the API.

### 1. Data Preprocessing
This stage downloads the raw Cleveland Heart Disease dataset, processes it (handles missing values represented by '?', assigns column names), and then prepares it for model training (feature encoding, scaling). The preprocessor object used is also saved.

Execute the scripts:
```bash
python src/preprocessing/load_data.py
python src/preprocessing/preprocess_data.py
```
**Outputs:**
*   `data/processed.cleveland.data`: Downloaded raw data.
*   `data/processed_heart_disease.csv`: Cleaned data with binary target.
*   `data/preprocessor.joblib`: Saved scikit-learn ColumnTransformer.
*   `data/train_X.csv`, `data/train_y.csv`, `data/test_X.csv`, `data/test_y.csv`: Train/test splits.

### 2. Model Training and Experiment Tracking
This step trains multiple machine learning models, performs hyperparameter tuning for each, and logs all relevant information (parameters, metrics, model artifacts) to MLflow.

Run the training script:
```bash
bash scripts/run_training.sh
```
**Outputs:**
*   MLflow experiments and runs stored in the `mlruns/` directory.
*   Trained model artifacts logged within their respective MLflow run directories.

**View MLflow UI:**
To inspect the training runs, metrics, and artifacts:
```bash
mlflow ui
```
Then open your browser to `http://localhost:5000` (or the address shown in the terminal).

### 3. Model Evaluation and Selection
This script fetches data from all MLflow runs, compares models based on their F1-scores, and saves information about the best-performing model.

Run the evaluation script:
```bash
bash scripts/run_evaluation.sh
```
**Outputs:**
*   `data/best_model_info.json`: A JSON file containing the Run ID, Model URI, and key metrics of the best model.

### 4. Start the Flask API Server
This deploys the best model (identified in the previous step) as a Flask API, making it available for predictions.

Run the Flask application script:
```bash
bash scripts/run_flask_app.sh
```
**Outputs:**
*   A Flask server running, typically on `http://0.0.0.0:5001`.
*   Logs in the terminal indicating the server is up and ready for requests.

## API Usage

The API provides two main endpoints:

*   **`GET /`**: Returns a status message indicating the API is running.
    ```bash
    curl http://localhost:5001/
    ```
    Expected Response:
    ```json
    {
      "message": "Heart Disease Prediction API is running.",
      "model_status": "Loaded",
      "preprocessor_status": "Loaded"
    }
    ```

*   **`POST /predict`**: Accepts JSON data for making heart disease predictions.
    *   **Input JSON Structure:** A JSON object (for a single prediction) or a list of JSON objects (for batch predictions) where keys are feature names. Features should be in their "raw" format (before preprocessing). Example:
        ```json
        {
            "age": 52,
            "sex": 1,        // 0 for female, 1 for male
            "cp": 0,         // Chest pain type (0-3)
            "trestbps": 128, // Resting blood pressure (mm Hg)
            "chol": 204,     // Serum cholesterol (mg/dl)
            "fbs": 1,        // Fasting blood sugar > 120 mg/dl (1 = true; 0 = false)
            "restecg": 1,    // Resting electrocardiographic results (0-2)
            "thalach": 156,  // Maximum heart rate achieved
            "exang": 1,      // Exercise induced angina (1 = yes; 0 = no)
            "oldpeak": 1.0,  // ST depression induced by exercise relative to rest
            "slope": 1,      // Slope of the peak exercise ST segment (0-2)
            "ca": 0,         // Number of major vessels (0-3) colored by fluoroscopy
            "thal": 0        // Thalassemia (e.g., 0, 1, 2, 3 - needs to match values seen in training,
                             // dataset uses 3=normal; 6=fixed defect; 7=reversible defect.
                             // The preprocessor handles this based on training data values.)
        }
        ```
        (Note: For `thal`, the dataset has specific values. The example above uses 0, which might be handled by the preprocessor if it encountered it or has an 'unknown' strategy. Refer to `src/preprocessing/load_data.py` for actual values used if needed.)

    *   **Example `curl` Request (single instance):**
        ```bash
        curl -X POST -H "Content-Type: application/json" \
        -d '{
            "age": 52, "sex": 1, "cp": 0, "trestbps": 128, "chol": 204, "fbs": 1,
            "restecg": 1, "thalach": 156, "exang": 1, "oldpeak": 1.0, "slope": 1,
            "ca": 0, "thal": 0
        }' \
        http://localhost:5001/predict
        ```

    *   **Example Response:**
        ```json
        {
          "predictions": [1] // 0 for no heart disease, 1 for presence of heart disease
        }
        ```

## Running Tests

The project includes a suite of unit and integration tests to ensure components function correctly.

To run all tests:
```bash
python -m unittest discover -s tests -v
```
To run a specific test file (e.g., API tests):
```bash
python -m unittest tests.test_flask_api -v
```
The API tests (`tests/test_flask_api.py`) are designed to be comprehensive and will automatically run the necessary preprocessing, training, and evaluation steps in a test-specific environment to gather required artifacts before testing the API endpoints.

## Model Monitoring (Conceptual)

Effective machine learning systems require continuous monitoring in production to ensure they maintain performance and reliability over time. Without monitoring, issues like data drift, concept drift, or model staleness can go unnoticed, leading to degraded performance and potentially harmful outcomes.

This project provides the foundational elements for deploying an ML model. A full-fledged production system would incorporate a robust monitoring strategy. Key areas to monitor include:

### 1. Data Drift
Changes in the statistical properties of input features over time compared to the data the model was trained on. For example, if the average resting blood pressure (`trestbps`) or serum cholesterol (`chol`) of patients being scored by the API starts to differ significantly from the training dataset, the model's assumptions may no longer hold.

*   **Detection Strategies:**
    *   **Statistical Tests:** Use tests like Kolmogorov-Smirnov (KS) for numerical features or Chi-Squared for categorical features to compare distributions. Population Stability Index (PSI) can also be useful.
    *   **Monitoring:** Collect incoming prediction request data (features) and periodically (e.g., daily, weekly) calculate descriptive statistics and compare distributions against the training set.
*   **Potential Tools:**
    *   Python libraries: `scipy.stats`, `numpy`.
    *   Specialized libraries: `evidentlyai`, `deepchecks`, `great_expectations`.
    *   Custom scripting and visualization dashboards.

### 2. Concept Drift (Model Drift)
Changes in the underlying relationship between input features and the target variable (heart disease diagnosis). The model's learned patterns may become less accurate as real-world relationships evolve.

*   **Detection Strategies:**
    *   **Performance Monitoring:** If new labeled data becomes available (e.g., new patient outcomes), periodically re-evaluate the deployed model's F1-score, accuracy, ROC AUC, etc., on this new data. A significant drop is a strong indicator of concept drift.
    *   **Adaptive Windowing (ADWIN):** Algorithms designed to detect changes in data streams, which can be adapted for monitoring model error rates or other performance indicators.
    *   **Proxy Metrics:** If direct performance feedback is delayed, monitor drift in features known to be highly predictive. For instance, if a key diagnostic marker's typical values shift, it might signal that the model's understanding is outdated.
*   **Potential Tools:**
    *   Monitoring systems that can track model performance against ground truth.
    *   Libraries offering drift detection algorithms (e.g., `river`).

### 3. Prediction/Output Drift
Changes in the distribution of the model's predictions. For example, if the model suddenly starts predicting a much higher or lower percentage of positive heart disease cases without a known corresponding change in the input population.

*   **Detection Strategies:**
    *   Monitor the distribution of predicted probabilities (if the model outputs them).
    *   Track the frequency of each predicted class (0 or 1).
    *   Significant, unexplained shifts can be an early warning that the model might be behaving unexpectedly, possibly due to data drift or an emerging issue.
*   **Potential Tools:**
    *   Logging and dashboarding systems to visualize prediction distributions over time.

### 4. Model Performance & Operational Metrics
Beyond data and concept drift, it's crucial to monitor the operational health of the deployed model and its predictive performance.

*   **Operational Metrics:**
    *   **API Latency:** Time taken to serve a prediction.
    *   **API Throughput:** Number of requests served per unit of time.
    *   **API Error Rates:** Frequency of HTTP errors (5xx, 4xx).
*   **Model Performance (with new labeled data):**
    *   Regularly re-calculate accuracy, precision, recall, F1-score, ROC AUC.
*   **Potential Tools:**
    *   API Gateway logs, web server logs.
    *   Application Performance Monitoring (APM) tools (e.g., Prometheus, Grafana, Datadog, New Relic).
    *   MLflow for tracking performance of re-evaluated models.

### Retraining Strategy
A clear strategy for when and how to retrain the model is essential.

*   **Triggers for Retraining:**
    *   Significant data or concept drift detected.
    *   Model performance metrics fall below predefined thresholds.
    *   Scheduled retraining (e.g., monthly, quarterly) regardless of drift, to incorporate new data.
*   **Process:**
    *   Automated retraining pipelines are ideal, using the latest curated dataset.
    *   MLflow (as used in this project) is valuable for versioning new datasets, code, models, and comparing performance against previously deployed models.
    *   A/B testing or shadow deployment can be used to validate a newly retrained model before it fully replaces the current production model.

### Logging and Alerting
Comprehensive logging of inputs, predictions, and any detected drift or performance degradation is vital. Alerts should be configured to notify relevant teams when issues arise.

*   **Potential Tools:**
    *   Logging frameworks: ELK Stack (Elasticsearch, Logstash, Kibana), Splunk.
    *   Alerting systems: Integrated with monitoring tools (e.g., Grafana alerting, Prometheus Alertmanager).

**Conclusion for Monitoring:**
This project lays the groundwork for model deployment. Implementing a comprehensive monitoring solution as outlined above would be a critical next step for taking such a system into a reliable production environment. The specific tools and depth of implementation would depend on the application's criticality, available infrastructure, and resources.
