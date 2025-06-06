import os
import sys
import unittest
import json
import shutil
import subprocess
import pandas as pd # For crafting test payload
import time # For waiting for server to start if run separately

# Add src directory to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

# Import the Flask app instance
# We need to ensure MLFLOW_TRACKING_URI is set BEFORE app is imported if app.py uses it at module level
os.environ['MLFLOW_TRACKING_URI'] = os.path.abspath('./mlruns_test_api') # Use a dedicated mlruns for this test

# Now import the app
from deployment.app import app, load_model_and_preprocessor, EXPECTED_RAW_FEATURE_COLUMNS


class TestFlaskApi(unittest.TestCase):

    PREPROCESSOR_PATH_ORIGINAL = "data/preprocessor.joblib"
    BEST_MODEL_INFO_PATH_ORIGINAL = "data/best_model_info.json"

    # Test specific paths for artifacts to avoid interference
    TEST_DATA_DIR = "data_test_api"
    PREPROCESSOR_PATH_TEST = os.path.join(TEST_DATA_DIR, "preprocessor.joblib")
    BEST_MODEL_INFO_PATH_TEST = os.path.join(TEST_DATA_DIR, "best_model_info.json")
    MLRUNS_TEST_API_DIR = os.path.abspath("./mlruns_test_api") # Ensure this is used by app

    @classmethod
    def setUpClass(cls):
        print("Setting up TestFlaskApi class...")

        # Create test-specific directories
        os.makedirs(cls.TEST_DATA_DIR, exist_ok=True)
        # Do NOT pre-create MLRUNS_TEST_API_DIR; let MLflow create it.
        # This ensures the default experiment '0' is also created correctly by MLflow.
        if os.path.exists(cls.MLRUNS_TEST_API_DIR): # Clean up if exists from previous failed run
            shutil.rmtree(cls.MLRUNS_TEST_API_DIR)

        # --- Run prerequisite scripts to generate artifacts ---
        # 1. Preprocessing (to get preprocessor.joblib and data for training)
        print("Running preprocessing to generate preprocessor.joblib and data...")
        # First, ensure data directory used by load_data.py exists. load_data.py creates ./data if not present.
        os.makedirs("data", exist_ok=True)
        subprocess.run([sys.executable, "src/preprocessing/load_data.py"], check=True)
        subprocess.run([sys.executable, "src/preprocessing/preprocess_data.py"], check=True)

        # Copy generated preprocessor to test data directory
        if os.path.exists(cls.PREPROCESSOR_PATH_ORIGINAL):
            shutil.copy(cls.PREPROCESSOR_PATH_ORIGINAL, cls.PREPROCESSOR_PATH_TEST)
        else:
            raise FileNotFoundError(f"{cls.PREPROCESSOR_PATH_ORIGINAL} not found after running preprocess_data.py")

        # 2. Training (to populate mlruns for evaluation)
        print("Running training script...")
        # run_training.sh sets MLFLOW_TRACKING_URI to ./mlruns, we need it to be ./mlruns_test_api
        training_env = os.environ.copy()
        training_env["MLFLOW_TRACKING_URI"] = cls.MLRUNS_TEST_API_DIR
        training_env["PYTHONPATH"] = os.path.abspath("src") + os.pathsep + training_env.get("PYTHONPATH", "")

        # Ensure script/run_training.sh is executable
        run_training_script_path = "scripts/run_training.sh"
        if not os.access(run_training_script_path, os.X_OK):
            subprocess.run(['chmod', '+x', run_training_script_path], check=True)

        subprocess.run([run_training_script_path], check=True, env=training_env)

        # 3. Evaluation (to get best_model_info.json)
        print("Running evaluation script...")
        evaluation_env = os.environ.copy()
        evaluation_env["MLFLOW_TRACKING_URI"] = cls.MLRUNS_TEST_API_DIR
        evaluation_env["PYTHONPATH"] = os.path.abspath("src") + os.pathsep + evaluation_env.get("PYTHONPATH", "")
        evaluation_env["DATA_DIR_OVERRIDE"] = cls.TEST_DATA_DIR # To make evaluate_models save best_model_info here. Needs change in evaluate_models.py
                                                              # For now, we will copy it.

        # Ensure script/run_evaluation.sh is executable
        run_evaluation_script_path = "scripts/run_evaluation.sh"
        if not os.access(run_evaluation_script_path, os.X_OK):
            subprocess.run(['chmod', '+x', run_evaluation_script_path], check=True)

        subprocess.run([run_evaluation_script_path], check=True, env=evaluation_env)

        # Copy generated best_model_info.json to test data directory
        if os.path.exists(cls.BEST_MODEL_INFO_PATH_ORIGINAL):
            shutil.copy(cls.BEST_MODEL_INFO_PATH_ORIGINAL, cls.BEST_MODEL_INFO_PATH_TEST)
        else:
            raise FileNotFoundError(f"{cls.BEST_MODEL_INFO_PATH_ORIGINAL} not found after running evaluate_models.py")

        # --- Configure Flask app for testing ---
        app.config['TESTING'] = True
        # Override paths for the app to use test-specific artifacts
        app.config['BEST_MODEL_INFO_PATH'] = cls.BEST_MODEL_INFO_PATH_TEST
        app.config['PREPROCESSOR_PATH'] = cls.PREPROCESSOR_PATH_TEST

        # Load model and preprocessor within the app context for testing
        # This is crucial because the app usually loads this on startup.
        # For tests, we need to trigger it after setting up test paths.
        with app.app_context():
            try:
                # Point app's global vars to test paths before loading
                import src.deployment.app as flask_app_module
                flask_app_module.BEST_MODEL_INFO_PATH = cls.BEST_MODEL_INFO_PATH_TEST
                flask_app_module.PREPROCESSOR_PATH = cls.PREPROCESSOR_PATH_TEST
                load_model_and_preprocessor()
            except Exception as e:
                print(f"Error loading model/preprocessor in test setup: {e}")
                raise

        cls.client = app.test_client()
        print("TestFlaskApi setup complete.")

    @classmethod
    def tearDownClass(cls):
        print("Tearing down TestFlaskApi class...")
        # Clean up test-specific directories and files
        if os.path.exists(cls.TEST_DATA_DIR):
            shutil.rmtree(cls.TEST_DATA_DIR)
        if os.path.exists(cls.MLRUNS_TEST_API_DIR):
            shutil.rmtree(cls.MLRUNS_TEST_API_DIR)

        # Clean up main data files created by prerequisite scripts
        main_data_files = [
            "data/processed.cleveland.data",
            "data/processed_heart_disease.csv",
            "data/train_X.csv", "data/train_y.csv",
            "data/test_X.csv", "data/test_y.csv",
            cls.PREPROCESSOR_PATH_ORIGINAL,
            cls.BEST_MODEL_INFO_PATH_ORIGINAL
        ]
        for f_path in main_data_files:
            if os.path.exists(f_path):
                os.remove(f_path)
        if os.path.exists("data") and not os.listdir("data"):
            os.rmdir("data")

        print("TestFlaskApi teardown complete.")

    def test_01_home_endpoint(self):
        """Test the home endpoint."""
        print("Testing / endpoint...")
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        json_data = response.get_json()
        self.assertEqual(json_data['message'], "Heart Disease Prediction API is running.")
        self.assertEqual(json_data['model_status'], "Loaded")
        self.assertEqual(json_data['preprocessor_status'], "Loaded")
        print("/ endpoint test passed.")

    def test_02_predict_endpoint_valid_input_single(self):
        """Test the /predict endpoint with valid single instance input."""
        print("Testing /predict endpoint (valid single input)...")
        # Sample data (use actual column names from EXPECTED_RAW_FEATURE_COLUMNS)
        # This data should be in the "raw" format before preprocessing
        sample_input = {
            "age": 52, "sex": 1, "cp": 0, "trestbps": 128, "chol": 204, "fbs": 1,
            "restecg": 1, "thalach": 156, "exang": 1, "oldpeak": 1.0, "slope": 1,
            "ca": 0, "thal": 0 # Note: thal=0 is unusual based on dataset description (3,6,7)
                               # but preprocessor should handle it if it saw it or via 'handle_unknown'
        }
        # Ensure all expected columns are present, matching app.py's logic
        for col in EXPECTED_RAW_FEATURE_COLUMNS:
            if col not in sample_input:
                sample_input[col] = 0 # Or a more sensible default / NaN if appropriate

        response = self.client.post('/predict', json=sample_input)
        self.assertEqual(response.status_code, 200, f"Error: {response.get_data(as_text=True)}")
        json_data = response.get_json()
        self.assertIn('predictions', json_data)
        self.assertIsInstance(json_data['predictions'], list)
        self.assertTrue(len(json_data['predictions']) == 1)
        self.assertIn(json_data['predictions'][0], [0, 1])
        print("/predict endpoint (valid single input) test passed.")

    def test_03_predict_endpoint_valid_input_batch(self):
        """Test the /predict endpoint with valid batch input."""
        print("Testing /predict endpoint (valid batch input)...")
        sample_input_batch = [
            {
                "age": 63, "sex": 1, "cp": 3, "trestbps": 145, "chol": 233, "fbs": 1,
                "restecg": 0, "thalach": 150, "exang": 0, "oldpeak": 2.3, "slope": 0,
                "ca": 0, "thal": 1
            },
            {
                "age": 37, "sex": 1, "cp": 2, "trestbps": 130, "chol": 250, "fbs": 0,
                "restecg": 1, "thalach": 187, "exang": 0, "oldpeak": 3.5, "slope": 0,
                "ca": 0, "thal": 2
            }
        ]
        # Ensure all expected columns are present
        processed_batch = []
        for record in sample_input_batch:
            new_record = record.copy()
            for col in EXPECTED_RAW_FEATURE_COLUMNS:
                if col not in new_record:
                    new_record[col] = 0
            processed_batch.append(new_record)

        response = self.client.post('/predict', json=processed_batch)
        self.assertEqual(response.status_code, 200, f"Error: {response.get_data(as_text=True)}")
        json_data = response.get_json()
        self.assertIn('predictions', json_data)
        self.assertIsInstance(json_data['predictions'], list)
        self.assertEqual(len(json_data['predictions']), 2)
        for pred in json_data['predictions']:
            self.assertIn(pred, [0, 1])
        print("/predict endpoint (valid batch input) test passed.")


    def test_04_predict_endpoint_invalid_input_missing_fields(self):
        """Test /predict with missing fields."""
        print("Testing /predict endpoint (missing fields)...")
        invalid_input = {"age": 50, "sex": 1} # Missing many fields
        response = self.client.post('/predict', json=invalid_input)
        # The app.py currently fills missing EXPECTED_RAW_FEATURE_COLUMNS with NaN,
        # so this will result in a valid prediction if the model/preprocessor handles NaNs.
        self.assertEqual(response.status_code, 200, f"Error: {response.get_data(as_text=True)}")
        json_data = response.get_json()
        self.assertIn('predictions', json_data)
        print("/predict endpoint (missing fields, current app behavior) test passed.")


    def test_05_predict_endpoint_invalid_input_empty_json(self):
        """Test /predict with empty JSON."""
        print("Testing /predict endpoint (empty JSON)...")
        response = self.client.post('/predict', json={})
        # app.py should return 400 if input_data is empty after request.get_json()
        self.assertEqual(response.status_code, 400, f"Error: {response.get_data(as_text=True)}")
        json_data = response.get_json()
        self.assertIn('error', json_data)
        self.assertEqual(json_data['error'], "No input data provided.")
        print("/predict endpoint (empty JSON) test passed.")

    def test_06_predict_endpoint_invalid_input_bad_type(self):
        """Test /predict with non-JSON payload (e.g. string)."""
        print("Testing /predict endpoint (bad payload type)...")
        # Sending data that is not json and without content_type='application/json'
        # request.get_json() will fail, raising werkzeug.exceptions.UnsupportedMediaType (415)
        response = self.client.post('/predict', data="this is not json")
        self.assertEqual(response.status_code, 415)
        # Flask's default error handler for 415 should return a JSON response
        # if the request Accept header indicates JSON, or if it's a direct API call.
        json_data = response.get_json()
        self.assertIn('name', json_data) # Default Werkzeug/Flask error JSON has 'name' and 'description'
        self.assertEqual(json_data['name'], "Unsupported Media Type")
        print("/predict endpoint (bad payload type) test passed.")

if __name__ == '__main__':
    unittest.main()
