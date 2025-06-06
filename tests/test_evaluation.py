import os
import subprocess
import shutil
import unittest
import json
import sys

# Add src to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

class TestEvaluationPipeline(unittest.TestCase):

    MLRUNS_DIR = "mlruns"
    BEST_MODEL_INFO_PATH = "data/best_model_info.json"
    RUN_TRAINING_SCRIPT_PATH = "scripts/run_training.sh"
    RUN_EVALUATION_SCRIPT_PATH = "scripts/run_evaluation.sh"

    @classmethod
    def setUpClass(cls):
        """Ensure training has run and mlruns exists."""
        print("Setting up TestEvaluationPipeline class...")
        # Clean up any existing mlruns and best_model_info.json to ensure fresh run
        if os.path.exists(cls.MLRUNS_DIR):
            shutil.rmtree(cls.MLRUNS_DIR)
        if os.path.exists(cls.BEST_MODEL_INFO_PATH):
            os.remove(cls.BEST_MODEL_INFO_PATH)

        print(f"Running training script {cls.RUN_TRAINING_SCRIPT_PATH} to generate data for evaluation tests...")
        try:
            # Ensure training script is executable
            if not os.access(cls.RUN_TRAINING_SCRIPT_PATH, os.X_OK):
                 subprocess.run(['chmod', '+x', cls.RUN_TRAINING_SCRIPT_PATH], check=True)

            result = subprocess.run(
                [cls.RUN_TRAINING_SCRIPT_PATH],
                capture_output=True, text=True, check=True,
                # Set MLFLOW_TRACKING_URI for the training script if it doesn't set it itself robustly
                env=dict(os.environ, MLFLOW_TRACKING_URI="./mlruns")
            )
            print("Training script stdout:")
            # print(result.stdout) # Can be very verbose
            if result.stderr:
                print("Training script stderr:")
                # print(result.stderr) # Can be very verbose
            if not os.path.exists(cls.MLRUNS_DIR) or not os.listdir(cls.MLRUNS_DIR): # Check if mlruns exists and is not empty
                 raise Exception("MLruns directory not created or is empty after training script execution.")
            # Further check if the default experiment '0' has any runs, can be more specific if needed
            exp_0_path = os.path.join(cls.MLRUNS_DIR, "0")
            if not os.path.exists(exp_0_path) or not any(os.path.isdir(os.path.join(exp_0_path, item)) for item in os.listdir(exp_0_path) if item != 'meta.yaml'):
                raise Exception("Default experiment '0' not found in mlruns or contains no run directories.")
            print("Training script executed successfully, mlruns populated.")

        except subprocess.CalledProcessError as e:
            print(f"Error running {cls.RUN_TRAINING_SCRIPT_PATH} during setup:")
            print(f"Return code: {e.returncode}")
            print(f"Stdout: {e.stdout}")
            print(f"Stderr: {e.stderr}")
            raise Exception(f"Failed to execute training script: {e}")
        except Exception as e:
            print(f"An unexpected error occurred during setUpClass: {e}")
            raise

    def setUp(self):
        """Remove best_model_info.json before each test."""
        if os.path.exists(self.BEST_MODEL_INFO_PATH):
            os.remove(self.BEST_MODEL_INFO_PATH)
        print(f"Removed {self.BEST_MODEL_INFO_PATH} for a clean test run.")

    def test_evaluation_script_runs_and_creates_info_file(self):
        """
        Tests that evaluate_models.py runs and creates best_model_info.json.
        """
        print(f"Running test: test_evaluation_script_runs_and_creates_info_file using {self.RUN_EVALUATION_SCRIPT_PATH}")

        # Ensure evaluation script is executable
        if not os.access(self.RUN_EVALUATION_SCRIPT_PATH, os.X_OK):
            print(f"Script {self.RUN_EVALUATION_SCRIPT_PATH} not executable, attempting chmod +x")
            subprocess.run(['chmod', '+x', self.RUN_EVALUATION_SCRIPT_PATH], check=True)

        try:
            result = subprocess.run(
                [self.RUN_EVALUATION_SCRIPT_PATH],
                capture_output=True, text=True, check=True,
                env=dict(os.environ, MLFLOW_TRACKING_URI="./mlruns", PYTHONPATH=os.getenv("PYTHONPATH", "") + ":" + os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
            )
            print("Evaluation script stdout:")
            # print(result.stdout) # Can be verbose
            if result.stderr:
                print("Evaluation script stderr:")
                # print(result.stderr) # Can be verbose
        except subprocess.CalledProcessError as e:
            print(f"Error running {self.RUN_EVALUATION_SCRIPT_PATH}:")
            print(f"Return code: {e.returncode}")
            print(f"Stdout: {e.stdout}")
            print(f"Stderr: {e.stderr}")
            self.fail(f"Evaluation script {self.RUN_EVALUATION_SCRIPT_PATH} failed with return code {e.returncode}")

        self.assertTrue(os.path.exists(self.BEST_MODEL_INFO_PATH),
                        f"{self.BEST_MODEL_INFO_PATH} was not created.")

        with open(self.BEST_MODEL_INFO_PATH, 'r') as f:
            info = json.load(f)

        self.assertIn("run_id", info, "run_id not found in best_model_info.json")
        self.assertIn("model_uri", info, "model_uri not found in best_model_info.json")
        self.assertIn("model_name", info, "model_name not found in best_model_info.json")
        self.assertIn("f1_score", info, "f1_score not found in best_model_info.json")
        self.assertTrue(info["run_id"] is not None and info["run_id"] != "")
        self.assertTrue(info["model_uri"].startswith("runs:/"))

        print(f"Test passed: {self.BEST_MODEL_INFO_PATH} created and contains expected keys.")

    @classmethod
    def tearDownClass(cls):
        """Clean up generated files after all tests."""
        print("Tearing down TestEvaluationPipeline class...")
        if os.path.exists(cls.MLRUNS_DIR):
            shutil.rmtree(cls.MLRUNS_DIR)
        if os.path.exists(cls.BEST_MODEL_INFO_PATH):
            os.remove(cls.BEST_MODEL_INFO_PATH)

        # Remove data files created by preprocessing during training script run
        # These paths are from test_training.py's setUpClass, assuming they are consistent
        data_files_to_remove = [
            "data/processed_heart_disease.csv", "data/train_X.csv", "data/train_y.csv",
            "data/test_X.csv", "data/test_y.csv", "data/processed.cleveland.data"
        ]
        for f_path in data_files_to_remove:
            if os.path.exists(f_path):
                try:
                    os.remove(f_path)
                except OSError as e:
                    print(f"Warning: Could not remove {f_path}: {e}")

        # Remove data directory if empty
        if os.path.exists("data") and not os.listdir("data"):
            try:
                os.rmdir("data")
            except OSError as e:
                 print(f"Warning: Could not remove data directory: {e}")

        print("Cleaned up generated files (mlruns, best_model_info.json, data files).")

if __name__ == '__main__':
    unittest.main()
