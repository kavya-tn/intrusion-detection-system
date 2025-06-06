import os
import subprocess
import shutil
import unittest
import sys

# Add src to Python path to allow direct import of modules for potential pre-checks
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

# Attempt to import MODELS_GRID for test validation
try:
    from training.train_models import MODELS_GRID
except ImportError:
    print("Warning: Could not import MODELS_GRID from training.train_models. Test for number of runs might be inaccurate.")
    # Define a fallback or skip the specific assertion if MODELS_GRID is crucial and not found.
    # For now, let it proceed; the test might fail at the assertion if MODELS_GRID is not defined.
    MODELS_GRID = {} # Fallback to an empty dict if import fails


class TestTrainingPipeline(unittest.TestCase):

    MLRUNS_DIR = "mlruns"
    SCRIPT_PATH = "scripts/run_training.sh"
    # Define data paths that train_models.py expects
    PROCESSED_DATA_PATH = "data/processed_heart_disease.csv"
    TRAIN_X_PATH = "data/train_X.csv"
    TRAIN_Y_PATH = "data/train_y.csv"
    TEST_X_PATH = "data/test_X.csv"
    TEST_Y_PATH = "data/test_y.csv"


    @classmethod
    def setUpClass(cls):
        """Ensure data files exist before running tests. If not, create them."""
        # Check if data files exist, if not, run preprocessing.
        # This makes the test suite more robust to being run in a clean environment.
        if not all(os.path.exists(p) for p in [cls.TRAIN_X_PATH, cls.TRAIN_Y_PATH, cls.TEST_X_PATH, cls.TEST_Y_PATH]):
            print("Preprocessed data not found. Running preprocessing scripts first...")
            try:
                from preprocessing import load_data as ld, preprocess_data as ppd

                # Create data directory if it doesn't exist
                if not os.path.exists("data"):
                    os.makedirs("data")

                # Run load_data if its output is missing
                if not os.path.exists(cls.PROCESSED_DATA_PATH):
                    print(f"Running load_data.py as {cls.PROCESSED_DATA_PATH} is missing...")
                    ld.load_and_process_data()

                # Run preprocess_data
                print("Running preprocess_data.py...")
                ppd.preprocess_data()
                print("Preprocessing scripts executed.")
            except ImportError:
                print("Failed to import preprocessing modules. Ensure they are in src/preprocessing and PYTHONPATH is set.")
                raise
            except Exception as e:
                print(f"Error running preprocessing scripts during test setup: {e}")
                raise

    def setUp(self):
        """Remove mlruns directory before each test to ensure a clean state."""
        if os.path.exists(self.MLRUNS_DIR):
            shutil.rmtree(self.MLRUNS_DIR)
        print(f"Removed {self.MLRUNS_DIR} for a clean test run.")

    def test_training_script_runs_and_creates_mlruns(self):
        """
        Tests that the training script runs without errors and creates the mlruns directory.
        """
        print(f"Running test: test_training_script_runs_and_creates_mlruns using {self.SCRIPT_PATH}")

        # Ensure the script is executable
        if not os.access(self.SCRIPT_PATH, os.X_OK):
            print(f"Script {self.SCRIPT_PATH} not executable, attempting chmod +x")
            subprocess.run(['chmod', '+x', self.SCRIPT_PATH], check=True)

        try:
            # Run the training script
            result = subprocess.run(
                [self.SCRIPT_PATH],
                capture_output=True,
                text=True,
                check=True  # This will raise CalledProcessError if the script exits with a non-zero code
            )
            print("Training script stdout:")
            print(result.stdout)
            if result.stderr:
                print("Training script stderr:")
                print(result.stderr)
        except subprocess.CalledProcessError as e:
            print(f"Error running {self.SCRIPT_PATH}:")
            print(f"Return code: {e.returncode}")
            print(f"Stdout: {e.stdout}")
            print(f"Stderr: {e.stderr}")
            self.fail(f"Training script {self.SCRIPT_PATH} failed with return code {e.returncode}")

        self.assertTrue(os.path.exists(self.MLRUNS_DIR), f"{self.MLRUNS_DIR} was not created.")
        self.assertTrue(os.path.isdir(self.MLRUNS_DIR), f"{self.MLRUNS_DIR} is not a directory.")
        print(f"Test passed: {self.MLRUNS_DIR} created successfully.")

        # Optional: Check for experiment runs within mlruns
        # MLflow creates a default experiment '0' if no experiment is set.
        experiment_0_path = os.path.join(self.MLRUNS_DIR, "0")
        self.assertTrue(os.path.exists(experiment_0_path) and os.path.isdir(experiment_0_path),
                        "Default experiment '0' not found in mlruns.")

        # Count number of actual runs (subdirectories in experiment_0_path that are not 'meta.yaml')
        runs = [d for d in os.listdir(experiment_0_path)
                if os.path.isdir(os.path.join(experiment_0_path, d)) and d != 'meta.yaml']

        # We expect 6 models to be trained
        expected_runs = len(MODELS_GRID) if 'MODELS_GRID' in sys.modules['training.train_models'].__dict__ else 6
        # A bit of a hack to get MODELS_GRID length without direct import if script changes location
        # A better way would be to parse this from train_models.py or have it in a shared config

        # For simplicity, we'll check if there's at least one run, as the exact number can be fragile
        expected_runs = len(MODELS_GRID)
        self.assertEqual(len(runs), expected_runs, f"Expected {expected_runs} runs, found {len(runs)}.")
        # self.assertTrue(len(runs) > 0, "No runs found in the default experiment.")
        print(f"Found {len(runs)} runs in the default experiment. Check for {expected_runs} runs passed.")


    @classmethod
    def tearDownClass(cls):
        """Clean up mlruns directory after all tests in the class are done."""
        if os.path.exists(cls.MLRUNS_DIR):
            shutil.rmtree(cls.MLRUNS_DIR)
        print(f"Cleaned up {cls.MLRUNS_DIR} after all tests.")

        # Clean up data files created during setUpClass
        # This makes the test suite fully clean up after itself.
        data_files_to_remove = [
            cls.PROCESSED_DATA_PATH, cls.TRAIN_X_PATH, cls.TRAIN_Y_PATH,
            cls.TEST_X_PATH, cls.TEST_Y_PATH, "data/processed.cleveland.data"
        ]
        for f_path in data_files_to_remove:
            if os.path.exists(f_path):
                os.remove(f_path)
        if os.path.exists("data/processed.cleveland.data.1"): # if multiple downloads happened
            os.remove("data/processed.cleveland.data.1")
        print("Cleaned up generated data files.")


if __name__ == '__main__':
    unittest.main()
