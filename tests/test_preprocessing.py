import os
import pandas as pd
import sys

# Add src to Python path to allow direct import of modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from preprocessing import load_data, preprocess_data

# Define file paths (consistent with the scripts)
RAW_DATA_PATH = "data/processed.cleveland.data"
PROCESSED_DATA_PATH = "data/processed_heart_disease.csv"
TRAIN_X_PATH = "data/train_X.csv"
TRAIN_Y_PATH = "data/train_y.csv"
TEST_X_PATH = "data/test_X.csv"
TEST_Y_PATH = "data/test_y.csv"

def cleanup_files():
    """Removes generated files to ensure tests are idempotent."""
    files_to_remove = [
        RAW_DATA_PATH, PROCESSED_DATA_PATH,
        TRAIN_X_PATH, TRAIN_Y_PATH, TEST_X_PATH, TEST_Y_PATH
    ]
    for f_path in files_to_remove:
        if os.path.exists(f_path):
            os.remove(f_path)
    print("Cleaned up generated files.")

def test_load_data_creates_processed_file():
    """Tests that load_data.py creates processed_heart_disease.csv."""
    cleanup_files() # Ensure clean state
    print("Running test_load_data_creates_processed_file...")
    load_data.load_and_process_data()
    assert os.path.exists(PROCESSED_DATA_PATH), f"{PROCESSED_DATA_PATH} was not created."
    print(f"Test passed: {PROCESSED_DATA_PATH} created.")
    # Basic check on content
    df = pd.read_csv(PROCESSED_DATA_PATH)
    assert not df.isnull().values.any(), "There are missing values in the processed file."
    assert "target" in df.columns, "Target column is missing."
    assert df["target"].isin([0, 1]).all(), "Target column is not binary."
    print("Content checks for processed file passed.")


def test_preprocess_data_creates_split_files():
    """Tests that preprocess_data.py creates train/test split files."""
    # Ensure load_data has run first if processed file doesn't exist
    if not os.path.exists(PROCESSED_DATA_PATH):
        print(f"{PROCESSED_DATA_PATH} not found, running load_data first...")
        load_data.load_and_process_data()

    print("Running test_preprocess_data_creates_split_files...")
    preprocess_data.preprocess_data()
    assert os.path.exists(TRAIN_X_PATH), f"{TRAIN_X_PATH} was not created."
    assert os.path.exists(TRAIN_Y_PATH), f"{TRAIN_Y_PATH} was not created."
    assert os.path.exists(TEST_X_PATH), f"{TEST_X_PATH} was not created."
    assert os.path.exists(TEST_Y_PATH), f"{TEST_Y_PATH} was not created."
    print("Test passed: Train/test split files created.")

def test_preprocessed_data_properties():
    """Checks properties of the preprocessed data."""
    if not os.path.exists(TRAIN_X_PATH): # Ensure previous steps ran
        test_load_data_creates_processed_file() # This will also call cleanup
        preprocess_data.preprocess_data()

    print("Running test_preprocessed_data_properties...")
    train_x_df = pd.read_csv(TRAIN_X_PATH)
    raw_df = pd.read_csv(PROCESSED_DATA_PATH) # Load the state before one-hot encoding

    # Check for no missing values
    assert not train_x_df.isnull().values.any(), "Missing values found in preprocessed training data (X_train)."

    train_y_df = pd.read_csv(TRAIN_Y_PATH)
    assert not train_y_df.isnull().values.any(), "Missing values found in preprocessed training labels (y_train)."

    test_x_df = pd.read_csv(TEST_X_PATH)
    assert not test_x_df.isnull().values.any(), "Missing values found in preprocessed test data (X_test)."

    test_y_df = pd.read_csv(TEST_Y_PATH)
    assert not test_y_df.isnull().values.any(), "Missing values found in preprocessed test labels (y_test)."
    print("No missing values check passed for all split files.")

    # Check if number of columns increased (due to one-hot encoding)
    # This depends on the actual categorical features identified and if they have multiple categories
    # The 'target' column is dropped from raw_df for this comparison
    # The exact number of columns after OHE can be tricky to predict without running,
    # so we check if it's greater than the original number of features.
    original_feature_count = raw_df.shape[1] - 1 # -1 for the target column
    assert train_x_df.shape[1] > original_feature_count, \
        f"Number of columns in preprocessed data ({train_x_df.shape[1]}) is not greater than original features ({original_feature_count}). OHE might not have worked as expected."
    print(f"Column count check passed: {train_x_df.shape[1]} (processed) > {original_feature_count} (original features).")
    cleanup_files() # Clean up at the end of all tests in this function

if __name__ == "__main__":
    # Run tests sequentially
    print("Starting preprocessing tests...")
    test_load_data_creates_processed_file()
    test_preprocess_data_creates_split_files()
    test_preprocessed_data_properties() # This will also call cleanup at its end
    print("All preprocessing tests finished.")
