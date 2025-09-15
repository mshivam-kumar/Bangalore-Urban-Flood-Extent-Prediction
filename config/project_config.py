from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# --- PROJECT DIRECTORIES ---
PROJECT_NAME = "Binary-AI-Flood-Emulator"
PROJECT_DIR = BASE_DIR

# --- DATA DIRECTORIES ---
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
INTERIM_DATA_DIR = DATA_DIR / "interim"
TRAINING_DATA_DIR = PROCESSED_DATA_DIR / "training_data"
VALIDATION_DATA_DIR = PROCESSED_DATA_DIR / "validation_data"
TESTING_DATA_DIR = PROCESSED_DATA_DIR / "testing_data"

# --- NOTEBOOK DIRECTORIES ---
NOTEBOOKS_DIR = BASE_DIR / "notebooks"

# --- MODEL DIRECTORIES ---
CHECKPOINT_DIR = BASE_DIR / "checkpoints"

# --- OUTPUT DIRECTORIES ---
OUTPUT_DIR = BASE_DIR / "outputs"
PLOTS_DIR = OUTPUT_DIR / "plots"

# --- LOGGING DIRECTORIES ---
LOGS_DIR = BASE_DIR / "logs"
DATA_PIPELINE_LOG = LOGS_DIR / "data_pipeline.log"
MODEL_PIPELINE_LOG = LOGS_DIR / "model_pipeline.log"
MAIN_PIPELINE_LOG = LOGS_DIR / "main_pipeline.log"

# --- MODEL HYPERPARAMETERS ---
AOI = [77.45, 12.85, 77.75, 13.10] # Bangalore metropolitan area
AOI_DATABASE_PATH = DATA_DIR / "aoi_database_india.json" # state/district bounding boxes
START_DATE = "2023-07-01"
# END_DATE = "2023-09-30"
END_DATE = "2023-07-05" # For testing purposes, use a small date range
LEARNING_RATE = 0.001
BATCH_SIZE = 32
DOWNLOAD_PATCH_SIZE = 250 # So now if patch size deviate then we can pad near to 256 which is 2^n and make creating and managing model easily
PADDED_PATCH_SIZE = 256 # 256x256 pixels image (interim data to make consistent patches by padding)
DEGREE_SIZE = 1.0 # 1° tiles
SCALE = 10 # 10m resolution

PERMANENT_WATER_THRESHOLD = 50 # Percentage threshold for permanent water detection
