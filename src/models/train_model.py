from config import project_config
from src.utils.log_config import setup_log

log = setup_log(__name__, project_config.MODEL_PIPELINE_LOG)

def run():
    """Main function to run the model training pipeline."""
    log.info("--- Starting Model Training Pipeline ---")

    log.info("Step 1: Loading training data...")
    # Your code to load training data
    log.info("Training data loaded.")
    
    log.info("Step 2: Building model...")
    # Your code to build model
    log.info("Model built successfully.")

    log.info("Step 3: Starting model training...")
    # Your training loop
    log.info("Model training complete.")
    
    log.info(f"Step 4: Saving final model to {project_config.FINAL_MODEL_PATH}")
    # Your code to save model
    log.info("Model saved.")

    log.info("--- Model Training Pipeline Finished ---")

if __name__ == '__main__':
    run()