from src.data import make_dataset
from src.utils.log_config import setup_log
from config import project_config
from src.data.download import start_download
from src.data.preprocess_data import start_preprocess_data
from src.data.make_dataset import start_make_dataset
from src.models.train_model import start_train # this wraps the train and resume_training both functions
from src.models.model import BinaryFloodClassifier_dem_input, BinaryFloodClassifier_dem_s1_vh_or_water_channels 

log = setup_log(__name__, project_config.MAIN_PIPELINE_LOG, console_output=True) # printing logs to console for main pipeline

def run_pipeline():
    # Executes the full project pipeline.
    log.info("

>>>>>> Starting the entire project pipeline <<<<<<")
    train_loader, val_loader, test_loader = None, None, None  # Initialize data loaders
    
    tag = "dem_rainfall_s1"  # Tag for the dataset
    # tag = "dem_input"  # Tag for the dataset
    # tag = "dem_s1_vh_band_input"  # Tag for the dataset
    # tag = "dem_s1_water_band_input"  # Tag for the dataset
    
    # Flags to control the execution of different stages
    download_data_flag = False # Set this true before downloading data
    interim_data_preprocess_data_flag = False
    make_dataset_flag = False # Once saved the dataset for a particular tag, we can load it again. So make it false then. tag processed:['dem_input']
    load_dataset_flag = True
    if(make_dataset_flag== True):
        load_dataset_flag = False  # If we are making a new dataset, we don't need to load it
    if(load_dataset_flag == True):
        make_dataset_flag = False  # If we are loading a dataset, we don't need to make it again

    try:
        if(download_data_flag == True):
            log.info("--- STAGE 0: Data Downloading ---")
            start_download()  # Download data from the source
            log.info("--- STAGE 0: Data Downloading Finished ---")
        
        if(interim_data_preprocess_data_flag == True):
            log.info("--- STAGE 1:Interim Data Processing ---")
            start_preprocess_data()  # Preprocess the data
            log.info("--- STAGE 1: Data Processing Finished ---")

        if(make_dataset_flag == True):        
            log.info("--- STAGE 2: Fresh Dataset Creation ---")
        
            log.info(f"Creating dataset with tag: {tag}")
            train_loader, val_loader, test_loader = start_make_dataset(tag=tag, 
                                        load_processed_dataset=False)  # Create datasets from the interim data
            log.info("--- STAGE 2: Dataset Creation Finished ---")

        if(load_dataset_flag == True):
            log.info("--- Loading processed pytorch object datasets ---")
            train_loader, val_loader, test_loader = start_make_dataset(tag=tag, 
                            load_processed_dataset=True)  # Load processed datasets
            log.info("--- Loaded processed pytorch object datasets. Dataloaders created successfully ---")

        # In a real project, you would have a feature engineering step here too
        # log.info("--- STAGE 2: Feature Engineering ---")
        # build_features.run()
        # log.info("--- STAGE 2: Feature Engineering Finished ---")

        # log.info("--- STAGE 3: Model Training ---")
        # num_epochs = 20  # Number of epochs for training
        # learning_rate = project_config.LEARNING_RATE  # Learning rate for the optimizer
        # tag = "dem_input"  
        # # tag = "dem_s1_vh_band_input"  
        # # tag = "dem_s1_water_band_input"  

        # resume_train = False  # Set to True if you want to resume training from a checkpoint
        # last_checkpoint_path = None  # Path to the checkpoint file if resuming training
        # model = None
        # # Initialize the model based on the tag
        # if tag == "dem_input":
        #     model = BinaryFloodClassifier_dem_input()
        # elif tag == "dem_s1_vh_band_input" or tag == "dem_s1_vh_or_water_channels":
        #     model = BinaryFloodClassifier_dem_s1_vh_or_water_channels()

        # if(last_checkpoint_path is not None):
        #     tag_checkpoint_dir = project_config.CHECKPOINT_DIR / tag
        #     last_checkpoint_path = tag_checkpoint_dir / "last_model.pth" # last model checkpoint path. Contains all information about the model like epoch, optimizer state, etc.

        # model, history = start_train(model, 
        #                              train_loader=train_loader, 
        #                              val_loader=val_loader, 
        #                              num_epochs=num_epochs, 
        #                              learning_rate=learning_rate,
        #                              tag=tag, # to distinguish various models. Therefore organized with tag sub-directories
        #                              resume_train=resume_train, # Train the model from scratch (0th epoch)
        #                              checkpoint_path= last_checkpoint_path)  
        # log.info("--- STAGE 3: Model Training Finished ---")

        # log.info("--- STAGE 4: Model Evaluation ---")
        # # Here you would typically evaluate the model on the test set and generate reports
        # tag_checkpoint_dir = project_config.CHECKPOINT_DIR / tag
        # best_checkpoint_path = tag_checkpoint_dir / f"best_model.pth"

        # log.info(f"--- Model evaluation completed for checkpoint: {best_checkpoint_path} ---")


        log.info(">>>>>> Project pipeline finished successfully <<<<<<")

    except Exception as e:
        log.error("Pipeline failed with error: %s", e, exc_info=True)

if __name__ == '__main__':
    run_pipeline()