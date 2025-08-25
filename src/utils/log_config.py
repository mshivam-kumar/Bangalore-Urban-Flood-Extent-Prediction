import logging
from logging.handlers import RotatingFileHandler

def setup_log(name, log_file, level=logging.INFO, console_output=False):
    # To set up a logger that writes to a file and optionally to console.
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Get or create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Clear any existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # Set up file handler
    file_handler = RotatingFileHandler(log_file, maxBytes=1_000_000, backupCount=5)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    # Only add console handler if explicitly requested
    if console_output:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter('%(name)s - %(levelname)s - %(message)s'))
        logger.addHandler(console_handler)
    
    # Prevent propagation to parent loggers (this is key!)
    logger.propagate = False
    
    return logger