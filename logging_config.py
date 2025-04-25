# logging_config.py
import logging
import os

# Configuration
LOGGING_ENABLED = True
LOG_FILE = "logs/application.log"

# Ensure logs directory exists
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

# Set up logging
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO if LOGGING_ENABLED else logging.CRITICAL,  # Disable logs if LOGGING_ENABLED is False
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Create a logger instance
logger = logging.getLogger(__name__)  # Use module-level logger
