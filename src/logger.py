import logging
import os
from datetime import datetime

LOG_DIR = "logs"
LOG_FILE = f"{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.log"
LOG_PATH = os.path.join(os.getcwd(), LOG_DIR, LOG_FILE)

os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)

# FIX: use a named logger + explicit FileHandler instead of basicConfig.
# basicConfig is silently ignored if any library (e.g. sklearn) has already
# attached a handler to the root logger before this module is imported.
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH),
        logging.StreamHandler(),          # also echo to stdout
    ],
)

logging = logging.getLogger("fake_news")
