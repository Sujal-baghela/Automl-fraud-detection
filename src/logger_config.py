import logging
import os


def setup_logging(log_level: str = "INFO", log_file: str = "app.log"):
    """
    Configure centralized logging for the project.
    """

    os.makedirs("logs", exist_ok=True)

    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[
            logging.FileHandler(os.path.join("logs", log_file)),
            logging.StreamHandler()
        ]
    )