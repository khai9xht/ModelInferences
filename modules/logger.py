import logging
import os
from logging.handlers import TimedRotatingFileHandler

LOG_FORMAT = "[%(asctime)s %(levelname)s] %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
LOG_LEVEL = logging.INFO
DEBUG = True


def setup_file_logger(log_fn):
    log_dir = "logs"
    log_fp = os.path.join(log_dir, log_fn)
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        handlers=[TimedRotatingFileHandler(log_fp, when="midnight")],
        format=LOG_FORMAT,
        level=LOG_LEVEL,
        datefmt=DATE_FORMAT,
    )


def setup_stdout_logger():
    logging.basicConfig(format=LOG_FORMAT, level=LOG_LEVEL, datefmt=DATE_FORMAT)


class Logger:
    def __init__(self, name: str, config: dict = None) -> None:
        self.name = name

        if config is not None:
            log_to_file = config["log_to_file"]
            if log_to_file:
                log_fn = config["filename"]
                setup_file_logger(log_fn)
            else:
                setup_stdout_logger()
        else:
            setup_stdout_logger()

    def log_debug(self, message):
        if DEBUG:
            logging.info(f"[{self.name}] {message}")

    def log_info(self, message):
        logging.info(f"[{self.name}] {message}")

    def log_error(self, message):
        logging.error(f"[{self.name}] {message}")
