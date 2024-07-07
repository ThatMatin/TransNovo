import logging
import sys

def setup_logger(module_name: str) -> logging.Logger:
    logger = logging.getLogger(module_name)
    logger.setLevel(logging.DEBUG)
    stream_handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter("%(asctime)s [%(threadName)s-%(thread)d] <%(funcName)s>: %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger
