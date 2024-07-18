import logging
import sys
import torch
from typing import Tuple
from torch.cuda import CudaError

format = "%(asctime)s.%(msecs)03d[P:%(processName)s-T:%(threadName)s(%(thread)d)] <%(funcName)s>: %(message)s"
datefmt = "%Y-%m-%d %H:%M:%S"

def setup_logger(module_name: str) -> logging.Logger:
    logger = logging.getLogger(module_name)
    logger.setLevel(logging.DEBUG)
    stream_handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(format, datefmt)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger


logger = setup_logger(__name__)
profiler_logger = None

def get_gpu_memory_info() -> Tuple[float, float, float]:
    """
    returns (total_memory, free_memory, allocated_memory)
    """
    if torch.cuda.is_available():
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1024 ** 2
        reserved_memory = torch.cuda.memory_reserved(0) / 1024 ** 2
        allocated_memory = torch.cuda.memory_allocated(0) / 1024 ** 2
        free_memory = reserved_memory - allocated_memory

        return total_memory, free_memory, allocated_memory
    else:
        logger.error("CUDA is not available.")
        # TODO: check code
        raise CudaError(0)

def log_memory(msg: str=""):
    logger.debug(f"{msg}> GPU: Free:{get_gpu_memory_info()[1]:6.4f} Allocated: {get_gpu_memory_info()[2]:6.4f}")

def log_profiler(msg):
    global profiler_logger
    if profiler_logger is None:
        logger = logging.getLogger('profiler_logger')
        logger.setLevel(logging.INFO)  # Set the logging level for logger1
        file_handler = logging.FileHandler('profile.log')
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter(format, datefmt)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        profiler_logger = logger

    profiler_logger.info(msg)

def set_all_loggers_level_to_error():
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.ERROR)

    for logger_name in logging.root.manager.loggerDict:
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.ERROR)
