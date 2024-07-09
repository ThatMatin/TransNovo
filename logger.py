import logging
import sys
from typing import Tuple

import torch
from torch.cuda import CudaError


def setup_logger(module_name: str) -> logging.Logger:
    logger = logging.getLogger(module_name)
    logger.setLevel(logging.DEBUG)
    stream_handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter("%(asctime)s [%(threadName)s-%(thread)d] <%(funcName)s>: %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger


logger = setup_logger(__name__)

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
