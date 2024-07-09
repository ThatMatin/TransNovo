import queue
import traceback
import torch
import threading
from pathlib import Path
from typing import Tuple, Generator

from torch.utils.data import IterableDataset
from data.splitter import FileManager
from data.tensor import TensorBatch
from logger import log_memory, setup_logger

T = torch.Tensor
logger = setup_logger(__name__)

class AsyncDataset(IterableDataset):
    def __init__(self, path: Path, train_batch_size: int, device="cuda", queue_size: int=10):
        super().__init__()
        self.threads = list()
        self.files = FileManager()
        self.path = path
        self.train_batch_size = train_batch_size
        # TODO: dynamic queue size?
        self.queue = queue.Queue(queue_size)
        self.__stop_event = threading.Event()
        self.__len = None
        self.thread = threading.Thread(target=self.load_data)
        self.device = device

        self.inspect_files(path)
        self.thread.start()

    def inspect_files(self, path: Path):
        for p in path.glob("*.msp.tensor"):
            self.files.add(p)
        assert len(self.files) > 0
        logger.debug(f"found {len(self.files)} msp.tensor files.")

    def load_data(self):
        try:
            log_memory()
            for _, file in self.files():
                if self.__stop_event.is_set():
                    break
                tensors = self.load_file(file)
                tensors.to(self.device)
                steps = int(tensors.get_batch_size()/self.train_batch_size)
                for i in range(steps):
                    if self.__stop_event.is_set():
                        break
                    start = i * self.train_batch_size
                    end = (i + 1) * self.train_batch_size
                    self.put(tensors[start:end])
                remaining = tensors.get_batch_size() % self.train_batch_size
                self.put(tensors[-remaining:])

        except Exception as e:
            logger.error(f"{traceback.format_exc()}\n{e}")

    def put(self, tensor: Tuple[T, T, T, T]):
        while not self.__stop_event.is_set():
            try:
                self.queue.put(tensor, timeout=1)
                return
            except queue.Full:
                pass

        logger.debug("clearing queue and putting None")
        self.queue.queue.clear()
        self.queue.put(None, timeout=1)


    def stop(self):
        try:
            self.__stop_event.set()
        except Exception as e:
            logger.error(f"{traceback.format_exc()}\n{e}")


    def load_file(self, path: Path) -> TensorBatch:
        tensor_batch = TensorBatch(1, (0, 0))
        tensor_batch.load_file(path)
        return tensor_batch

    def __call__(self) -> Generator[Tuple[T, T, T, T], None, None]:
        while True:
            batch = self.queue.get()
            if batch is None:
                break
            yield batch

    def __iter__(self):
        return self.__call__()

    def __len__(self) -> int:
        if self.__len:
            return self.__len

        total_batches = 0
        for _, file in self.files():
            tensors = self.load_file(file)
            steps = int(tensors.get_batch_size()/self.train_batch_size)
            if tensors.get_batch_size() % self.train_batch_size == 0:
                last_batch = 0
            else:
                last_batch = 1
            total_batches += steps + last_batch
            
            del tensors

        self.__len = total_batches
        return total_batches
