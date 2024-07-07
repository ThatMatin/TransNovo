import queue
import torch
import threading
from pathlib import Path
from typing import Tuple, Generator
from data.splitter import DataManifest, FileManager
from data.tensor import TensorBatch
from logger import setup_logger

T = torch.Tensor
logger = setup_logger(__name__)

class AsyncDataset:
    def __init__(self, path: Path, train_batch_size: int, queue_size: int=500):
        self.threads = list()
        self.files = FileManager()
        self.path = path
        self.train_batch_size = train_batch_size
        # TODO: dynamic?
        self.queue = queue.Queue(queue_size)
        self.__stop_event = threading.Event()
        self.thread = threading.Thread(target=self.load_data)

        self.inspect_files(path)
        self.thread.start()

    def inspect_files(self, path: Path):
        for p in path.glob("*.msp.tensor"):
            self.files.add(p)
        assert len(self.files) > 0
        logger.debug(f"found {len(self.files)} msp.tensor files.")

    def load_data(self):
        try:
            for _, file in self.files():
                tensors = self.load_file(file)
                steps = int(tensors.get_batch_size()/self.train_batch_size)
                for i in range(steps):
                    start = i * self.train_batch_size
                    end = (i + 1) * self.train_batch_size
                    if self.__stop_event.is_set():
                        return
                    self.queue.put(tensors[start:end])

                remaining = tensors.get_batch_size() % self.train_batch_size
                self.queue.put(tensors[-remaining:])

        except Exception as e:
            logger.error(e)

        finally:
            self.queue.put(None)


    def stop(self):
        self.__stop_event.set()
        self.queue.put(None)
        self.thread.join()

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

        self.stop()

    def __len__(self) -> int:
        total_batches = 0
        for _, file in self.files():
            tensors = self.load_file(file)
            steps = int(tensors.get_batch_size()/self.train_batch_size)
            if tensors.get_batch_size() % self.train_batch_size == 0:
                last_batch = 0
            else:
                last_batch = 1
            total_batches += steps + last_batch
 
        return total_batches

