import traceback
import torch
import multiprocessing
from multiprocessing import Queue
from pathlib import Path
from typing import Tuple, Generator
from queue import Empty, Full

from torch.utils.data import IterableDataset
from data.splitter import FileManager
from data.tensor import TensorBatch
from logger import setup_logger

T = torch.Tensor
logger = setup_logger(__name__)

class AsyncDataset(IterableDataset):
    def __init__(self, path: Path, train_batch_size: int, device="cuda", queue_size: int=10):
        super().__init__()
        self.files = FileManager()
        self.path = path
        self.train_batch_size = train_batch_size
        # TODO: dynamic queue size?
        self.queue = Queue(queue_size)
        self.stop_event = multiprocessing.Event()
        self.done_event = multiprocessing.Event()
        self.__len = None
        self.process = multiprocessing.Process(target=self.load_data)
        self.device = device

        self.inspect_files(path)

    def start_worker(self):
        self.done_event.clear()
        self.process = multiprocessing.Process(target=self.load_data)
        self.process.start()

    def inspect_files(self, path: Path):
        for p in path.glob("*.msp.tensor"):
            self.files.add(p)
        assert len(self.files) > 0
        logger.debug(f"found {len(self.files)} msp.tensor files.")

    def load_data(self):
        try:
            for _, file in self.files():
                if self.stop_event.is_set():
                    break
                tensors = self.load_file(file)
                for i in range(tensors.get_batch_size()):
                    if self.stop_event.is_set():
                        break
                    self.put(tensors[i])

        except Exception as e:
            logger.error(f"{traceback.format_exc()}\n{e}")
        finally:
            self.queue.put(None)
            self.done_event.wait()

    def put(self, tensor: Tuple[T, T, T, T]):
        while not self.stop_event.is_set():
            try:
                self.queue.put(tensor, timeout=1)
                return
            except Full:
                pass

    def clear_queue(self):
        queue = self.queue
        while not queue.empty():
            try:
                queue.get_nowait()
            except Empty:
                break

    def stop(self):
        try:
            self.stop_event.set()
            self.process.kill()
            self.process.join()

        except Exception as e:
            logger.error(f"{traceback.format_exc()}\n{e}")


    def load_file(self, path: Path) -> TensorBatch:
        tensor_batch = TensorBatch(1, (0, 0))
        tensor_batch.load_file(path)
        tensor_batch.set_requires_grad_true()
        return tensor_batch

    def __call__(self) -> Generator[Tuple[T, T, T, T], None, None]:
        self.start_worker()
        try:
            while True:
                batch = self.queue.get()
                if batch is None:
                    break
                assert all(isinstance(x, T) for x in batch)
                X, _, _, P = batch
                X.requires_grad_()
                P.requires_grad_()
                yield batch
        except Exception as e:
            logger.error(f"{traceback.format_exc()}\n{e}")
        finally:
            self.done_event.set()


    def __iter__(self):
        return self.__call__()

    def __len__(self) -> int:
        if self.__len:
            return self.__len

        total = 0
        for _, file in self.files():
            tensors = self.load_file(file)
            total += tensors.get_batch_size()
            del tensors

        self.__len = total
        return total


def cuda_collate_fn(batch):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch = torch.utils.data.dataloader.default_collate(batch)
    def to_device(obj):
        if isinstance(obj, torch.Tensor):
            return obj.to(device)
        elif isinstance(obj, dict):
            return {key: to_device(val) for key, val in obj.items()}
        elif isinstance(obj, list):
            return [to_device(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(to_device(item) for item in obj)
        else:
            return obj

    return to_device(batch)
