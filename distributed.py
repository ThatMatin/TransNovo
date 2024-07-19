from pathlib import Path
import traceback
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import os

from torch.utils.data import DataLoader

import data as D
from config import get
from interrupt import InterruptHandler
from logger import setup_logger
from modules.parameters import Parameters
from modules.transformer import TransNovo
import training

logger = setup_logger(__name__)
def setup(rank, world_size):

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '21119'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

def main(rank, world_size):
    setup(rank, world_size)
    p = Parameters(
        d_model=get("model.d_model"),
        n_layers=get("model.n_layers"),
        d_ff=get("model.d_ff"),
        batch_size=get("model.batch_size"),
        lr=float(get("model.lr")),
        n_epochs=get("model.n_epochs"),
        )


    model = TransNovo(p).to(rank)
    model = DDP(model, device_ids=[rank])
       # Data preparation
    train_ds = D.AsyncDataset(Path(get("data.train-path")), p.batch_size, p.device, get("dataloader.queue_size"))
    test_ds = D.AsyncDataset(Path(get("data.test-path")), p.batch_size, p.device, get("dataloader.queue_size"))

    train_dl = DataLoader(train_ds, batch_size=p.batch_size, pin_memory=True)
    test_dl = DataLoader(test_ds, batch_size=p.batch_size, pin_memory=True)

    # Update data stats
    manifest = D.DataManifest(Path(get("data.manifest-path")))
    manifest.load_manifest()
    p.data_point_count = manifest.total_spectra()
    p.max_spectrum_length, p.max_peptide_length = manifest.maxes

    # Create model
    model = TransNovo(p)
    model.load_if_file_exists()
    model.to(model.hyper_params.device)
    logger.debug("compile")
    if bool(get("train.compile")):
        model = torch.compile(model, mode="max-autotune")

    logger.debug(f"Model size: {model.get_model_size_mb()}")

    # loss, optimizer, scheduler, interrupt
    optimizer, scheduler = training.init_adam(model)
    interrup_handler = InterruptHandler(train_ds.stop, test_ds.stop)
    loss_fn = nn.CrossEntropyLoss()

    try:
        # train
        training.train_loop(model,
                         optimizer,loss_fn,train_dl,
                         test_dl,interrup_handler, scheduler)
    except Exception as e:
        logger.error(f"{traceback.format_exc()}\n{e}")

    finally:
        train_ds.stop()
        test_ds.stop()

    cleanup()

if __name__ == "__main__":
    world_size = 2
    dist.spawn(main, args=(world_size,), nprocs=world_size, join=True)