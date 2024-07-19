import data as D
from data.async_dataset import cuda_collate_fn
import training
import traceback
from config import get
from pathlib import Path
from logger import set_all_loggers_level_to_error, setup_logger
from modules.parameters import Parameters
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from interrupt import InterruptHandler
from modules import TransNovo
from torch import set_float32_matmul_precision, compile

logger = setup_logger(__name__)

def main():

    set_all_loggers_level_to_error()
    set_float32_matmul_precision("high")

    p = Parameters(
            d_model=get("model.d_model"),
            n_layers=get("model.n_layers"),
            d_ff=get("model.d_ff"),
            batch_size=get("model.batch_size"),
            lr=float(get("model.lr")),
            n_epochs=get("model.n_epochs"),
            )

    # Data preparation
    train_ds = D.AsyncDataset(Path(get("data.train-path")), p.batch_size, p.device, get("dataloader.queue_size"))
    test_ds = D.AsyncDataset(Path(get("data.test-path")), p.batch_size, p.device, get("dataloader.queue_size"))

    train_dl = DataLoader(train_ds, batch_size=p.batch_size, collate_fn=cuda_collate_fn)
    test_dl = DataLoader(test_ds, batch_size=p.batch_size, collate_fn=cuda_collate_fn)

    # Update data stats
    manifest = D.DataManifest(Path(get("data.manifest-path")))
    manifest.load_manifest()
    p.data_point_count = manifest.total_spectra()
    p.max_spectrum_length, p.max_peptide_length = manifest.maxes

    # Create model
    model = TransNovo(p)
    model.load_if_file_exists()
    model.to(model.hyper_params.device)
    if bool(get("train.compile")):
        model = compile(model, mode="max-autotune")

    logger.debug(f"Model size: {model.get_model_size_mb()}")

    # loss, optimizer, scheduler, interrupt
    optimizer, scheduler = training.init_adam(model)
    interrup_handler = InterruptHandler(train_ds.stop, test_ds.stop)
    loss_fn = CrossEntropyLoss()

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


if __name__ == "__main__":
    main()
