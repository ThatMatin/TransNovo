import data as D
import training
import traceback
from config import get
from pathlib import Path
from logger import setup_logger
from modules.parameters import Parameters
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from interrupt import InterruptHandler
from modules import TransNovo

logger = setup_logger(__name__)

def main():

    p = Parameters(
            d_model=get("model.d_model"),
            n_layers=get("model.n_layers"),
            d_ff=get("model.d_ff"),
            batch_size=get("model.batch_size"),
            lr=float(get("model.lr")),
            n_epochs=get("model.n_epochs"),
            )

    # Data preparation
    train_ds = D.AsyncDataset(Path("datafiles/train/"), p.batch_size, p.device, get("dataloader.queue_size"))
    test_ds = D.AsyncDataset(Path("datafiles/test/"), p.batch_size, p.device, get("dataloader.queue_size"))

    train_dl = DataLoader(train_ds, batch_size=p.batch_size, pin_memory=True)
    test_dl = DataLoader(test_ds, batch_size=p.batch_size, pin_memory=True)

    # Update data stats
    manifest = D.DataManifest(Path("datafiles"))
    manifest.load_manifest()
    p.data_point_count = manifest.total_spectra()
    p.max_spectrum_length, p.max_peptide_length = manifest.maxes

    # Create model
    model = TransNovo(p)
    model.load_if_file_exists()
    model.to(model.hyper_params.device)
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
