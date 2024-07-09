from torch.utils.data import DataLoader
import data as D
import training
import traceback
from pathlib import Path
from logger import setup_logger
from modules.parameters import Parameters
from torch.nn import CrossEntropyLoss
from interrupt import InterruptHandler
from modules import TransNovo

logger = setup_logger(__name__)

def main():
    p = Parameters(
            d_model=128,
            n_layers=2,
            d_ff=512,
            batch_size=128,
            lr=5e-4,
            n_epochs=70,
            )

    # Data preparation
    train_ds = D.AsyncDataset(Path("datafiles/200M/train"), p.batch_size, p.device, 10)
    test_ds = D.AsyncDataset(Path("datafiles/200M/test"), p.batch_size, p.device, 10)

    # NOTE: Used to remove the batch dimension added by dataloader
    def custom_collate(batch):
        X, Y, Ch, P = batch[0]
        return X, Y, Ch, P
    train_dl = DataLoader(train_ds, collate_fn=custom_collate)
    test_dl = DataLoader(test_ds, collate_fn=custom_collate)

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
