from pathlib import Path
import data as D
import training
from modules.parameters import Parameters
from torch.nn import CrossEntropyLoss
from interrupt import InterruptHandler
from modules import TransNovo


def main():
    p = Parameters(
            d_model=32,
            n_layers=1,
            d_ff=128,
            batch_size=256,
            max_file_size=10,
            # lr=5e-4,
            n_epochs=70,
            )

    # Data preparation
    train_dl = D.AsyncDataset(Path("datafiles/train"), p.batch_size, p.device)
    test_dl = D.AsyncDataset(Path("datafiles/test"), p.batch_size, p.device)

    # Update data stats
    manifest = D.DataManifest(Path("datafiles"))
    p.data_point_count = manifest.total_spectra()
    p.max_spectrum_length, p.max_peptide_length = manifest.maxes

    # Create model
    model = TransNovo(p)
    model.load_if_file_exists()
    model.to(model.hyper_params.device)

    # loss, optimizer, scheduler, interrupt
    optimizer, scheduler = training.init_adam(model)
    interrup_handler = InterruptHandler()
    loss_fn = CrossEntropyLoss()

    try:
        # train
        training.train_loop(model,
                         optimizer,loss_fn,train_dl,
                         test_dl,interrup_handler, scheduler)
    finally:
        train_dl.stop()
        test_dl.stop()


if __name__ == "__main__":
    main()
