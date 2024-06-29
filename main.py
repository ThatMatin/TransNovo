import data as D
import training
from modules.parameters import Parameters
from torch.nn import CrossEntropyLoss
from data.split import get_train_test_dataloaders
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
    data = D.MSPManager()
    data.auto_create(p.data_path,
                     batch_size=100000,
                     size_limit=p.max_file_size,
                     is_discretized=False)
    data.to(p.device)

    # Update data stats
    p.data_point_count = len(data)
    p.max_spectrum_length, p.max_peptide_length = data.current_x_y_max()

    # Create model
    model = TransNovo(p)
    model.load_if_file_exists()
    model.to(model.hyper_params.device)

    # loss, optimizer, scheduler, interrupt
    optimizer, scheduler = training.init_adam(model)
    interrup_handler = InterruptHandler()
    loss_fn = CrossEntropyLoss()

    # train
    train_dl, test_dl = get_train_test_dataloaders(data,
                                                   p.batch_size,
                                                   p.train_test_split,
                                                   True)
    training.train_loop(model,
                     optimizer,loss_fn,train_dl,
                     test_dl,interrup_handler, scheduler)


if __name__ == "__main__":
    main()
