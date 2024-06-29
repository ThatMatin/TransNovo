import data as D
import training
from modules.parameters import Parameters
from torch.nn import CrossEntropyLoss
from data.split import get_train_test_dataloaders
from interrupt import InterruptHandler
from modules import TransNovo


def main():
    p = Parameters(
            d_model=256,
            n_layers=3,
            d_ff=1024,
            batch_size=128,
            max_file_size=30,
            lr=1e-5,
            n_epochs=100,
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

    # Create model
    model = TransNovo(p)
    model.load_if_file_exists()
    p = model.hyper_params
    p.n_epochs = 1000
    p.learning_rate = 5e-4
    model.to(p.device)

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
