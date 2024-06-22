import data as D
from modules.parameters import Parameters
import training
from torch.nn import CrossEntropyLoss
from data.split import get_train_test_dataloaders
from interrupt import InterruptHandler
from modules import TransNovo


def main():
    p = Parameters(d_model=512, d_ff=2024, batch_size=128, max_file_size=400, lr=1e-5)
    data = D.MSP(p)
    model = TransNovo(p)
    model.load_if_file_exists()
    model.hyper_params.learning_rate = 1e-2

    p = model.hyper_params
    model.to(p.device)
    optimizer = training.init_adam(model)
    interrup_handler = InterruptHandler()
    loss_fn = CrossEntropyLoss()
    train_dl, test_dl = get_train_test_dataloaders(data,
                                                   p.batch_size,
                                                   p.train_test_split,
                                                   True)

    training.train_loop(model, 
                     optimizer,loss_fn,train_dl,
                     test_dl,interrup_handler)


if __name__ == "__main__":
    main()
