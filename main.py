import data as D
from modules.parameters import Parameters
import train
from torch.nn import CrossEntropyLoss
from data.split import get_train_test_dataloaders
from interrupt import InterruptHandler
from modules import TransNovo


def main():
    p = Parameters(max_file_size=500)
    data = D.MSP(p)
    model = TransNovo(p)
    model.load_if_file_exists()

    p = model.hyper_params
    model.to(p.device)
    optimizer = train.init_adam(model)
    interrup_handler = InterruptHandler()
    loss_fn = CrossEntropyLoss()
    train_dl, test_dl = get_train_test_dataloaders(data,
                                                   p.batch_size,
                                                   p.train_test_split,
                                                   True)

    train.train_loop(model, 
                     optimizer,loss_fn,train_dl,
                     test_dl,interrup_handler)


if __name__ == "__main__":
    main()
