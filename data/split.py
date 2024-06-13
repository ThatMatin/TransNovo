from typing import Tuple
from torch.utils.data import Dataset, DataLoader, random_split


def get_train_test_dataloaders(dataset: Dataset, batch_size, split=0.2, shuffle=True) -> Tuple[DataLoader, DataLoader]:
    """
    returns: (train_dl: DataLoader, test_dl: DataLoader)
    """
    train_size = int(0.8*len(dataset))
    test_size = len(dataset) - train_size
    train_ds, test_ds = random_split(dataset, [train_size, test_size])
    train_dl = DataLoader(train_ds, batch_size, shuffle)
    test_dl = DataLoader(test_ds, batch_size, shuffle)
    return train_dl, test_dl


