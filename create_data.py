import data as D
from pathlib import Path

from interrupt import InterruptHandler


def create_msp_tensor():
    file_size_mb = 200
    data_path = Path("datafiles")

    interrupt = InterruptHandler()
    data = D.MSPSplitDataset(data_path, interrupt)
    data.auto(file_size_mb)


if __name__ == "__main__":
    create_msp_tensor()
