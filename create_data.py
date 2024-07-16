from config import get
import data as D
from pathlib import Path
from interrupt import InterruptHandler


def create_msp_tensor():
    file_size_mb = int(get("data.size"))
    data_path = Path(get("data.manifest-path"))

    interrupt = InterruptHandler()
    data = D.MSPSplitDataset(data_path, interrupt, bool(get("data.round-power-two")))
    data.auto(file_size_mb)


if __name__ == "__main__":
    create_msp_tensor()
