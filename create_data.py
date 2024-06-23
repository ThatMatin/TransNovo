import data as D


def create_msp_tensor():
    # in MB
    max_file_size = 10
    batch_size = 100000
    data_dir = "datafiles"
    discretize = True

    data = D.MSPManager()
    data.auto_create(data_dir, batch_size, max_file_size, discretize)


if __name__ == "__main__":
    create_msp_tensor()
