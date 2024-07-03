from typing import Optional
import torch

class Parameters:
    """
    ! d_model // n_heads
    lr: if None, uses the original paper's lr function
    """
    def __init__(self,
                 d_model: int = 64,
                 n_layers: int = 6,
                 n_heads: int = 8,
                 data_path: str = "./datafiles",
                 n_epochs: int = 100,
                 warmup_steps: int = 1,
                 batch_size: int = 32,
                 d_ff: int = 512,
                 dropout_rate: float = 0.1,
                 max_file_size: float = 0,
                 lr: float = 0.0,
                 device=None):

        self.n_epochs = n_epochs
        self.n_epochs_sofar = 0
        self.batch_size = batch_size
        self.train_test_split = 0.2

        # model definition
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.d_key = d_model // n_heads
        self.d_val = d_model // n_heads
        self.dropout_rate = dropout_rate

        # train data
        self.max_spectrum_length = 0
        self.max_peptide_length = 0
        self.data_point_count = 0

        # optimizer
        self.warmup_steps = warmup_steps
        self.optimizer_state_dict = {}
        self.optimizer_adam_betas = (0.9, 0.98) 
        self.optimizer_adam_eps = 1e-9

        # train stats
        self.train_result_matrix = torch.tensor([])
        self.test_result_matrix = torch.tensor([])

        # Saving and loading
        self.data_path = data_path
        self.max_file_size = max_file_size

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        self.last_learning_rate = 0
        self.new_learning_rate = lr

    def model_save_path(self):
        f = f"D{self.d_model}N{self.n_layers}H{self.n_heads}FF{self.d_ff}DO{int(10*self.dropout_rate)}"
        f += f"B{self.batch_size}X{self.max_spectrum_length}Y{self.max_peptide_length}.pth"
        return f


    def __call__(self, state_dict: Optional[dict] = None):
        """
        Accepts a dictionary of fields to set or update
        If empty, returns object's fields
        """
        if state_dict is not None:
            for k,v in state_dict.items():
                setattr(self, k, v)

        return self.__dict__
