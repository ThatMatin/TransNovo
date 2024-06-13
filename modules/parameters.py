from typing import Optional
import torch

def lr_func(step, d_model, warmup_steps):
    return d_model**-0.5 * min(step**-0.5, warmup_steps**1-.5)

class Parameters:
    """
    ! d_model // n_heads
    lr: if None, uses the original paper's lr function
    """
    def __init__(self,
                 d_model: int = 64,
                 n_heads: int = 8,
                 data_path: str = "./datafiles",
                 n_epochs: int = 100,
                 warmup_steps: int = 10,
                 batch_size: int = 32,
                 d_ff: int = 512,
                 dropout_rate: float = 0.2,
                 max_file_size: float = 10,
                 lr=None,
                 device=None):

        self.n_epochs = n_epochs
        self.n_epochs_sofar = 0
        self.batch_size = batch_size
        self.train_test_split = 0.2

        # model definition
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.d_key = d_model // n_heads
        self.d_val = d_model // n_heads
        self.dropout_rate = dropout_rate

        # train data
        self.max_spectrum_lenght = 200
        self.max_peptide_lenght = 100

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
        self.model_save_path = f"tn_d{d_model}_h{n_heads}_ff{d_ff}_dr{10*dropout_rate}_X{self.max_spectrum_lenght}_Y{self.max_peptide_lenght}.pth"

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        # NOTE: Beacuse of pickle's issues with local functions we can't create closures with set d_model and warmup_steps. So we have to pass them to learning_rate func every time
        if lr is None:
            lr = lr_func
        self.learning_rate = lr


    def set_data_lenght_params(self, max_spectrum_lenght, max_peptide_lenght):
        self.max_peptide_lenght = max_peptide_lenght
        self.max_spectrum_lenght = max_spectrum_lenght


    def __call__(self, state_dict: Optional[dict] = None):
        """
        Accepts a dictionary of fields to set or update
        If empty, returns object's fields
        """
        if state_dict is not None:
            for k,v in state_dict.items():
                setattr(self, k, v)

        return self.__dict__
