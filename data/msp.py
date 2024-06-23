from typing import Dict, List, TextIO, Tuple
import torch
from torch.utils.data import Dataset
from pathlib import Path
import gzip
import re
import os

from tqdm.auto import tqdm

from modules import Parameters
from tokenizer import encode, pad_encoded_seq
from tokenizer import pad_spectrum

class MSPManager(Dataset):
    """
    Base class to read and process NIST's MSP (msp.gz) files,
    if their sizes are below the max_size(in MB)
    """
    def __init__(self):
        super().__init__()
        self.X = None
        self.Y = None
        self.consumed_files = []


    def auto_create(self, files_path: str="datafiles",batch_size: int=50000, size_limit: float=10.):
        """
        Automatically inspects files in the path, whose size is below threshold
        gets max lenght of peptide and peaks across all data.
        Reads in .msp files and creates data tensors and finally stores them.
        """
        files = self.get_files_list(files_path, size_limit)
        max_x, max_y = self.get_x_y_max_len(files)
        max_x += 2 # allocate space for <SOS>, <EOS>
        max_y += 2
        save_path = self.get_save_path(max_x, max_y)
        self.load(save_path)
        self.read_files(files, batch_size, max_x, max_y)
        self.save(Path(save_path))


    def get_x_y_max_len(self, files_list: List[os.PathLike]):
        """
        returns (max_x, max_y)
        """
        max_x = 0
        max_y = 0
        for f in files_list:
            print(f"inspecting max sizes: {f}")
            file_max_x, file_max_y = self.__get_x_y_max_len_for_file(f)
            max_x = max(max_x, file_max_x)
            max_y = max(max_y, file_max_y)
        return max_x, max_y


    def read_files(self, files: List[os.PathLike], batch_size: int,
                   x_len: int, y_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        returns (X, Y)
        """
        for f in tqdm(files):
            self.read_new_file(f, batch_size, x_len, y_len)
            self.consumed_files.append(f)

        assert isinstance(self.X, torch.Tensor)
        assert isinstance(self.Y, torch.Tensor)
        return self.X, self.Y


    def read_new_file(self, file: os.PathLike, batch_size: int,
                      x_len: int, y_len: int):

        if file in self.consumed_files:
            print(f"file {file} has been already consumed.")
            return

        if self.X is None or self.Y is None:
            self.X = torch.empty((0, x_len, 2))
            self.Y = torch.empty((0, y_len))

        max_x, max_y = self.get_x_y_max_len([file])
        if max_x > x_len or max_y > y_len:
            raise IndexError("width of data is larger than max")

        with gzip.open(file,"rt") as handle:
            print(f"parsing {file} to tensor...")
            pos = 0
            while True:
                x_tensor, y_tensor, pos, count = self.get_batch_n_encode(handle, pos, batch_size, x_len, y_len)
                self.X = torch.cat((self.X, x_tensor[:count]), dim=0)
                self.Y = torch.cat((self.Y, y_tensor[:count]), dim=0)
                if pos == 0:
                    break


    def save(self, save_path:os.PathLike):
        if self.X is not None and self.Y is not None:
            checkpoint = {"X": self.X, "Y": self.Y,
                          "files": self.consumed_files}
            torch.save(checkpoint, save_path)
            print(f"Dataset saved to {save_path}.")
        else:
            print("No dataset to save.")

    def load(self, data_path: os.PathLike):
        try:
            checkpoint = torch.load(data_path)
            assert isinstance(checkpoint, Dict)
            self.X = checkpoint["X"]
            self.Y = checkpoint["Y"]
            self.consumed_files = checkpoint["files"]
            assert isinstance(self.X, torch.Tensor)
            assert isinstance(self.Y, torch.Tensor)
            print(f"Dataset loaded from {data_path}.")
        except FileNotFoundError:
            print(f"No dataset found at {data_path}. Initialized empty dataset.")


    def to(self, device:torch.device|str):
        """
        In place transfer to new device
        """
        assert isinstance(self.X, torch.Tensor)
        assert isinstance(self.Y, torch.Tensor)
        self.X = self.X.to(device)
        self.Y = self.Y.to(device)

    def device(self) -> torch.device:
        assert isinstance(self.X, torch.Tensor)
        assert isinstance(self.Y, torch.Tensor)
        assert self.X.device == self.Y.device
        return self.X.device


    def get_files_list(self, files_dir: str, max_file_size: float) -> List[os.PathLike]:
        files = []
        for p in Path(files_dir).glob("*.msp.gz"):
            if os.path.getsize(p)*1e-6 > max_file_size:
                continue
            print(f"size: {os.path.getsize(p)*1e-6:4.2f} | {p}")
            files.append(p)
        return files


    def get_save_path(self, max_x_count, max_y_length) -> os.PathLike:
        return Path(f"X{max_x_count}Y{max_y_length}.tensor")


    def get_batch_n_encode(self, file_handle:TextIO, pos:int, batch_size:int, max_x:int, max_y:int):
        x_tensor = torch.zeros((batch_size, max_x, 2))
        y_tensor = torch.zeros((batch_size, max_y), dtype=torch.int64)
        batch_counter = -1 # for proper indexing
        peak_counter = 0

        file_handle.seek(pos)
        while True:
            # loop line
            pos = file_handle.tell()
            line = file_handle.readline()
            # EOF
            if line == '':
                return x_tensor, y_tensor, 0, batch_counter + 1

            line = line.strip()
            if line.startswith("Name:"):
                if batch_counter == batch_size -1:
                    return x_tensor, y_tensor, pos, batch_counter + 1
                batch_counter += 1
                peak_counter = 0
                pep_seq = line.split()[1]  # Assuming the format "Name: SEQUENCE"
                pep_seq = re.sub(r'/\d+$', '', pep_seq)
                pep_seq_enc_pad = pad_encoded_seq(encode(pep_seq), max_y)
                y_tensor[batch_counter, :] = torch.tensor(pep_seq_enc_pad)

            elif re.match(r'^\d', line):
                mz, intensity = map(float, line.split()[:2])
                x_tensor[batch_counter, peak_counter, :] = torch.tensor([mz, intensity])
                peak_counter += 1

    def discretize(self, X: torch.Tensor) -> torch.Tensor:
        _, indices = torch.sort(X[:, :, 1], descending=True)
        sorted_indices = indices.unsqueeze(-1).expand(-1, -1, 2)
        X_sorted = X.gather(1, sorted_indices)

        # Non zero elements per spectrum (batch element)
        non_zeros_counts = X_sorted.count_nonzero(1)[:, 0]
        # start indeces of the weakest 33%
        w33_st_idxs = (2/3 * non_zeros_counts).int()

        # pluck the weakest 33% and mean them, then create a tensor from means
        means_list = []
        for b in range(X.size(0)):
            m = X_sorted[b, w33_st_idxs[b]:non_zeros_counts[b], 1].mean()
            means_list.append(m)
        w33_means = torch.stack(means_list)

        w33_means_div = w33_means.unsqueeze(-1).unsqueeze(-1).expand_as(X).clone()
        w33_means_div[:, :, 0] = 1

        X_w33_normalized = X / w33_means_div

        intensities = X_w33_normalized[:, :, 1]
        discretized = torch.zeros_like(intensities)
        discretized[intensities >= 10] = 3
        discretized[(intensities >= 2) & (intensities < 10)] = 2
        discretized[(intensities >= 0.05) & (intensities < 2)] = 1
        discretized[intensities < 0.05] = 0

        X_intens_disc = X_w33_normalized.clone()
        X_intens_disc[:, :, 1] = discretized

        return X_intens_disc


    def __len__(self):
        assert isinstance(self.X, torch.Tensor)
        return self.X.shape[0]


    def __getitem__(self, index):
        assert isinstance(self.X, torch.Tensor)
        assert isinstance(self.Y, torch.Tensor)
        if torch.is_tensor(index):
            index = index.tolist()
        return self.X[index], self.Y[index]


    def __get_x_y_max_len_for_file(self, file_path):
        max_peptide_length = 0
        max_peak_count = 0
        current_peptide_length = 0
        current_peak_count = 0

        with gzip.open(file_path, 'rt') as file:
            for line in file:
                line = line.strip()
                if line.startswith("Name:"):
                    if current_peak_count > 0 or current_peptide_length > 0:
                        max_peptide_length = max(max_peptide_length, current_peptide_length)
                        max_peak_count = max(max_peak_count, current_peak_count)
                    # Extract peptide sequence directly from the "Name:" line
                    peptide_sequence = line.split()[1]  # Assuming the format "Name: SEQUENCE"
                    peptide_sequence = re.sub(r'/\d+$', '', peptide_sequence)
                    current_peptide_length = len(peptide_sequence)
                    current_peak_count = 0  # Reset peak count for the new entry
                elif line and line[0].isdigit():
                    current_peak_count += 1

            # Check the last entry if the file doesn't end with a blank line
            if current_peak_count > 0 or current_peptide_length > 0:
                max_peptide_length = max(max_peptide_length, current_peptide_length)
                max_peak_count = max(max_peak_count, current_peak_count)

        return max_peak_count, max_peptide_length


class MSP(Dataset):
    """
    Base class to read and process NIST's MSP (msp.gz) files,
    if their sizes are below the max_size(in MB)
    """
    def __init__(self, params: Parameters):
        super().__init__()
        self.MAX_X = params.max_spectrum_length
        self.MAX_Y = params.max_peptide_lenght
        self.X = torch.tensor([])
        self.Y = torch.tensor([])

        _spectra = []
        for p in Path(params.data_path).glob("*.msp.gz"):
            if os.path.getsize(p)*1e-6 > params.max_file_size:
                continue
            print(f"*> reading file: {p} | size: {os.path.getsize(p)*1e-6:.2f}")
            _spectra += self._parse_msp_gz(p)

        # FIX: important
        # self.set_max_lenghts(params, _spectra)
        params.data_point_count = len(_spectra)

        names, spectra = [], []
        for name, spectrum in _spectra:
            names.append(pad_encoded_seq(name, self.MAX_Y))
            spectra.append(pad_spectrum(spectrum, self.MAX_X))
        self.X = torch.tensor(spectra).to(params.device)
        self.X = self.discretize(self.X)
        self.Y = torch.tensor(names).to(params.device)


    def set_max_lenghts(self, params: Parameters, spectra):
        for spectrum in spectra:
            self.MAX_X = max(self.MAX_X, len(spectrum[1]))
            self.MAX_Y = max(self.MAX_Y, len(spectrum[0]))
        self.MAX_Y += 2 # started and end tokens accounted forse
        self.MAX_X += 2
        
        if params.max_peptide_lenght < self.MAX_Y or \
                params.max_spectrum_length < self.MAX_X:
            params.set_data_lenght_params(self.MAX_X, self.MAX_Y)


    def discretize(self, X):
        _, indices = torch.sort(X[:, :, 1], descending=True)
        sorted_indices = indices.unsqueeze(-1).expand(-1, -1, 2)
        X_sorted = X.gather(1, sorted_indices)

        # Non zero elements per spectrum (batch element)
        non_zeros_counts = X_sorted.count_nonzero(1)[:, 0]
        # start indeces of the weakest 33%
        w33_st_idxs = (2/3 * non_zeros_counts).int()

        # pluck the weakest 33% and mean them, then create a tensor from means
        means_list = []
        for b in range(X.size(0)):
            m = X_sorted[b, w33_st_idxs[b]:non_zeros_counts[b], 1].mean()
            means_list.append(m)
        w33_means = torch.stack(means_list)

        w33_means_div = w33_means.unsqueeze(-1).unsqueeze(-1).expand_as(X).clone()
        w33_means_div[:, :, 0] = 1

        X_w33_normalized = X / w33_means_div

        intensities = X_w33_normalized[:, :, 1]
        discretized = torch.zeros_like(intensities)
        discretized[intensities >= 10] = 3
        discretized[(intensities >= 2) & (intensities < 10)] = 2
        discretized[(intensities >= 0.05) & (intensities < 2)] = 1
        discretized[intensities < 0.05] = 0

        X_intens_disc = X_w33_normalized.clone()
        X_intens_disc[:, :, 1] = discretized

        return X_intens_disc


    def __len__(self):
        return self.X.shape[0]


    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        return self.X[index], self.Y[index]


    def _parse_msp_gz(self, file_path):
        with gzip.open(file_path, 'rt') as file:
            content = file.read()

        entries = content.split("\n\n")
        parsed_entries = []

        for entry in entries:
            if not entry.strip():
                continue
            
            peaks = []
            name = None
            for line in entry.split('\n'):
                if line.startswith('Name:'):
                    # Remove the suffix from the name
                    name = re.split(r'/\d+', line.split('Name: ')[1].strip())[0]
                    name = encode(name)
                elif re.match(r'^\d', line):
                    mz, intensity = map(float, line.strip().split()[:2])
                    peaks.append((mz, intensity))

            if peaks and name:
                parsed_entries.append((name, peaks))
        
        return parsed_entries
