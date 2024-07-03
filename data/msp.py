from contextlib import contextmanager
import io
import tarfile
import torch
import tokenizer as T
from torch.types import Number
from torch.utils.data import Dataset
from tqdm.auto import tqdm
from typing import Dict, List, TextIO, Tuple
from pathlib import Path
import gzip, re, os

class MSPManager(Dataset):
    """
    Base class to read and process NIST's MSP (msp.gz) files,
    if their sizes are below the max_size(in MB)
    X: spectra (containing tuples of (mz, intensity))
    Y: Peptide sequences
    Ch: Precursor ion charge
    P: Precursor mz
    """
    def __init__(self):
        super().__init__()
        self.X = None
        self.Y = None
        self.Ch = None
        self.P = None
        self.consumed_files = []
        self.is_discretized = False


    def auto_create(self, files_path: str="datafiles",batch_size: int=100000, size_limit: float=0, is_discretized: bool=False):
        """
        Automatically inspects files in the path, whose size is below threshold
        gets max lenght of peptide and peaks across all data.
        Reads in .msp files and creates data tensors and finally stores them.
        """
        self.is_discretized = is_discretized
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
            print(f"(MSP) inspecting max sizes: {f}")
            file_max_x, file_max_y = self.__get_x_y_max_len_for_file(f)
            print(f"(MSP) max_x: {file_max_x}, max_y: {file_max_y} | {f}")
            max_x = max(max_x, file_max_x)
            max_y = max(max_y, file_max_y)
        return max_x, max_y

    def current_x_y_max(self):
        """
        returns (max_x, max_y)
        """
        assert isinstance(self.X, torch.Tensor)
        assert isinstance(self.Y, torch.Tensor)
        return self.X.size(1), self.Y.size(1)

    def read_files(self, files: List[os.PathLike], batch_size: int,
                   x_len: int, y_len: int) -> Tuple[torch.Tensor,
                                                    torch.Tensor,
                                                    torch.Tensor,
                                                    torch.Tensor]:
        """
        returns (X, Y, Ch, P)
        """
        for f in tqdm(files):
            self.read_new_file(f, batch_size, x_len, y_len)
            self.consumed_files.append(f)

        assert isinstance(self.X, torch.Tensor)
        assert isinstance(self.Y, torch.Tensor)
        assert isinstance(self.P, torch.Tensor)
        assert isinstance(self.Ch, torch.Tensor)
        return self.X, self.Y, self.Ch, self.P


    def read_new_file(self, file: os.PathLike, batch_size: int,
                      x_len: int, y_len: int):

        if file in self.consumed_files:
            print(f"(MSP) file {file} has been already consumed.")
            return

        if self.__is_any_tensor_none():
            self.__init_empty_tesnors(x_len, y_len)
        # for linter error bypass
        assert self.X is not None
        assert self.Y is not None
        assert self.Ch is not None
        assert self.P is not None

        max_x, max_y = self.get_x_y_max_len([file])
        if max_x > x_len or max_y > y_len:
            raise IndexError("width of data is larger than max")

        with self.__msp_file_context(file) as handle:
            print(f"(MSP) parsing {file} to tensor...")
            pos = 0
            while True:
                x_tensor, y_tensor, ch_tensor, p_tensor, pos, count = self.get_batch_n_encode(handle, pos, batch_size, x_len, y_len)
                if self.is_discretized:
                    x_tensor = self.discretize(x_tensor)
                self.X = torch.cat((self.X, x_tensor[:count]), dim=0)
                self.Y = torch.cat((self.Y, y_tensor[:count]), dim=0)
                self.Ch = torch.cat((self.Ch, ch_tensor[:count]), dim=0)
                self.P = torch.cat((self.P, p_tensor[:count]), dim=0)
                if pos == 0:
                    break

    @contextmanager
    def __msp_file_context(self, file: os.PathLike):
        with tarfile.open(file, "r:gz") as tar:
            for member in tar.getmembers():
                if member.isfile():
                    f = tar.extractfile(member)
                    if f is not None:
                        handle = io.TextIOWrapper(f, encoding='utf-8')
                        yield handle


    def __init_empty_tesnors(self, x_len: int, y_len: int):
        self.X = torch.empty((0, x_len, 2))
        self.Y = torch.empty((0, y_len), dtype=torch.int64)
        self.Ch = torch.empty((0), dtype=torch.int64)
        self.P = torch.empty((0))


    def __is_any_tensor_none(self) -> bool:
        return self.X is None or self.Y is None or self.Ch is None or self.P is None


    def save(self, save_path:os.PathLike):
        if self.X is not None and self.Y is not None:
            checkpoint = {"X": self.X, "Y": self.Y,
                          "Ch": self.Ch, "P": self.P,
                          "files": self.consumed_files,
                          "is_discretized": self.is_discretized}
            torch.save(checkpoint, save_path)
            print(f"(MSP) Dataset saved to {save_path}.")
        else:
            print("(MSP) No dataset to save.")


    def load(self, data_path: os.PathLike):
        try:
            checkpoint = torch.load(data_path)
            assert isinstance(checkpoint, Dict)
            self.X = checkpoint["X"]
            self.Y = checkpoint["Y"]
            self.Ch = checkpoint["Ch"]
            self.P = checkpoint["P"]
            self.consumed_files = checkpoint["files"]
            self.is_discretized = checkpoint["is_discretized"]

            assert isinstance(self.X, torch.Tensor)
            assert isinstance(self.Y, torch.Tensor)
            assert isinstance(self.Ch, torch.Tensor)
            assert isinstance(self.P, torch.Tensor)
            print(f"(MSP) Dataset loaded from {data_path}.")

        except FileNotFoundError:
            print(f"(MSP) No dataset found at {data_path}. Initialized empty dataset.")


    def to(self, device:torch.device|str):
        """
        In place transfer to new device
        """
        assert isinstance(self.X, torch.Tensor)
        assert isinstance(self.Y, torch.Tensor)
        assert isinstance(self.Ch, torch.Tensor)
        assert isinstance(self.P, torch.Tensor)
        self.X = self.X.to(device)
        self.Y = self.Y.to(device)
        self.Ch = self.Ch.to(device)
        self.P = self.P.to(device)


    def device(self) -> torch.device:
        assert isinstance(self.X, torch.Tensor)
        assert isinstance(self.Y, torch.Tensor)
        assert isinstance(self.P, torch.Tensor)
        assert isinstance(self.Ch, torch.Tensor)
        assert self.X.device == self.Y.device == self.P.device == self.Ch.device, "tensors on different devices"
        return self.X.device


    def get_files_list(self, files_dir: str, max_file_size: float) -> List[os.PathLike]:
        files = []
        for p in Path(files_dir).glob("*.msp.tar.gz"):
            if max_file_size > 0 and os.path.getsize(p)*1e-6 > max_file_size:
                continue
            print(f"(MSP) size: {os.path.getsize(p)*1e-6:4.2f} | {p}")
            files.append(p)
        return files


    def get_save_path(self, max_x_count, max_y_length) -> os.PathLike:
        txt = f"X{max_x_count}Y{max_y_length}"
        if self.is_discretized:
            txt += "D"
        return Path(txt + ".tensor")

    def get_batch_n_encode(self, file_handle:TextIO, pos:int, batch_size:int, max_x:int, max_y:int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
        """returns (x_tensor, y_tensor, ch_tensor, p_tensor, file_pos, num_parsed_entries)"""

        x_tensor = torch.zeros((batch_size, max_x, 2))
        y_tensor = torch.zeros((batch_size, max_y), dtype=torch.int64)
        ch_tensor = torch.zeros(batch_size, dtype=torch.int64)
        p_tensor = torch.zeros(batch_size)

        batch_counter = -1 # for proper indexing
        peak_counter = 0

        # compile here for less overhead
        parent_re = re.compile(r'Parent=(\d+\.\d+)')
        name_pattern = re.compile(r'Name:\s+([A-Z]+)\/(\d+)_\d+')
        mod_pattern = re.compile(r'\((\d+),([A-Z]),([^)]+)\)')

        file_handle.seek(pos)
        while True:
            # loop line
            pos = file_handle.tell()
            line = file_handle.readline()
            # EOF
            if line == '':
                return x_tensor, y_tensor, ch_tensor, p_tensor, 0, batch_counter + 1

            line = line.strip()
            if line.startswith("Name:"):
                if batch_counter == batch_size -1:
                    return x_tensor, y_tensor, ch_tensor, p_tensor, pos, batch_counter + 1
                batch_counter += 1
                peak_counter = 0

                pep_seq, charge, modifications = self.__extract_info_from_name(line, name_pattern, mod_pattern)
                assert isinstance(charge, Number)
                ch_tensor[batch_counter] = charge
                pep_seq = T.modifications_to_lower(pep_seq, modifications)
                assert pep_seq is not None
                pep_seq_enc_pad = T.pad_encoded_seq(T.encode(pep_seq), max_y)
                y_tensor[batch_counter, :] = torch.tensor(pep_seq_enc_pad)

            elif line.startswith("Comment:"):
                match = parent_re.search(line)
                assert match is not None
                parent_mz = float(match.group(1))
                p_tensor[batch_counter] = parent_mz

            elif re.match(r'^\d', line):
                mz, intensity = map(float, line.split()[:2])
                x_tensor[batch_counter, peak_counter, :] = torch.tensor([mz, intensity])
                peak_counter += 1


    def __extract_info_from_name(self, string: str, name_pattern: re.Pattern, mod_pattern: re.Pattern):

        name_match = name_pattern.search(string)
        if name_match:
            peptide_sequence = name_match.group(1)
            charge = int(name_match.group(2))
        else:
            peptide_sequence = None
            charge = None

        # Extracting modifications
        modifications = mod_pattern.findall(string)
        modifications = [(int(pos), aa, mod) for pos, aa, mod in modifications]

        return peptide_sequence, charge, modifications


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
        assert isinstance(self.P, torch.Tensor)
        assert isinstance(self.Ch, torch.Tensor)
        if torch.is_tensor(index):
            index = index.tolist()
        return self.X[index], self.Y[index], self.Ch[index], self.P[index]


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
