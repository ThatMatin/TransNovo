import torch
from torch.utils.data import Dataset
from pathlib import Path
import gzip
import re
import os
from modules import Parameters
from tokenizer import encode, pad_encoded_seq
from tokenizer import pad_spectrum

class MSP(Dataset):
    """
    Base class to read and process NIST's MSP (msp.gz) files,
    if their sizes are below the max_size(in MB)
    """
    def __init__(self, params: Parameters):
        super().__init__()
        self.MAX_X = 0
        self.MAX_Y = 0
        self.X = torch.tensor([])
        self.Y = torch.tensor([])

        _spectra = []
        for p in Path(params.data_path).glob("*.msp.gz"):
            if os.path.getsize(p)*1e-6 > params.max_file_size:
                continue
            print(f"*> reading file: {p} | size: {os.path.getsize(p)*1e-6:.2f}")
            _spectra += _parse_msp_gz(p)

        self.set_max_lenghts(params, _spectra)
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
                params.max_spectrum_lenght < self.MAX_X:
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


def _parse_msp_gz(file_path):
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
