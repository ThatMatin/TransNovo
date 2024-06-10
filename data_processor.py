import gzip
from typing import Dict
import numpy as np
import csv
import re

#----------------------------------------------------------------------
# Amino Acids encoder decoder

def read_csv_to_dict(file_path) -> Dict:
    data_dict = {}
    with open(file_path, mode='r', encoding='utf-8-sig') as csv_file:
        csv_reader = csv.DictReader(csv_file, delimiter='\t')
        for row in csv_reader:
            key = row[csv_reader.fieldnames[0]]
            del row[csv_reader.fieldnames[0]]
            data_dict[key] = row
    return data_dict

def get_AA_Dec_Enc(aa_csv_file):
    """returns a tuple of (encoder, decoder) for amino acids"""
    AA_table = read_csv_to_dict(aa_csv_file)
    AAtoI = {a:i+1 for i,a in enumerate(AA_table) }
    AAtoI['.'] = 0  # <EOF> token
    ItoAA = {i:a for a,i in AAtoI.items()}
    encoder = lambda s: [AAtoI[a] for a in s]
    decoder = lambda l: ''.join([ItoAA[i] for i in l])
    return encoder, decoder

#----------------------------------------------------------------------
# read in MSP files

def parse_msp_gz(file_path):
    enc, _ = get_AA_Dec_Enc("./data/amino_acids.csv")

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
                name = enc(name)
            # elif line.startswith('Num peaks:'):
            #     num_peaks = int(line.split('Num Peaks: ')[1].strip())
            #     max_num_peaks = max(max_num_peaks, num_peaks)
            elif re.match(r'^\d', line):
                mz, intensity = map(float, line.strip().split()[:2])
                peaks.append((mz, intensity))

        if peaks and name:
            parsed_entries.append((name, peaks))
    
    return parsed_entries

#----------------------------------------------------------------------
# Chat GPT Tokenization model
def prepare_data_for_tokenization(entries, bin_size=0.1, levels=4):
    tokenized_data = {}
    
    for name, peaks in entries:
        tokens = assign_tokens(peaks, bin_size, levels)
        tokenized_data[name] = tokens
    
    return tokenized_data

import csv
# Assuming you have the tokenization functions defined earlier in the same script or imported
def bin_mz(peaks, bin_size=0.1):
    bins = np.arange(0, max([mz for mz, _ in peaks]) + bin_size, bin_size)
    binned_peaks = [(np.digitize(mz, bins) - 1, intensity) for mz, intensity in peaks]
    return binned_peaks

def quantize_intensity(peaks, levels):
    max_intensity = max([intensity for _, intensity in peaks])
    intensity_bins = np.linspace(0, max_intensity, levels + 1)
    quantized_peaks = [(mz_bin, np.digitize(intensity, intensity_bins) - 1) for mz_bin, intensity in peaks]
    return quantized_peaks

def assign_tokens(peaks, bin_size=0.1, levels=4):
    binned_peaks = bin_mz(peaks, bin_size)
    quantized_peaks = quantize_intensity(binned_peaks, levels)
    tokens = [(mz_bin, int_level) for mz_bin, int_level in quantized_peaks]
    token_dict = {token: idx + 1 for idx, token in enumerate(set(tokens))}
    token_sequence = [token_dict[token] for token in tokens]
    return token_sequence

# Example usage
# file_path = './test_data.msp'  # Replace with your .msp.gz file path
# entries = parse_msp_gz(file_path)
# tokenized_data = prepare_data_for_tokenization(entries, bin_size=0.1, levels=4)
#
# # Print tokenized data for verification
# for name, tokens in tokenized_data.items():
#     print(f"Name: {name}")
#     print(f"Tokens: {tokens}")
#
