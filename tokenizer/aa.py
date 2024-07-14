import csv
import torch
import tokenizer.molecules as M
from typing import Dict, List, Sequence, Tuple


PAD_TOKEN = 0
PAD_TOKEN_STR = " "
START_TOKEN = 1
START_TOKEN_STR = ">"
END_TOKEN = 2
END_TOKEN_STR = "<"
Amino_Acids_File = "tokenizer/amino_acids.csv"
AAtoI = None
ItoAA = None
AAtoMass = None
ItoMass = None
Mass_Lookup_Tensor = torch.Tensor([])

def aa_tables() -> Tuple[Dict[str, int], Dict[int, str]]:
    """returns a tuple of (AminoAcidToInt[Dict], IntToAminoAcid[Dict])"""
    global AAtoI
    global ItoAA
    if AAtoI is None or ItoAA is None:
        AA_table = _read_amino_acids_table(Amino_Acids_File)
        AAtoI = { a:i+3 for i,a in enumerate(AA_table) }
        AAtoI[PAD_TOKEN_STR] = 0
        AAtoI[START_TOKEN_STR] = 1
        AAtoI[END_TOKEN_STR] = 2
        AAtoI = {k:v for k,v in sorted(AAtoI.items(), key= lambda item: item[1])}
        ItoAA = {i:a for a,i in AAtoI.items()}
    return AAtoI, ItoAA


def mass_tables() -> Tuple[Dict[str, float], Dict[int, float]]:
    global AAtoMass
    global ItoMass
    global ItoAA

    if ItoAA is None:
        _, ItoAA = aa_tables()

    if AAtoMass is None or ItoMass is None:
        AA_table = _read_amino_acids_table(Amino_Acids_File)
        AAtoMass = {v[0]:float(v[1]) for v in [list(k.values()) for k in AA_table.values()]}
        AAtoMass[START_TOKEN_STR] = M.PROTON
        AAtoMass[END_TOKEN_STR] = M.OH
        AAtoMass[PAD_TOKEN_STR] = 0
        ItoMass = {i:AAtoMass[m] for i,m in ItoAA.items()}
    return AAtoMass, ItoMass


def mass(aa: str) -> float:
    AAtoMass, _ = mass_tables()
    return sum([AAtoMass[a] for a in aa])


def mass_tensor(Y: torch.Tensor, device: torch.device = torch.device("cuda")) -> torch.Tensor:
    """
    returns the mass of amino acids in an integer tensor whose entries
    are index of amino acids in the table
    """
    global Mass_Lookup_Tensor
    global ItoMass
    if ItoMass is None:
        _, ItoMass = mass_tables()

    if len(Mass_Lookup_Tensor) == 0:
        max_key = max(ItoMass.keys()) 
        Mass_Lookup_Tensor = torch.zeros(max_key + 1,
                                         requires_grad=False,
                                         device=device)
        for k,v in ItoMass.items():
            Mass_Lookup_Tensor[k] = v

    if Mass_Lookup_Tensor.device != device:
        Mass_Lookup_Tensor = Mass_Lookup_Tensor.to(device)

    return Mass_Lookup_Tensor[Y]


def encode(AA: str):
    AAtoI, _ = aa_tables()
    return [AAtoI[a] for a in AA]


def decode(I: List[int]):
    _, ItoAA = aa_tables()
    return ''.join([ItoAA[i] for i in I])
 

def pad_encoded_seq(aa_seq: List[int], max_len: int) -> List[int]:
    s = [START_TOKEN] + aa_seq + [END_TOKEN]
    y_offset = max_len - len(s)
    if y_offset:
        s += y_offset*[PAD_TOKEN]
    return s

def get_vocab_size():
    AAtoI, _ = aa_tables()
    return len(AAtoI)

def _read_amino_acids_table(file_path) -> Dict:
    data_dict = {}
    with open(file_path, mode='r', encoding='utf-8-sig') as csv_file:
        csv_reader = csv.DictReader(csv_file, delimiter='\t')
        assert isinstance(csv_reader.fieldnames, Sequence)
        for row in csv_reader:
            key = row[csv_reader.fieldnames[0]]
            data_dict[key] = row
    return data_dict

def modifications_to_lower(peptide_sequence, modifications):
    peptide_list = list(peptide_sequence)

    for pos, aa, mod in modifications:
        if mod == 'CAM' and aa == 'C':
            peptide_list[pos] = 'c'
        elif mod == 'Oxidation' and aa == 'M':
            peptide_list[pos] = 'm'

    return ''.join(peptide_list)

def lower_to_modifications(peptide_sequence):
    modifications = []

    peptide_list = list(peptide_sequence)

    for i, aa in enumerate(peptide_list):
        if aa == 'c':
            modifications.append((i, 'C', 'CAM'))
            peptide_list[i] = 'C'
        elif aa == 'm':
            modifications.append((i, 'M', 'Oxidation'))
            peptide_list[i] = 'M'

    return ''.join(peptide_list), modifications
