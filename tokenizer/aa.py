import csv
from typing import Dict, List, Tuple

PAD_TOKEN = 0
START_TOKEN = 1
END_TOKEN = 2
Amino_Acids_File = "tokenizer/amino_acids.csv"
AAtoI = None
ItoAA = None
AAtoMass = None

def aa_tables() -> Tuple[Dict[str, int], Dict[int, str]]:
    """returns a tuple of (AminoAcidToInt[Dict], IntToAminoAcid[Dict])"""
    global AAtoI
    global ItoAA
    if AAtoI is None or ItoAA is None:
        AA_table = _read_amino_acids_table(Amino_Acids_File)
        AAtoI = { a:i+3 for i,a in enumerate(AA_table) }
        AAtoI["<P>"] = 0
        AAtoI["<S>"] = 1
        AAtoI["<E>"] = 2
        AAtoI = {k:v for k,v in sorted(AAtoI.items(), key= lambda item: item[1])}
        ItoAA = {i:a for a,i in AAtoI.items()}
    return AAtoI, ItoAA


def mass_table() -> Dict[str, float]:
    global AAtoMass
    if AAtoMass is None:
        AA_table = _read_amino_acids_table(Amino_Acids_File)
        AAtoMass = {v[0]:float(v[1]) for v in [list(k.values()) for k in AA_table.values()]}
    return AAtoMass


def mass(aa: str) -> float:
    AAtoMass = mass_table()
    return sum([AAtoMass[a] for a in aa])


def encode(AA: str):
    AAtoI, _ = aa_tables()
    return [AAtoI[a] for a in AA]


def decode(I: List[int]):
    _, ItoAA = aa_tables()
    return ''.join([ItoAA[i] for i in I])
 

def pad_encoded_seq(aa_seq: List[int], max_len: int):
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
        for row in csv_reader:
            key = row[csv_reader.fieldnames[0]]
            data_dict[key] = row
    return data_dict
