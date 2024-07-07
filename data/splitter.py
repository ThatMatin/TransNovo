import os, io, re, tarfile, pickle, random
import threading
import torch
from data.tensor import TensorBatch
import tokenizer as T
from interrupt import InterruptHandler
from logger import setup_logger
from math import ceil
from pathlib import Path
from typing import Generator, List, Optional, Tuple
from torch.types import Number
from torch.utils.data import Dataset
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor
from tqdm.auto import tqdm

logger = setup_logger(__name__)

class DataManifest():
    # TODO: Add functionality to update upon discovering new data files

    def __init__(self, dir: Path):
        self.data_path = dir
        self.manifest_file_name = "msp.manifest"
        self.data_files = FileManager()
        self.positions = {} # id: []
        self.maxes = (0, 0) # (max_x, max_y)


    def auto_manifest(self, max_file_size_mb: float=0.0):
        self.search_files(max_file_size_mb)
        if self.manifest_file_exists():
            self.load_manifest()
        else:
            self.inspect_maxes()
            self.inspect_locations()
            self.save_manifest()


    def search_files(self, max_file_size_mb: float=0.0):
        files = []
        assert isinstance(self.data_path, Path)
        for p in self.data_path.glob("*.msp.tar.gz"):
            if max_file_size_mb > 0 and os.path.getsize(p)*1/(1024*1024) > max_file_size_mb:
                continue
            logger.debug(f"Manifest> size: {os.path.getsize(p)*1/(1024*1024):4.2f} | {p}")
            files.append(p)

        self.data_files.add(files)


    def inspect_maxes(self):
        assert len(self.data_files) > 0, "No files in the manifest"
        max_x = 0
        max_y = 0
        for i, f in self.data_files():
            logger.debug(f"Manifest> inspecting max sizes: {i}- {f}")
            file_max_x, file_max_y = self.get_max_xy_per_file(f)
            logger.debug(f"Manifest> max_x: {file_max_x}, max_y: {file_max_y} | {f}")
            max_x = max(max_x, file_max_x)
            max_y = max(max_y, file_max_y)

        self.maxes = (max_x, max_y)

    
    def inspect_locations(self):
        for i, f in self.data_files():
            self.get_spectra_locations_per_file(i, f)


    def tensor_files_count(self, per_file_size_mb: float=1024) -> int:
        total_data_points = self.total_spectra()
        per_data_size = self.data_point_size_bytes()
        per_file_size_bytes = per_file_size_mb * 1024 * 1024
        num_files = ceil((total_data_points * per_data_size) / per_file_size_bytes)
        return max(1, num_files)


    def data_point_per_file(self, file_size_mb: float=1024):
        return int((file_size_mb * 1024 * 1024) / self.data_point_size_bytes())


    def total_spectra(self) -> int:
        t = 0
        for _, positions in self.positions.items():
            t += len(positions)

        return t


    def get_max_xy_per_file(self, file_path: os.PathLike):
        max_peptide_length = 0
        max_peak_count = 0
        current_peptide_length = 0
        current_peak_count = 0

        with self.msp_tar_gz_file_ctx(file_path) as file:
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


    def get_spectra_locations_per_file(self, id: int, file: os.PathLike):
        locs = []
        logger.debug(f"Manifest> inspecting for locations: {id}- {file}")
        with self.msp_tar_gz_file_ctx(file) as f:
            f.seek(0, os.SEEK_END)
            file_size = f.tell()
            f.seek(0)
            progress = tqdm(total=file_size, desc=f"loc inspect - {id}")
            while True:
                pos = f.tell()
                progress.update(pos)
                line = f.readline()
                if not line:
                    break

                line = line.strip()
                if line.startswith("Name:"):
                    locs.append(pos)

            progress.close()

        self.positions[id] = locs


    def positions_iterator(self, file_id:int):
        for pos in self.positions[file_id]:
            yield pos


    def batch_indices_generator(self, batch_size: int) -> Generator[List[int], None, None]:
        """
        returns a generator that iterates over the range of total data
        and each time returns a list of randomly selected ids from that
        range without replacement, until all ids are returned
        """
        total = self.total_spectra()
        ids = [i for i in range(total)]
        random.shuffle(ids)
        batch = min(batch_size, len(ids)) 
        while len(ids) != 0:
            yield ids[-batch:]
            del ids[-batch:]
            batch = min(batch_size, len(ids)) 


    def file_positions_generator(self, ids: List[int]):
        """
        for each data file (starting from zero) given the indices (which
        are the ids of data in the total range of data point) in ids,
        returns a list of tuples: 
            [ (pos_in_file, index_of_pos_in_ids),... ,(...)]
        """
        original_ids = ids.copy()
        ids.sort()
        start = 0
        for i in range(len(self.data_files)):
            n_data_in_file = len(self.positions[i])
            positions = self.positions[i]
            p_i = [(positions[i - start], original_ids.index(i)) for i in ids if start <= i < start + n_data_in_file]

            start += n_data_in_file
            yield p_i

    
    @contextmanager
    def msp_tar_gz_file_ctx(self, file: os.PathLike) -> Generator[io.TextIOWrapper, None, None]:
        with tarfile.open(file, "r:gz") as tar:
            for member in tar.getmembers():
                if member.isfile():
                    f = tar.extractfile(member)
                    if f is not None:
                        handle = io.TextIOWrapper(f, encoding='utf-8')
                        yield handle


    def data_point_size_bytes(self) -> int:
        if self.maxes == (0, 0):
            self.inspect_maxes()

        x_tensor_size = self.maxes[0] * 2 * 8 # 8 bytes for float64
        y_tensor_size = self.maxes[1] * 8
        total = x_tensor_size + y_tensor_size + 2 * 8 # (2 * 8) charge , parent mz * int64
        return total

    def set_non_defualt_manifest_file_name(self, name:str):
        self.manifest_file_name = name

    def save_manifest(self, path: Optional[os.PathLike]=None):
        if path is None:
            path = self.get_save_path()
        with open(path, "wb") as f:
            pickle.dump(self.__dict__, f)
        logger.debug(f"manifest saved to {path}")

    def load_manifest(self, path: Optional[os.PathLike]=None):
        if path is None:
            path = self.get_save_path()
        with open(path, "rb") as f:
            self.__dict__ = pickle.load(f)
        logger.debug(f"manifest loaded from {path}")

    def manifest_file_exists(self):
        return os.path.exists(self.get_save_path())

    def get_save_path(self) -> Path:
        assert isinstance(self.data_path, Path)
        return self.data_path / Path(self.manifest_file_name)


class FileManager:
    def __init__(self):
        self.files = set([])
        self.sizes = {}

    def add(self, file: os.PathLike | List[os.PathLike]):
        if isinstance(file, os.PathLike):
            self.sizes[len(self.files)] = os.path.getsize(file)*1/(1024*1024)
            self.files.add( (len(self.files), file) )

        if isinstance(file, list):
            for f in file:
                self.sizes[len(self.files)] = os.path.getsize(f)*1/(1024*1024)
                self.files.add( (len(self.files), f) )

    def get(self, query: int | str) -> Tuple[int, os.PathLike]:
        if isinstance(query, int):
            for (i, file) in self.files:
                if query == i:
                    return (i, file)

        if isinstance(query, str):
             for (i, file) in self.files:
                assert isinstance(file, os.PathLike)
                if query == str(file):
                    return (i, file)

        raise KeyError("not found")


    def get_size(self, id: int):
        return self.sizes[id]

    def __call__(self):
        for f in self.files:
            yield f

    def __len__(self):
        return len(self.files)


class MSPSplitDataset(Dataset):
    def __init__(self, data_path: Path, interrupt: InterruptHandler) -> None:
        super().__init__()
        self.manifest = DataManifest(data_path)
        self.tensor_files = FileManager()
        self.base_file_name = "msp.tensor"
        self.lock = threading.Lock()
        self.interrupt = interrupt
        self.__counter = 0


    def get_and_inc_counter(self) -> int:
        self.lock.acquire()
        if self.__progress:
            self.__progress.update(1)
            self.__progress.refresh()
        val = self.__counter
        self.__counter += 1
        self.lock.release()
        return val


    def auto(self, file_size_mb:float):
        self.manifest.auto_manifest(0)
        self.create(file_size_mb)


    def create(self, file_size_mb: float):
        """
        Creates different files of size (file_size_mb)
        """
        N = self.manifest.data_point_per_file(file_size_mb)
        self.__progress = tqdm(total=self.manifest.tensor_files_count(file_size_mb))

        threads = []
        for _, file_path in self.manifest.data_files():
            th = threading.Thread(target=self.create_file, args=(N, file_path))
            th.start()
            threads.append(th)

        for t in threads:
            t.join()
            
        return self.__counter

    
    def create_file(self, batch_size:int, file_path: Path):
        with self.manifest.msp_tar_gz_file_ctx(file_path) as f:
            self.parse_msp_file(f, batch_size)


    def parse_msp_file(self, handle: io.TextIOWrapper, batch_size: int):
        pos = 0
        while True:
            tensor_batch, pos = self.parse_batch_to_tensor(handle, pos, batch_size)
            if self.interrupted():
                break
            tensor_batch.save_to_file(self.get_base_tensor_file_path(self.get_and_inc_counter()))
            if pos == 0:
                break
        

    def parse_batch_to_tensor(self, file_handle: io.TextIOWrapper,
                    pos:int, batch_size: int) -> Tuple[TensorBatch, int]:
        parent_re = re.compile(r'Parent=(\d+\.\d+)')
        name_pattern = re.compile(r'Name:\s+([A-Z]+)\/(\d+)_\d+')
        mod_pattern = re.compile(r'\((\d+),([A-Z]),([^)]+)\)')
        max_x, max_y = self.manifest.maxes
        tensor = TensorBatch(batch_size, self.manifest.maxes)

        x_tensor = torch.zeros(max_x, 2, dtype=torch.float64)
        y_tensor = torch.empty(1)
        charge = -1
        parent_mz = 0
        pep_seq = ""

        file_handle.seek(pos)
        batch_counter = 0
        peak_counter = 0

        while True:
            if self.interrupted():
                return tensor, pos

            pos = file_handle.tell()
            line = file_handle.readline()

            # EOF
            if line == "":
                tensor.trim(batch_counter)
                return tensor, 0

            line = line.strip()
            if line.startswith("Name:"):

                # update from previous iteration
                if batch_counter != 0:
                    tensor.update(batch_counter - 1, x_tensor, y_tensor, charge, parent_mz)

                # break the loop
                if batch_counter == batch_size:
                    assert charge >= 0
                    assert parent_mz > 0
                    assert x_tensor[:, 0].count_nonzero().item() == peak_counter
                    assert y_tensor.count_nonzero().item() == len(pep_seq) + 2
                    return tensor, pos 

                batch_counter += 1
                peak_counter = 0
                x_tensor = torch.zeros(max_x, 2, dtype=torch.float64)
                y_tensor = torch.empty(1)

                pep_seq, charge, modifications = self.extract_info_from_name(line, name_pattern, mod_pattern)
                assert isinstance(charge, Number)
                pep_seq = T.modifications_to_lower(pep_seq, modifications)
                assert pep_seq is not None
                pep_seq_enc_pad = T.pad_encoded_seq(T.encode(pep_seq), max_y)
                y_tensor = torch.tensor(pep_seq_enc_pad)

            elif line.startswith("Comment:"):
                match = parent_re.search(line)
                assert match is not None
                parent_mz = float(match.group(1))

            elif re.match(r'^\d', line):
                mz, intensity = map(float, line.split()[:2])
                x_tensor[peak_counter] = torch.tensor([mz, intensity])
                peak_counter += 1


    def extract_info_from_name(self, string: str, name_pattern: re.Pattern, mod_pattern: re.Pattern):

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

    def get_base_tensor_file_path(self, counter:int) -> Path:
        x, y = self.manifest.maxes
        return self.manifest.data_path / f"{counter}.X{x}Y{y}.{self.base_file_name}"

    def set_base_tensor_file_name(self, name: str):
        self.base_file_name = name

    def interrupted(self) -> bool:
        return self.interrupt.is_interrupted()
