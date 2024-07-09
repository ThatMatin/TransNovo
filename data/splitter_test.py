import os
from typing import Generator
import torch
import unittest, logging
from pathlib import Path
from data.splitter import DataManifest, FileManager, MSPSplitDataset
from interrupt import InterruptHandler

class FileManagerTest(unittest.TestCase):
    def setUp(self):
        self.fm = FileManager()

    def test_add_one_file(self):
        self.fm.add(Path("."))
        self.assertEqual(self.fm.get(0), (0, Path(".")))

    def test_add_another(self):
        self.fm.add(Path("."))
        self.fm.add(Path("."))
        self.assertEqual(self.fm.get(1), (1, Path(".")))

    def test_add_list(self):
        ls = [ Path("."), Path(".")]
        self.fm.add(ls)
        self.assertIn((0, Path(".")), list(self.fm()))
        self.assertIn((1, Path(".")), list(self.fm()))

    def test_get_int(self):
        self.fm.add(Path("."))
        self.assertEqual(self.fm.get(0), (0, Path(".")))

    def test_get_str(self):
        self.fm.add(Path("."))
        self.assertEqual(self.fm.get("."), (0, Path(".")))

    def test_generator(self):
        ls = [ Path("."), Path("..")]
        ls_check = []
        self.fm.add(ls)
        for _, path in self.fm():
            ls_check.append(path)
        self.assertIn(Path("."), ls_check)
        self.assertIn(Path(".."), ls_check)

    def test_file_size(self):
        with open("test", "w") as f:
            f.seek(1024 * 1024)
            f.write(".")
        
        path = Path("test")
        self.fm.add(path)
        self.assertAlmostEqual(self.fm.get_size(0), 1, 3)
        os.remove(path)


class DataManifestTest(unittest.TestCase):
    def setUp(self):
        self.dm = DataManifest(Path("data/test_data/"))
        self.dm.set_non_defualt_manifest_file_name("msp.manifest.test")
        self.dm.search_files()

    def test_search_files(self):
        self.assertEqual(len(self.dm.data_files), 2)

    def test_maxes(self):
        self.dm.inspect_maxes()
        self.assertTupleEqual(self.dm.maxes, (275, 81))

    def test_locations_reads_all(self):
        self.dm.inspect_locations()
        self.assertGreater(len(self.dm.positions), 0)
        self.assertEqual(len(self.dm.positions[0]), 5)

    def test_location_is_valid(self):
        self.dm.inspect_locations()
        _, test_file = self.dm.data_files.get(0)
        for i, test_file in self.dm.data_files():
            with self.dm.msp_tar_gz_file_ctx(test_file) as f:
                for pos in self.dm.positions[i]:
                    f.seek(pos)
                    line = f.readline().strip()
                    self.assertTrue(line.startswith("Name:"))

    def test_total_spectra(self):
        self.dm.inspect_locations()
        self.assertEqual(self.dm.total_spectra(), 7)

    def test_data_point_size_mb(self):
        self.dm.maxes = (10, 10)
        data_point_size = self.dm.data_point_size_bytes()

        x_tensor = torch.ones((1, self.dm.maxes[0], 2), dtype=torch.float32)
        y_tensor = torch.ones((1, self.dm.maxes[1]), dtype=torch.int64)
        p_or_ch_tensor = torch.ones((1, 1), dtype=torch.int64)

        x_tensor_size = x_tensor.element_size() * x_tensor.numel()
        y_tensor_size = y_tensor.element_size() * y_tensor.numel()
        p_or_ch_tensor_size = p_or_ch_tensor.element_size() * p_or_ch_tensor.numel()
        tensors_sum = x_tensor_size + y_tensor_size + 2 * p_or_ch_tensor_size

        self.assertEqual(data_point_size, tensors_sum)

    def test_data_points_per_file(self):
        self.dm.positions = {0: [0]}
        self.dm.maxes = (50, 23) # yields exactly 1000 bytes per data point
        self.assertEqual(self.dm.data_point_size_bytes(), 1000)
        self.assertEqual(self.dm.data_point_per_file(1), 1048)

    def test_positions_iterator(self):
        self.dm.inspect_locations()
        it = self.dm.positions_iterator(0)
        self.assertIsInstance(it, Generator)
        positions = [pos for pos in self.dm.positions_iterator(0)]
        self.assertListEqual(positions, self.dm.positions[0])

    def test_tensor_files_count(self):
        self.dm.positions = {0: [0]}
        self.dm.maxes = (50, 23) # yields exactly one 1000 bytes per data point
        self.assertLess(self.dm.data_point_size_bytes(), 1024)
        self.assertEqual(self.dm.tensor_files_count(1 / 1024), 1)
        self.dm.maxes = (100, 25)
        self.assertGreater(self.dm.data_point_size_bytes(), 1024)
        self.assertEqual(self.dm.tensor_files_count(1 / 1024), 2)

    def test_save_and_load_self(self):
        self.dm.inspect_maxes()
        self.dm.inspect_locations()
        object_dict = self.dm.__dict__.copy()
        self.dm.save_manifest()
        self.assertTrue(self.dm.manifest_file_exists())
        self.assertGreater(os.path.getsize(self.dm.get_save_path()), 0)

        self.dm = DataManifest(Path("data/test_data"))
        self.dm.set_non_defualt_manifest_file_name("msp.manifest.test")
        self.dm.load_manifest()
        self.assertEqual(self.dm.__dict__["positions"], object_dict["positions"])
        self.assertEqual(self.dm.__dict__["maxes"], object_dict["maxes"])
        self.assertGreater(len(self.dm.__dict__["data_files"]), 0)
        self.assertEqual(len(self.dm.__dict__["data_files"]), len(object_dict["data_files"]))

    def test_batch_ids_generator(self):
        self.dm.inspect_locations()
        ids_collector = []
        for l in self.dm.batch_indices_generator(3):
            ids_collector.append(l)

        ids_aggregator = []
        for l in ids_collector:
            ids_aggregator += l
        ids_aggregator.sort()

        total_spectra_ids = [i for i in range(self.dm.total_spectra())]
        is_equal = True
        for i,j in zip(ids_aggregator, total_spectra_ids):
            if i != j:
                is_equal = False
        
        self.assertTrue(is_equal)

    def tearDown(self):
        path = self.dm.get_save_path()
        if os.path.exists(path):
            os.remove(path)


class MSPSplitDatasetTest(unittest.TestCase):
    def setUp(self):
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)
        inter = InterruptHandler()
        self.msp = MSPSplitDataset(Path("data/test_data"), inter)
        self.msp.manifest.set_non_defualt_manifest_file_name("msp.manifest.test")
        self.msp.set_base_tensor_file_name("msp.tensor.test")

    def test_create_tensors(self):
        self.msp.manifest.auto_manifest()
        counter = self.msp.create(10)
        self.assertEqual(self.msp.manifest.tensor_files_count(10), counter)

    def tearDown(self):
        path = self.msp.manifest.get_save_path()
        if os.path.exists(path):
            os.remove(path)
        for f in self.msp.manifest.data_path.glob("*msp.tensor.test"):
            f.unlink()
