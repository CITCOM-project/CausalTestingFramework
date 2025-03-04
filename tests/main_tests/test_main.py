import unittest
import shutil
from pathlib import Path
from causal_testing.main import CausalTestingPaths


class TestCausalTestingPaths(unittest.TestCase):

    def setUp(self):
        self.dag_path = "tests/resources/data/dag.dot"
        self.data_paths = ["tests/resources/data/data.csv"]
        self.test_config_path = "tests/resources/data/tests.json"
        self.output_path = Path("results/results.json")

    def test_missing_dag(self):
        with self.assertRaises(FileNotFoundError) as e:
            CausalTestingPaths("missing.dot", self.data_paths, self.test_config_path, self.output_path).validate_paths()
        self.assertEqual("DAG file not found: missing.dot", str(e.exception))

    def test_missing_data(self):
        with self.assertRaises(FileNotFoundError) as e:
            CausalTestingPaths(self.dag_path, ["missing.csv"], self.test_config_path, self.output_path).validate_paths()
        self.assertEqual("Data file not found: missing.csv", str(e.exception))

    def test_missing_tests(self):
        with self.assertRaises(FileNotFoundError) as e:
            CausalTestingPaths(self.dag_path, self.data_paths, "missing.json", self.output_path).validate_paths()
        self.assertEqual("Test configuration file not found: missing.json", str(e.exception))

    def test_output_file_created(self):
        self.assertFalse(self.output_path.parent.exists())
        CausalTestingPaths(self.dag_path, self.data_paths, self.test_config_path, self.output_path).validate_paths()
        self.assertTrue(self.output_path.parent.exists())

    def tearDown(self):
        if self.output_path.parent.exists():
            shutil.rmtree(self.output_path.parent)
