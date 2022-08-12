import unittest
from pathlib import Path
import os
from tests.test_helpers import create_temp_dir_if_non_existent, remove_temp_dir_if_existent
from causal_testing.json_front.json_class import JsonUtility
from causal_testing.specification.variable import Input, Output, Meta


class TestJsonClass(unittest.TestCase):
    """ Test the CausalTestEngine workflow using observational data.

    The causal test engine (CTE) is the main workflow for the causal testing framework. The CTE takes a causal test case
    and a causal specification and computes the causal effect of the intervention on the outcome of interest.
    """

    def setUp(self) -> None:
        temp_dir_path = create_temp_dir_if_non_existent()
        json_file_name = "tests.json"
        dag_file_name = "dag.dot"
        data_file_name = "data.csv"
        self.json_path = os.path.join(temp_dir_path, json_file_name)
        self.dag_path = os.path.join(temp_dir_path, json_file_name)
        self.data_path = os.path.join(temp_dir_path, json_file_name)
        self.json_class = JsonUtility("logs.log")
        self.input_dict_list = [{"name": "test_input", "type": float, "distribution": "uniform"}]
        self.output_dict_list = [{"name": "test_output", "type": float}]
        self.meta_dict_list = [{"name": "test_meta", "type": float, "populate": "populate_num_lines_unit"}]
        self.json_class.set_variables(self.input_dict_list, self.output_dict_list, self.meta_dict_list)

    def test_setting_paths(self):
        self.json_class.set_path(self.json_path, self.dag_path, self.data_path)
        self.assertEqual(self.json_class.json_path, Path(self.json_path))
        self.assertEqual(self.json_class.dag_path, Path(self.dag_path))
        self.assertEqual(self.json_class.data_path, Path(self.data_path))

    def test_set_inputs(self):

        pass


    def tearDown(self) -> None:
        remove_temp_dir_if_existent()
