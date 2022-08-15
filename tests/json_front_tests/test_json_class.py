import unittest
from pathlib import Path
import os
import json
import scipy
import csv
from tests.test_helpers import create_temp_dir_if_non_existent, remove_temp_dir_if_existent
from causal_testing.json_front.json_class import JsonUtility
from causal_testing.specification.variable import Input, Output, Meta
from causal_testing.specification.scenario import Scenario
from causal_testing.specification.causal_specification import CausalSpecification


class TestJsonClass(unittest.TestCase):
    """ Test the CausalTestEngine workflow using observational data.

    The causal test engine (CTE) is the main workflow for the causal testing framework. The CTE takes a causal test case
    and a causal specification and computes the causal effect of the intervention on the outcome of interest.
    """

    def setUp(self) -> None:
        temp_dir_path = Path(create_temp_dir_if_non_existent())
        json_file_name = "tests.json"
        dag_file_name = "dag.dot"
        data_file_name = "data.csv"
        self.json_path = temp_dir_path / json_file_name
        self.dag_path = temp_dir_path / dag_file_name
        self.data_path = temp_dir_path / data_file_name
        self.setup_data_file()
        self.setup_json_file()
        self.setup_dag_file()
        self.json_class = JsonUtility("logs.log")
        self.example_distribution = scipy.stats.uniform(0, 10)
        self.input_dict_list = [{"name": "test_input", "type": float, "distribution": self.example_distribution}]
        self.output_dict_list = [{"name": "test_output", "type": float}]
        self.meta_dict_list = [{"name": "test_meta", "type": float, "populate": populate_example}]
        self.json_class.set_variables(self.input_dict_list, self.output_dict_list, None)
        self.json_class.set_path(self.json_path, self.dag_path, self.data_path)

    def test_setting_paths(self):
        self.assertEqual(self.json_class.json_path, Path(self.json_path))
        self.assertEqual(self.json_class.dag_path, Path(self.dag_path))
        self.assertEqual(self.json_class.data_path, Path(self.data_path))

    def test_set_inputs(self):
        ctf_input = [Input("test_input", float, self.example_distribution)]
        self.assertEqual(ctf_input[0].name, self.json_class.inputs[0].name)
        self.assertEqual(ctf_input[0].datatype, self.json_class.inputs[0].datatype)
        self.assertEqual(ctf_input[0].distribution, self.json_class.inputs[0].distribution)

    def test_set_outputs(self):
        ctf_output = [Output("test_output", float)]
        self.assertEqual(ctf_output[0].name, self.json_class.outputs[0].name)
        self.assertEqual(ctf_output[0].datatype, self.json_class.outputs[0].datatype)

    def test_set_metas(self):
        self.json_class.set_variables(self.input_dict_list, self.output_dict_list, self.meta_dict_list)
        ctf_meta = [Meta("test_meta", float, populate_example)]
        self.assertEqual(ctf_meta[0].name, self.json_class.metas[0].name)
        self.assertEqual(ctf_meta[0].datatype, self.json_class.metas[0].datatype)

    def test_argparse(self):
        args = self.json_class.get_args(["--data_path=data.csv", "--dag_path=dag.dot", "--json_path=tests.json"])
        self.assertTrue(args.data_path)
        self.assertTrue(args.dag_path)
        self.assertTrue(args.json_path)

    def test_setup_modelling_scenario(self):
        self.json_class.setup()
        print(type(self.json_class.modelling_scenario))
        print(self.json_class.modelling_scenario)
        self.assertIsInstance(self.json_class.modelling_scenario, Scenario)

    def test_setup_causal_specification(self):
        self.json_class.setup()

        self.assertIsInstance(self.json_class.causal_specification, CausalSpecification)

    def test_concrete_tests_generated(self):
        self.setup_json_file()
        self.setup_data_file()
        self.setup_dag_file()

    def tearDown(self) -> None:
        remove_temp_dir_if_existent()

    def setup_json_file(self):
        json_test = {
            "tests": [
                {"name": "test1", "mutations": {}, "estimator": None, "estimate_type": None, "effect_modifiers": [],
                 "expectedEffect": {}, "skip": False}]
        }
        json_object = json.dumps(json_test)
        with open(self.json_path, "w") as f:
            f.write(json_object)
        f.close()

    def setup_data_file(self):
        header = ['test_input', 'test_output']
        data = [1, 2]
        with open(self.data_path, "w") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerow(data)

    def setup_dag_file(self):
        dag_dot = (
            "digraph G {A->B}"
        )
        with open(self.dag_path, "w") as f:
            f.write(dag_dot)

def populate_example(*args, **kwargs):
    pass
