import unittest
from pathlib import Path
import scipy
import csv
import json

from causal_testing.testing.estimators import LinearRegressionEstimator
from causal_testing.testing.causal_test_outcome import NoEffect
from tests.test_helpers import create_temp_dir_if_non_existent, remove_temp_dir_if_existent
from causal_testing.json_front.json_class import JsonUtility
from causal_testing.specification.variable import Input, Output, Meta
from causal_testing.specification.scenario import Scenario
from causal_testing.specification.causal_specification import CausalSpecification
from causal_testing.generation.abstract_causal_test_case import AbstractCausalTestCase


class TestJsonClass(unittest.TestCase):
    """Test the JSON frontend for the Causal Testing Framework (CTF)

    The JSON frontend is an alternative interface for the CTF where tests are specified in JSON format and ingested
    with the frontend. Tests involve testing that this correctly interfaces with the framework with some dummy data
    """

    def setUp(self) -> None:
        json_file_name = "tests.json"
        dag_file_name = "dag.dot"
        data_file_name = "data.csv"
        test_data_dir_path = Path("tests/resources/data")
        self.json_path = str(test_data_dir_path / json_file_name)
        self.dag_path = str(test_data_dir_path / dag_file_name)
        self.data_path = [str(test_data_dir_path / data_file_name)]
        self.json_class = JsonUtility("logs.log")
        self.example_distribution = scipy.stats.uniform(1, 10)
        self.input_dict_list = [{"name": "test_input", "datatype": float, "distribution": self.example_distribution}]
        self.output_dict_list = [{"name": "test_output", "datatype": float}]
        self.meta_dict_list = [{"name": "test_meta", "datatype": float, "populate": populate_example}]
        self.json_class.set_variables(self.input_dict_list, self.output_dict_list, None)
        self.json_class.set_paths(self.json_path, self.dag_path, self.data_path)

    def test_setting_paths(self):
        self.assertEqual(self.json_class.paths.json_path, Path(self.json_path))
        self.assertEqual(self.json_class.paths.dag_path, Path(self.dag_path))
        self.assertEqual(self.json_class.paths.data_paths, [Path(self.data_path[0])])  # Needs to be list of Paths

    def test_set_inputs(self):
        ctf_input = [Input("test_input", float, self.example_distribution)]
        self.assertEqual(ctf_input[0].name, self.json_class.variables.inputs[0].name)
        self.assertEqual(ctf_input[0].datatype, self.json_class.variables.inputs[0].datatype)
        self.assertEqual(ctf_input[0].distribution, self.json_class.variables.inputs[0].distribution)

    def test_set_outputs(self):
        ctf_output = [Output("test_output", float)]
        self.assertEqual(ctf_output[0].name, self.json_class.variables.outputs[0].name)
        self.assertEqual(ctf_output[0].datatype, self.json_class.variables.outputs[0].datatype)

    def test_set_metas(self):
        self.json_class.set_variables(self.input_dict_list, self.output_dict_list, self.meta_dict_list)
        ctf_meta = [Meta("test_meta", float, populate_example)]
        self.assertEqual(ctf_meta[0].name, self.json_class.variables.metas[0].name)
        self.assertEqual(ctf_meta[0].datatype, self.json_class.variables.metas[0].datatype)

    def test_argparse(self):
        args = self.json_class.get_args(["--data_path=data.csv", "--dag_path=dag.dot", "--json_path=tests.json"])
        self.assertEqual(args.data_path, ["data.csv"])
        self.assertEqual(args.dag_path, "dag.dot")
        self.assertEqual(args.json_path, "tests.json")

    def test_setup_modelling_scenario(self):
        self.json_class.setup()
        self.assertIsInstance(self.json_class.modelling_scenario, Scenario)

    def test_setup_causal_specification(self):
        self.json_class.setup()
        self.assertIsInstance(self.json_class.causal_specification, CausalSpecification)

    def test_generate_tests_from_json(self):
        example_test = {
            "tests": [
                {
                    "name": "test1",
                    "mutations": {"test_input": "Increase"},
                    "estimator": "LinearRegressionEstimator",
                    "estimate_type": "ate",
                    "effect_modifiers": [],
                    "expectedEffect": {"test_output": "NoEffect"},
                    "skip": False,
                }
            ]
        }
        self.json_class.setup()
        self.json_class.test_plan = example_test
        effects = {"NoEffect": NoEffect()}
        mutates = {
            "Increase": lambda x: self.json_class.modelling_scenario.treatment_variables[x].z3
            > self.json_class.modelling_scenario.variables[x].z3
        }
        estimators = {"LinearRegressionEstimator": LinearRegressionEstimator}

        with self.assertLogs() as captured:
            self.json_class.generate_tests(effects, mutates, estimators, False)

        # Test that the final log message prints that failed tests are printed, which is expected behaviour for this scenario
        self.assertIn("failed", captured.records[-1].getMessage())

    def tearDown(self) -> None:
        pass
        # remove_temp_dir_if_existent()


def populate_example(*args, **kwargs):
    pass
