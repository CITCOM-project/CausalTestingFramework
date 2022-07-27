import argparse
import logging
from pathlib import Path

from abc import ABC
import json
from fitter import Fitter, get_common_distributions
import pandas as pd
import scipy

from causal_testing.specification.variable import Input, Output, Meta
from causal_testing.specification.scenario import Scenario
from causal_testing.specification.causal_dag import CausalDAG
from causal_testing.specification.causal_specification import CausalSpecification
from causal_testing.generation.abstract_causal_test_case import AbstractCausalTestCase
from causal_testing.data_collection.data_collector import ObservationalDataCollector
from causal_testing.testing.causal_test_engine import CausalTestEngine
from causal_testing.testing.causal_test_case import CausalTestCase
from causal_testing.testing.estimators import Estimator

logger = logging.getLogger(__name__)

class JsonUtility(ABC):
    """
    The JsonUtility Class provides the functionality to use structured JSON to setup and run causal tests on the
    CausalTestingFramework.

    :attr {Path} json_path: Path to the JSON input file.
    :attr {Path} dag_path: Path to the dag.dot file containing the Causal DAG.
    :attr {Path} data_path: Path to the csv data file.
    :attr {Input} inputs: Causal variables representing inputs.
    :attr {Output} outputs: Causal variables representing outputs.
    :attr {Meta} metas: Causal variables representing metavariables.
    :attr {pd.DataFrame}: Pandas DataFrame containing runtime data.
    :attr {dict} test_plan: Dictionary containing the key value pairs from the loaded json test plan.
    :attr {Scenario} modelling_scenario:
    :attr {CausalSpecification} causal_specification:
    """

    def __init__(self, log_path):
        self.json_path = None
        self.dag_path = None
        self.data_path = None
        self.inputs = None
        self.outputs = None
        self.metas = None
        self.data = None
        self.test_plan = None
        self.modelling_scenario = None
        self.causal_specification = None
        self.setup_logger(log_path)

    def set_path(self, json_path: str, dag_path: str, data_path: str):
        """
            Takes a path of the directory containing all scenario specific files and creates individual paths for each file
            :param json_path: string path representation to .json file containing test specifications
            :param dag_path: string path representation to the .dot file containing the Causal DAG
            :param data_path: string path representation to the data file
            :returns:
                - json_path -
                - dag_path -
                - data_path -
            """
        self.json_path = Path(json_path)
        self.dag_path = Path(dag_path)
        self.data_path = Path(data_path)

    def set_variables(self, inputs: dict, outputs: dict, metas: dict, distributions: dict, populates: dict):
        """ Populate the Causal Variables
            :param inputs:
            :param outputs:
            :param metas:
            :param distributions:
            :param populates:
        """
        self.inputs = [Input(i['name'], i['type'], distributions[i['distribution']]) for i in
                       inputs]
        self.outputs = [Output(i['name'], i['type']) for i in outputs]
        self.metas = [Meta(i['name'], i['type'], populates[i['populate']]) for i in
                      metas] if metas else list()

    def setup(self):
        """ Function to populate all the necessary parts of the json_class needed to execute tests
        """
        self.modelling_scenario = Scenario(self.inputs + self.outputs + self.metas, None)
        self.modelling_scenario.setup_treatment_variables()
        self.causal_specification = CausalSpecification(scenario=self.modelling_scenario,
                                                        causal_dag=CausalDAG(self.dag_path))
        self._json_parse()
        self._populate_metas()

    def execute_tests(self, effects: dict, mutates: dict, estimators: dict, f_flag: bool):
        """ Runs and evaluates each test case specified in the JSON input

        :param effects: Dictionary mapping effect class instances to string representations.
        :param mutates: Dictionary mapping mutation functions to string representations.
        :param estimators: Dictionary mapping estimator classes to string representations.
        :param f_flag: Failure flag that if True the script will stop executing when a test fails.
        """
        executed_tests = 0
        failures = 0
        for test in self.test_plan['tests']:
            if "skip" in test and test['skip']:
                continue

            abstract_test = AbstractCausalTestCase(
                scenario=self.modelling_scenario,
                intervention_constraints=[mutates[v](k) for k, v in test['mutations'].items()],
                treatment_variables={self.modelling_scenario.variables[v] for v in test['mutations']},
                expected_causal_effect={self.modelling_scenario.variables[variable]: effects[effect] for
                                        variable, effect
                                        in
                                        test["expectedEffect"].items()},
                effect_modifiers={self.modelling_scenario.variables[v] for v in
                                  test['effect_modifiers']} if "effect_modifiers" in test else {},
                estimate_type=test['estimate_type']
            )

            concrete_tests, dummy = abstract_test.generate_concrete_tests(5, 0.05)
            logger.info(abstract_test)
            logger.info([(v.name, v.distribution) for v in abstract_test.treatment_variables])
            logger.info(len(concrete_tests))
            for concrete_test in concrete_tests:
                executed_tests += 1
                failed = self._execute_test_case(concrete_test, estimators[test['estimator']], f_flag)
                if failed:
                    failures += 1

        logger.info(f"{failures}/{executed_tests} failed")

    def _json_parse(self):
        """Parse a JSON input file into inputs, outputs, metas and a test plan
            :param distributions: dictionary of user defined scipy distributions
            :param populates: dictionary of user defined populate functions
        """
        with open(self.json_path) as f:
            self.test_plan = json.load(f)

        self.data = pd.read_csv(self.data_path)

    def _populate_metas(self):
        """
        Populate data with meta-variable values and add distributions to Causal Testing Framework Variables
        """

        for meta in self.metas:
            meta.populate(self.data)

        for var in self.metas + self.outputs:
            f = Fitter(self.data[var.name], distributions=get_common_distributions())
            f.fit()
            (dist, params) = list(f.get_best(method="sumsquare_error").items())[0]
            var.distribution = getattr(scipy.stats, dist)(**params)
            logger.info(var.name + f"{dist}({params})")

    def _execute_test_case(self, causal_test_case: CausalTestCase, estimator: Estimator, f_flag: bool) -> bool:
        """ Executes a singular test case, prints the results and returns the test case result
        :param causal_test_case: The concrete test case to be executed
        :param f_flag: Failure flag that if True the script will stop executing when a test fails.
        :return: A boolean that if True indicates the causal test case passed and if false indicates the test case failed.
        :rtype: bool
        """
        failed = False

        causal_test_engine, estimation_model = self._setup_test(causal_test_case, estimator)
        causal_test_result = causal_test_engine.execute_test(estimation_model,
                                                             estimate_type=causal_test_case.estimate_type)

        test_passes = causal_test_case.expected_causal_effect.apply(causal_test_result)

        result_string = str()
        if causal_test_result.ci_low() and causal_test_result.ci_high():
            result_string = f"{causal_test_result.ci_low()} < {causal_test_result.ate} <  {causal_test_result.ci_high()}"
        else:
            result_string = causal_test_result.ate
        if f_flag:
            assert test_passes, f"{causal_test_case}\n    FAILED - expected {causal_test_case.expected_causal_effect}, " \
                                f"got {result_string}"
        if not test_passes:
            failed = True
            logger.warning(
                f"    FAILED - expected {causal_test_case.expected_causal_effect}, got {causal_test_result.ate}")
        return failed

    def _setup_test(self, causal_test_case: CausalTestCase, estimator: Estimator) -> tuple[CausalTestEngine, Estimator]:
        """ Create the necessary inputs for a single test case
        :param causal_test_case: The concrete test case to be executed
        :returns:
                - causal_test_engine - Test Engine instance for the test being run
                - estimation_model - Estimator instance for the test being run
        """
        data_collector = ObservationalDataCollector(self.modelling_scenario, self.data_path)
        causal_test_engine = CausalTestEngine(causal_test_case, self.causal_specification, data_collector)
        minimal_adjustment_set = causal_test_engine.load_data(index_col=0)
        treatment_vars = list(causal_test_case.treatment_input_configuration)
        minimal_adjustment_set = minimal_adjustment_set - {v.name for v in treatment_vars}
        estimation_model = estimator((list(treatment_vars)[0].name,),
                                     [causal_test_case.treatment_input_configuration[v] for v in treatment_vars][0],
                                     [causal_test_case.control_input_configuration[v] for v in treatment_vars][0],
                                     minimal_adjustment_set,
                                     (list(causal_test_case.outcome_variables)[0].name,),
                                     causal_test_engine.scenario_execution_data_df,
                                     effect_modifiers=causal_test_case.effect_modifier_configuration
                                     )

        self.add_modelling_assumptions(estimation_model)

        return causal_test_engine, estimation_model

    def add_modelling_assumptions(self, estimation_model: Estimator):
        """ Optional abstract method where user functionality can be written to determine what assumptions are required
        for specific test cases
        :param estimation_model: estimator model instance for the current running test.
        """
        return

    @staticmethod
    def setup_logger(log_path: str):
        setup_log = logging.getLogger(__name__)
        fh = logging.FileHandler(Path(log_path) / "json_frontend.log")
        setup_log.addHandler(fh)

    @staticmethod
    def get_args() -> argparse.Namespace:
        """ Command-line arguments

        :return: parsed command line arguments
        """
        parser = argparse.ArgumentParser(
            description="A script for parsing json config files for the Causal Testing Framework")
        parser.add_argument(
            "-f", help="if included, the script will stop if a test fails", action="store_true")
        parser.add_argument(
            "--log_path", help="Specify a directory to change the location of the log file", default=".")
        parser.add_argument(
            "--data_path", help="Specify path to file containing runtime data", required=True
        )
        parser.add_argument(
            "--dag_path", help="Specify path to file containing the DAG, normally a .dot file", required=True
        )
        parser.add_argument(
            "--json_path", help="Specify path to file containing JSON tests, normally a .json file", required=True
        )
        return parser.parse_args()
