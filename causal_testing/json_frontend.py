from pathlib import Path

import argparse
import json
import pandas as pd
import scipy
from fitter import Fitter, get_common_distributions

from json_frontend.examples.poisson import causal_test_setup as cts
from causal_testing.specification.variable import Input, Output, Meta
from causal_testing.specification.scenario import Scenario
from causal_testing.specification.causal_dag import CausalDAG
from causal_testing.specification.causal_specification import CausalSpecification
from causal_testing.generation.abstract_causal_test_case import AbstractCausalTestCase
from causal_testing.data_collection.data_collector import ObservationalDataCollector
from causal_testing.testing.causal_test_engine import CausalTestEngine

def get_args() -> argparse.Namespace:
    """ Command-line arguments

    :return: parsed command line arguments
    """
    parser = argparse.ArgumentParser(
        description="A script for parsing json config files for the Causal Testing Framework")
    parser.add_argument("-f", help="if included, the script will stop if a test fails",
                        action="store_true")
    parser.add_argument("--directory_path", help="path to the json file containing the causal tests",
                        default="causal_tests.json", type=Path)
    parser.add_argument("--json_path", help="path to the json file containing the causal tests",
                        default="causal_tests.json", type=Path)
    parser.add_argument("--data_path", help="path to the data csv file containing the runtime data of the scenario",
                        default="data.csv", type=Path)
    parser.add_argument("--dag_path", help="path to the DAG (Directed Acyclic  Graph) dot file",
                        default="dag.dot", type=Path)
    return parser.parse_args()


def populate_metas(metas: list[Meta], outputs: list[Output], data: pd.DataFrame):
    """ Populate data with meta-variable values and add distributions to Causal Testing Framework Variables

    :param list metas: list of Causal Testing Framework Meta classes
    :param list outputs: list of Causal Testing Framework Output classes
    :param pd.Dataframe data: dataframe containing runtime data of scenario
    """

    for meta in metas:
        meta.populate(data)

    for var in metas + outputs:
        f = Fitter(data[var.name], distributions=get_common_distributions())
        f.fit()
        (dist, params) = list(f.get_best(method="sumsquare_error").items())[0]
        var.distribution = getattr(scipy.stats, dist)(**params)
        print(var.name, f"{dist}({params})")


def json_parse(json_path: Path, data_path: Path) -> tuple[list, list, list, dict, pd.DataFrame]:
    """Parse a JSON input file into inputs, outputs, metas and a test plan

    :param json_path: path to JSON input file to be parsed
    :param data_path: path to the csv containing run-time data
    :returns:
        - inputs -  variables that are input parameters to the system
        - outputs -  variables that are outputs from the system
        - metas - variables that are not directly observable but are relevant to system testing, e.g. a model may take
         a location parameter and expand this out into average_age and household_size variables "under the hood".
          These parameters can be made explicit by instantiating them as metavariables.
        - test_plan - json object containing execution data for the system-under-test
    """
    with open(json_path) as f:
        test_plan = json.load(f)

    type_dict = {"float": float, "str": str, "int": int}

    inputs = [Input(i['name'], type_dict[i['type']], cts.distributions[i['distribution']]) for i in test_plan['inputs']]
    outputs = [Output(i['name'], type_dict[i['type']]) for i in test_plan['outputs']]
    metas = [Meta(i['name'], type_dict[i['type']], getattr(cts, i['populate'])) for i in
             test_plan['metas']] if "metas" in test_plan else []

    data = pd.read_csv(data_path)
    return inputs, outputs, metas, test_plan, data


# TODO: Work out how best to get constraints in there


def execute_test_case(causal_test_case, estimator, modelling_scenario, data_path: Path, causal_specification,
                      f_flag: bool) -> bool():
    """
    Takes a given test case and runs it using the CausalTestingFramework.

    :param causal_test_case: A causal test case is a triple (X, Delta, Y), where X is an input configuration, Delta is
        an intervention, and Y is the expected causal effect on a particular output.
    :param estimator: An estimator contains all of the information necessary to compute a causal estimate for the effect of changing
    a set of treatment variables to a set of values.
    :param modelling_scenario: a series of constraints placed over a set of input variables
    :param data_path: path to the data file
    :param causal_specification: Combination of Causal Dag and scenario
    :param f_flag: Flag that if true, will stop script upon a test failing
    :return: True if test failed, returns false otherwise.
    """
    failed = False
    data_collector = ObservationalDataCollector(modelling_scenario, data_path)
    causal_test_engine = CausalTestEngine(causal_test_case, causal_specification, data_collector)
    minimal_adjustment_set = causal_test_engine.load_data(index_col=0)
    treatment_vars = list(causal_test_case.treatment_input_configuration)
    minimal_adjustment_set = minimal_adjustment_set - {v.name for v in treatment_vars}
    # @andrewc19, why can we only have atomic control/treatment values?
    # I think it'd be good to pass it in as two dicts instead of vars, control, treatment lists
    estimation_model = estimator((list(treatment_vars)[0].name,),
                                 [causal_test_case.treatment_input_configuration[v] for v in treatment_vars][0],
                                 [causal_test_case.control_input_configuration[v] for v in treatment_vars][0],
                                 minimal_adjustment_set,
                                 (list(causal_test_case.outcome_variables)[0].name,),
                                 causal_test_engine.scenario_execution_data_df,
                                 effect_modifiers=causal_test_case.effect_modifier_configuration
                                 )
    if "intensity" in [v.name for v in treatment_vars] and hasattr(estimation_model, "add_squared_term_to_df"):
        estimation_model.add_squared_term_to_df("intensity")
    if isinstance(estimation_model, cts.WidthHeightEstimator):
        estimation_model.add_product_term_to_df("width", "intensity")
        estimation_model.add_product_term_to_df("height", "intensity")
    causal_test_result = causal_test_engine.execute_test(estimation_model, estimate_type=causal_test_case.estimate_type)
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
        print(f"    FAILED - expected {causal_test_case.expected_causal_effect}, got {causal_test_result.ate}")
    return failed


def default_path_names(directory_path: Path) -> tuple[Path, Path, Path]:
    """
    Takes a path of the directory containing all scenario specific files and creates individual paths for each file
    :param directory_path: pathlib.Path pointing towards directory containing all scenario specific user code and files
    :returns:
        - json_path -  path to causal_tests.json
        - dag_path -  path to dag.dot
        - data_path - path to scenario data, expected in data.csv
    """
    json_path = directory_path.joinpath("causal_tests.json")
    dag_path = directory_path.joinpath("dag.dot")
    data_path = directory_path.joinpath("data.csv")
    return json_path, dag_path, data_path


def main():

    args = get_args()
    if args.directory_path:
        json_path, dag_path, data_path = default_path_names(args.directory_path)
    else:
        json_path = args.json_path
        dag_path = args.dag_path
        data_path = args.data_path
    f_flag = args.f

    inputs, outputs, metas, test_plan, data = json_parse(json_path, data_path)
    populate_metas(metas, outputs, data)

    modelling_scenario = Scenario(inputs + outputs + metas, None)
    modelling_scenario.setup_treatment_variables()
    causal_specification = CausalSpecification(scenario=modelling_scenario, causal_dag=CausalDAG(dag_path))

    mutate = {
        "Increase": lambda x: modelling_scenario.treatment_variables[x].z3 > modelling_scenario.variables[x].z3,
        "ChangeByFactor(2)": lambda x: modelling_scenario.treatment_variables[x].z3 == modelling_scenario.variables[
            x].z3 * 2
    }

    executed_tests = 0
    failures = 0

    for test in test_plan['tests']:
        if "skip" in test and test['skip']:
            continue

        abstract_test = AbstractCausalTestCase(
            scenario=modelling_scenario,
            intervention_constraints=[mutate[v](k) for k, v in test['mutations'].items()],
            treatment_variables={modelling_scenario.variables[v] for v in test['mutations']},
            expected_causal_effect={modelling_scenario.variables[variable]: cts.effects[effect] for variable, effect in
                                    test["expectedEffect"].items()},
            effect_modifiers={modelling_scenario.variables[v] for v in
                              test['effect_modifiers']} if "effect_modifiers" in test else {},
            estimate_type=test['estimate_type']
        )

        concrete_tests, dummy = abstract_test.generate_concrete_tests(5, 0.05)
        print(abstract_test)
        print([(v.name, v.distribution) for v in abstract_test.treatment_variables])
        print(len(concrete_tests))
        for concrete_test in concrete_tests:
            executed_tests += 1
            failed = execute_test_case(concrete_test, cts.estimators[test['estimator']], modelling_scenario, data_path,
                                       causal_specification, f_flag)
            if failed:
                failures += 1

    print(f"{failures}/{executed_tests} failed")

