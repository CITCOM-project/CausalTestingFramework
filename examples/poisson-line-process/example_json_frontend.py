import logging

from causal_testing.estimation.linear_regression_estimator import LinearRegressionEstimator
from causal_testing.testing.causal_test_outcome import ExactValue, Positive, Negative, NoEffect
from causal_testing.json_front.json_class import JsonUtility
from causal_testing.specification.scenario import Scenario
from causal_testing.specification.variable import Input, Output


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG, format="%(message)s")

effects = {
    "Positive": Positive(),
    "Negative": Negative(),
    "ExactValue4_05": ExactValue(4, atol=0.5),
    "NoEffect": NoEffect(),
}

estimators = {
    "LinearRegressionEstimator": LinearRegressionEstimator,
}

# 2. Create variables
width = Input("width", float)
height = Input("height", float)
intensity = Input("intensity", float)

num_lines_abs = Output("num_lines_abs", float)
num_lines_unit = Output("num_lines_unit", float)
num_shapes_abs = Output("num_shapes_abs", float)
num_shapes_unit = Output("num_shapes_unit", float)

# 3. Create scenario by applying constraints over a subset of the input variables
scenario = Scenario(
    variables={
        width,
        height,
        intensity,
        num_lines_abs,
        num_lines_unit,
        num_shapes_abs,
        num_shapes_unit,
    }
)
scenario.setup_treatment_variables()

mutates = {
    "Increase": lambda x: scenario.treatment_variables[x].z3 > scenario.variables[x].z3,
    "ChangeByFactor(2)": lambda x: scenario.treatment_variables[x].z3 == scenario.variables[x].z3 * 2,
}


if __name__ == "__main__":
    args = JsonUtility.get_args()
    json_utility = JsonUtility(args.log_path)  # Create an instance of the extended JsonUtility class
    json_utility.set_paths(
        args.json_path, args.dag_path, args.data_path
    )  # Set the path to the data.csv, dag.dot and causal_tests.json file

    # Load the Causal Variables into the JsonUtility class ready to be used in the tests
    json_utility.setup(scenario=scenario)  # Sets up all the necessary parts of the json_class needed to execute tests

    json_utility.run_json_tests(effects=effects, mutates=mutates, estimators=estimators, f_flag=args.f)
