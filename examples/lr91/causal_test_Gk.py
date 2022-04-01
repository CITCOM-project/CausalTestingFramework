import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib as mpl
from causal_testing.specification.causal_dag import CausalDAG
from causal_testing.specification.scenario import Scenario
from causal_testing.specification.variable import Input, Output
from causal_testing.specification.causal_specification import CausalSpecification
from causal_testing.data_collection.data_collector import ObservationalDataCollector
from causal_testing.testing.causal_test_case import CausalTestCase
from causal_testing.testing.causal_test_outcome import Positive, Negative
from causal_testing.testing.intervention import Intervention
from causal_testing.testing.causal_test_engine import CausalTestEngine
from causal_testing.testing.estimators import LinearRegressionEstimator
from matplotlib.pyplot import rcParams
OBSERVATIONAL_DATA_PATH = "./data/observational_data.csv"

rc_fonts = {
    "font.size": 8,
    "figure.figsize": (10, 6),
    "text.usetex": True,
    "font.family": "serif",
    "text.latex.preamble": r"\usepackage{libertine}",
}
rcParams.update(rc_fonts)


def effects_on_APD90(observational_data_path, treatment_var, control_val, treatment_val, expected_causal_effect):
    """ Perform causal testing for the scenario in which we investigate the causal effect of G_K on APD90.

    :param observational_data_path: Path to observational data containing previous executions of the LR91 model.
    :param treatment_var: The input variable whose effect on APD90 we are interested in.
    :param control_val: The control value for the treatment variable (before intervention).
    :param treatment_val: The treatment value for the treatment variable (after intervention).
    :return: ATE for the effect of G_K on APD90
    """
    # 1. Define Causal DAG
    causal_dag = CausalDAG('./dag.dot')

    # 2. Specify all inputs
    g_na = Input('G_Na', float)
    g_si = Input('G_si', float)
    g_k = Input('G_K', float)
    g_k1 = Input('G_K1', float)
    g_kp = Input('G_Kp', float)
    g_b = Input('G_b', float)

    # 3. Specify all outputs
    max_voltage = Output('max_voltage', float)
    rest_voltage = Output('rest_voltage', float)
    max_voltage_gradient = Output('max_voltage_gradient', float)
    dome_voltage = Output('dome_voltage', float)
    apd50 = Output('APD50', int)
    apd90 = Output('APD90', int)

    # 3. Create scenario by applying constraints over a subset of the inputs
    scenario = Scenario(
        variables={g_na, g_si, g_k, g_k1, g_kp, g_b,
                   max_voltage, rest_voltage, max_voltage_gradient, dome_voltage, apd50, apd90},
        constraints=set()
    )

    # 4. Create a causal specification from the scenario and causal DAG
    causal_specification = CausalSpecification(scenario, causal_dag)

    # 5. Create a causal test case
    causal_test_case = CausalTestCase(control_input_configuration={treatment_var: control_val},
                                      expected_causal_effect=expected_causal_effect,
                                      outcome_variables={apd90},
                                      intervention=Intervention((treatment_var,), (treatment_val,), ), )

    # 6. Create a data collector
    data_collector = ObservationalDataCollector(scenario, observational_data_path)

    # 7. Create an instance of the causal test engine
    causal_test_engine = CausalTestEngine(causal_test_case, causal_specification, data_collector)

    # 8. Obtain the minimal adjustment set from the causal DAG
    minimal_adjustment_set = causal_test_engine.load_data(index_col=0)

    linear_regression_estimator = LinearRegressionEstimator((treatment_var.name,), treatment_val, control_val,
                                                            minimal_adjustment_set,
                                                            ('APD90',))
    causal_test_result = causal_test_engine.execute_test(linear_regression_estimator, 'ate')
    print(causal_test_result)


if __name__ == '__main__':
    # These variables should have a negative effect on APD90
    g_k = Input('G_K', float)
    g_b = Input('G_b', float)
    g_k1 = Input('G_K1', float)
    effects_on_APD90(OBSERVATIONAL_DATA_PATH, g_k, 0.28200, 0.28200*1.1, Negative)
    effects_on_APD90(OBSERVATIONAL_DATA_PATH, g_b, 0.60470, 0.60470*1.1, Negative)
    effects_on_APD90(OBSERVATIONAL_DATA_PATH, g_k1, 0.03921, 0.03921*1.1, Negative)

    # This variable should have a positive effect on APD90
    g_si = Input('G_si', float)
    effects_on_APD90(OBSERVATIONAL_DATA_PATH, g_si, 0.09000, 0.09000*1.1, Positive)



