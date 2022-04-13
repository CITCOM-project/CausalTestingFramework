import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from causal_testing.specification.causal_dag import CausalDAG
from causal_testing.specification.scenario import Scenario
from causal_testing.specification.variable import Input, Output
from causal_testing.specification.causal_specification import CausalSpecification
from causal_testing.data_collection.data_collector import ObservationalDataCollector
from causal_testing.testing.causal_test_case import CausalTestCase
from causal_testing.testing.causal_test_outcome import Positive, Negative, NoEffect
from causal_testing.testing.intervention import Intervention
from causal_testing.testing.causal_test_engine import CausalTestEngine
from causal_testing.testing.estimators import LinearRegressionEstimator
from matplotlib.pyplot import rcParams
OBSERVATIONAL_DATA_PATH = "./data/normalised_results.csv"

rc_fonts = {
    "font.size": 8,
    "figure.figsize": (6, 4),
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
                                                            ('APD90',),
                                                           )
    causal_test_result = causal_test_engine.execute_test(linear_regression_estimator, 'ate')
    print(causal_test_result)
    return causal_test_result.ate, causal_test_result.confidence_intervals


def plot_ates_with_cis(results_dict, xs, save=True):
    """
    Plot the average treatment effects for a given treatment against a list of x-values with confidence intervals.

    :param results_dict: A dictionary containing results for sensitivity analysis of each input parameter.
    :param xs: Values to be plotted on the x-axis.
    """
    fig, axes = plt.subplots()
    for treatment, results in results_dict.items():
        ates = results['ate']
        cis = results['cis']
        before_underscore, after_underscore = treatment.split('_')
        after_underscore_braces = f"{{{after_underscore}}}"
        latex_compatible_treatment_str = rf"${before_underscore}_{after_underscore_braces}$"
        cis_low = [ci[0] for ci in cis]
        cis_high = [ci[1] for ci in cis]

        axes.fill_between(xs, cis_low, cis_high, alpha=.3, label=latex_compatible_treatment_str)
        axes.plot(xs, ates, color='black', linewidth=.5)
        axes.plot(xs, [0] * len(xs), color='red', alpha=.5, linestyle='--', linewidth=.5)
    axes.set_ylabel(r"ATE: Change in $APD_{90} (ms)$")
    axes.set_xlabel(r"Treatment value")
    axes.set_ylim(-150, 150)
    axes.set_xlim(min(xs), max(xs))
    box = axes.get_position()
    axes.set_position([box.x0, box.y0 + box.height * 0.3,
                       box.width, box.height * 0.7])
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), fancybox=True, ncol=6, title='Input Parameters')
    if save:
        plt.savefig(f"APD90_sensitivity.pdf", format="pdf")
    plt.show()


def normalise_data(df, columns=None):
    """ Normalise the data in the dataframe into the range [0, 1]. """
    if columns:
        df[columns] = (df[columns] - df[columns].min())/(df[columns].max() - df[columns].min())
        return df
    else:
        return (df - df.min())/(df.max() - df.min())


if __name__ == '__main__':
    df = pd.read_csv("data/results.csv")
    conductance_means = {'G_K': (.5, Positive),
                         'G_b': (.5, Positive),
                         'G_K1': (.5, Positive),
                         'G_si': (.5, Negative),
                         'G_Na': (.5, NoEffect),
                         'G_Kp': (.5, NoEffect)}
    normalised_df = normalise_data(df, columns=list(conductance_means.keys()))
    normalised_df.to_csv("data/normalised_results.csv")

    treatment_values = np.linspace(0, 1, 20)
    results_dict = {'G_K': {},
                    'G_b': {},
                    'G_K1': {},
                    'G_si': {},
                    'G_Na': {},
                    'G_Kp': {}}
    for conductance_param, mean_and_oracle in conductance_means.items():
        average_treatment_effects = []
        confidence_intervals = []
        for treatment_value in treatment_values:
            mean, oracle = mean_and_oracle
            conductance_input = Input(conductance_param, float)
            ate, cis = effects_on_APD90(OBSERVATIONAL_DATA_PATH, conductance_input, 0, treatment_value, oracle)
            average_treatment_effects.append(ate)
            confidence_intervals.append(cis)
        results_dict[conductance_param] = {"ate": average_treatment_effects, "cis": confidence_intervals}
    plot_ates_with_cis(results_dict, treatment_values)
