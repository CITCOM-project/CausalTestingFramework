from causal_testing.specification.causal_dag import CausalDAG
from causal_testing.specification.scenario import Scenario
from causal_testing.specification.variable import Input, Output
from causal_testing.specification.causal_specification import CausalSpecification
from causal_testing.data_collection.data_collector import ObservationalDataCollector
from causal_testing.testing.causal_test_case import CausalTestCase
from causal_testing.testing.causal_test_outcome import ExactValue, Positive
from causal_testing.testing.causal_test_engine import CausalTestEngine
from causal_testing.testing.estimators import LinearRegressionEstimator, Estimator

import pandas as pd


class EmpiricalMeanEstimator(Estimator):
    def add_modelling_assumptions(self):
        """
        Add modelling assumptions to the estimator. This is a list of strings which list the modelling assumptions that
        must hold if the resulting causal inference is to be considered valid.
        """
        self.modelling_assumptions += (
            "The data must contain runs with the exact configuration of interest."
        )

    def estimate_ate(self) -> float:
        """ Estimate the outcomes under control and treatment.
        :return: The empirical average treatment effect.
        """
        control_results = self.df.where(
            self.df[self.treatment[0]] == self.control_values
        )[self.outcome].dropna()
        treatment_results = self.df.where(
            self.df[self.treatment[0]] == self.treatment_values
        )[self.outcome].dropna()
        return treatment_results.mean()[0] - control_results.mean()[0], None

    def estimate_risk_ratio(self) -> float:
        """ Estimate the outcomes under control and treatment.
        :return: The empirical average treatment effect.
        """
        control_results = self.df.where(
            self.df[self.treatment[0]] == self.control_values
        )[self.outcome].dropna()
        treatment_results = self.df.where(
            self.df[self.treatment[0]] == self.treatment_values
        )[self.outcome].dropna()
        return treatment_results.mean()[0] / control_results.mean()[0], None


# 1. Read in the Causal DAG
causal_dag = CausalDAG("./dag.dot")

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

# 4. Construct a causal specification from the scenario and causal DAG
causal_specification = CausalSpecification(scenario, causal_dag)


def test_intensity_num_shapes(
    observational_data_path,
    causal_test_case,
    square_terms=[],
    inverse_terms=[],
    empirical=False,
):
    # 6. Create a data collector
    data_collector = ObservationalDataCollector(scenario, observational_data_path)

    # 7. Create an instance of the causal test engine
    causal_test_engine = CausalTestEngine(
        causal_specification, data_collector
    )

    # 8. Obtain the minimal adjustment set for the causal test case from the causal DAG
    causal_test_engine.identification(causal_test_case)

    # 9. Set up an estimator
    data = pd.read_csv(observational_data_path)

    treatment = list(causal_test_case.control_input_configuration)[0].name
    outcome = list(causal_test_case.outcome_variables)[0].name

    estimator = None
    if empirical:
        estimator = EmpiricalMeanEstimator(
            treatment=[treatment],
            control_values=list(causal_test_case.control_input_configuration.values())[
                0
            ],
            treatment_values=list(
                causal_test_case.treatment_input_configuration.values()
            )[0],
            adjustment_set=set(),
            outcome=[outcome],
            df=data,
            effect_modifiers=causal_test_case.effect_modifier_configuration,
        )
    else:
        estimator = LinearRegressionEstimator(
            treatment=[treatment],
            control_values=list(causal_test_case.control_input_configuration.values())[
                0
            ],
            treatment_values=list(
                causal_test_case.treatment_input_configuration.values()
            )[0],
            adjustment_set=set(),
            outcome=[outcome],
            df=data,
            intercept=0,
            effect_modifiers=causal_test_case.effect_modifier_configuration,
        )
        for t in square_terms:
            estimator.add_squared_term_to_df(t)
        for t in inverse_terms:
            estimator.add_inverse_term_to_df(t)

    # 10. Execute the test
    causal_test_result = causal_test_engine.execute_test(
        estimator, causal_test_case, causal_test_case.estimate_type
    )

    return causal_test_result


observational_data_path = "data/random/data_random_1000.csv"

intensity_num_shapes_results = []

for wh in range(1, 11):
    smt_data_path = f"data/smt_100/data_smt_wh{wh}_100.csv"
    for control_value, treatment_value in [(1, 2), (2, 4), (4, 8), (8, 16)]:
        print("=" * 33, "CAUSAL TEST", "=" * 33)
        print(f"WIDTH = HEIGHT = {wh}")

        print("Identifying")
        # 5. Create a causal test case
        causal_test_case = CausalTestCase(
            control_input_configuration={intensity: control_value},
            treatment_input_configuration={intensity: treatment_value},
            expected_causal_effect=ExactValue(4, tolerance=0.5),
            outcome_variables={num_shapes_unit},
            estimate_type="risk_ratio",
            # effect_modifier_configuration={width: wh, height: wh}
        )
        obs_causal_test_result = test_intensity_num_shapes(
            observational_data_path,
            causal_test_case,
            square_terms=["intensity"],
            empirical=False,
        )
        print("Observational", end=" ")
        print(obs_causal_test_result)
        smt_causal_test_result = test_intensity_num_shapes(
            smt_data_path, causal_test_case, square_terms=["intensity"], empirical=True
        )
        print("RCT", end=" ")
        print(smt_causal_test_result)

        results = {
            "width": wh,
            "height": wh,
            "control": control_value,
            "treatment": treatment_value,
            "smt_risk_ratio": smt_causal_test_result.test_value.value,
            "obs_risk_ratio": obs_causal_test_result.test_value.value,
        }
        intensity_num_shapes_results.append(results)

intensity_num_shapes_results = pd.DataFrame(intensity_num_shapes_results)
intensity_num_shapes_results.to_csv("intensity_num_shapes_results_random_1000.csv")
print(intensity_num_shapes_results)

width_num_shapes_results = []
for i in range(17):
    for w in range(1, 10):
        print("=" * 37, "CAUSAL TEST", "=" * 37)
        print("Identifying")
        # 5. Create a causal test case
        control_value = w
        treatment_value = w + 1
        causal_test_case = CausalTestCase(
            control_input_configuration={width: control_value},
            treatment_input_configuration={width: treatment_value},
            expected_causal_effect=Positive(),
            outcome_variables={num_shapes_unit},
            estimate_type="ate_calculated",
            effect_modifier_configuration={intensity: i},
        )
        causal_test_result = test_intensity_num_shapes(
            observational_data_path,
            causal_test_case,
            square_terms=["intensity"],
            inverse_terms=["width"],
        )
        print(causal_test_result)
        results = {
            "control": control_value,
            "treatment": treatment_value,
            "intensity": i,
            "ate": causal_test_result.ate,
            "ci_low": min(causal_test_result.confidence_intervals),
            "ci_high": max(causal_test_result.confidence_intervals),
        }
        width_num_shapes_results.append(results)
width_num_shapes_results = pd.DataFrame(width_num_shapes_results)
width_num_shapes_results.to_csv("width_num_shapes_results_random_1000.csv")
print(width_num_shapes_results)
