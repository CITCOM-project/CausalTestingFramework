import numpy as np
import pandas as pd
import scipy
import argparse
from pathlib import Path

from causal_testing.testing.estimators import LinearRegressionEstimator, CausalForestEstimator
from causal_testing.testing.causal_test_outcome import ExactValue, Positive, Negative, NoEffect, CausalTestOutcome, \
    CausalTestResult
from causal_testing.json_front.json_class import JsonUtility


class WidthHeightEstimator(LinearRegressionEstimator):
    """
    Extension of LinearRegressionEstimator class to include scenario specific user code
    """

    def add_modelling_assumptions(self):
        """
        Add modelling assumptions to the estimator. This is a list of strings which list the modelling assumptions that
        must hold if the resulting causal inference is to be considered valid.
        """
        self.modelling_assumptions += 'The output varies according to 2i(w+h)'

    def estimate_ate(self) -> (float, [float, float], float):
        """ Estimate the conditional average treatment effect of the treatment on the outcome. That is, the change
        in outcome caused by changing the treatment variable from the control value to the treatment value.
        :return: The conditional average treatment effect and the 95% Wald confidence intervals.
        """
        assert self.effect_modifiers, f"Must have at least one effect modifier to compute CATE - {self.effect_modifiers}."
        x = pd.DataFrame()
        x[self.treatment[0]] = [self.treatment_values, self.control_values]
        x['Intercept'] = 1
        for k, v in self.effect_modifiers.items():
            self.adjustment_set.add(k)
            x[k] = v
        if hasattr(self, "product_terms"):
            for a, b in self.product_terms:
                x[f"{a}*{b}"] = x[a] * x[b]

        x.drop(['width', 'height'], axis=1, inplace=True)
        self.adjustment_set = {"width*intensity", "height*intensity"}

        print(x)
        print(self.adjustment_set)

        model = self._run_linear_regression()
        print(model.summary())
        y = model.predict(x)
        treatment_outcome = y.iloc[0]
        control_outcome = y.iloc[1]

        return treatment_outcome - control_outcome, None


class PoissonWidthHeight(CausalTestOutcome):
    """An extension of TestOutcome representing that the expected causal effect should be positive."""

    def __init__(self, tolerance=0.5):
        self.tolerance = tolerance
        self.i2c = None

    def apply(self, res: CausalTestResult) -> bool:
        # TODO: confidence intervals?
        print("=== APPLYING ===")
        print("effect_modifier_configuration", res.effect_modifier_configuration)
        effect_modifier_configuration = {k.name: v for k, v in res.effect_modifier_configuration.items()}
        c = res.treatment_value - res.control_value
        i = effect_modifier_configuration['intensity']
        self.i2c = i * 2 * c
        print("2ic:", f"2*{i}*{c}={self.i2c}")
        print("ate:", res.ate)
        return np.isclose(res.ate, self.i2c, atol=self.tolerance)

    def __str__(self):
        if self.i2c is None:
            return f"PoissonWidthHeight±{self.tolerance}"
        return f"PoissonWidthHeight:{self.i2c}±{self.tolerance}"


def populate_width_height(data):
    data['width_plus_height'] = data['width'] + data['height']


def populate_num_lines_unit(data):
    area = data['width'] * data['height']
    data['num_lines_unit'] = data['num_lines_abs'] / area


def populate_num_shapes_unit(data):
    area = data['width'] * data['height']
    data['num_shapes_unit'] = data['num_lines_abs'] / area


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
    return parser.parse_args()


populates = {
    "populate_width_height": populate_width_height,
    "populate_num_lines_unit": populate_num_lines_unit,
    "populate_num_shapes_unit": populate_num_shapes_unit
}

distributions = {
    "uniform": scipy.stats.uniform(0, 10)
}

effects = {
    "PoissonWidthHeight": PoissonWidthHeight(),
    "Positive": Positive(),
    "Negative": Negative(),
    "ExactValue4_05": ExactValue(4, tolerance=0.5),
    "NoEffect": NoEffect()
}

estimators = {
    "WidthHeightEstimator": WidthHeightEstimator,
    "CausalForestEstimator": CausalForestEstimator,
    "LinearRegressionEstimator": LinearRegressionEstimator,
}


class MyJsonUtility(JsonUtility):

    def add_modelling_assumptions(self, estimator: LinearRegressionEstimator):

        if "intensity" in [v for v in estimator.treatment[0]] and hasattr(estimator, "add_squared_term_to_df"):
            estimator.add_squared_term_to_df("intensity")
        if isinstance(estimator, WidthHeightEstimator):
            estimator.add_product_term_to_df("width", "intensity")
            estimator.add_product_term_to_df("height", "intensity")

if __name__ == "__main__":
    args = get_args()
    json_utility = MyJsonUtility()
    json_utility.set_paths(args.directory_path)

    json_utility.json_parse(distributions, populates)
    json_utility.populate_metas()
    json_utility.setup_scenario()

    mutates = {
        "Increase": lambda x: json_utility.modelling_scenario.treatment_variables[x].z3 > json_utility.modelling_scenario.variables[x].z3,
        "ChangeByFactor(2)": lambda x: json_utility.modelling_scenario.treatment_variables[x].z3 == json_utility.modelling_scenario.variables[
            x].z3 * 2
    }

    json_utility.execute_tests(effects, mutates, estimators, args.f)
