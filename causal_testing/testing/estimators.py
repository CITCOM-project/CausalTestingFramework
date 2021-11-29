from abc import ABC, abstractmethod
from sklearn.linear_model import LinearRegression
import pandas as pd


class Estimator(ABC):
    """ An estimator contains all of the information necessary to compute a causal estimate for the effect of changing
    a set of treatment variables to a set of values.

    All estimators must implement the abstract methods add_modelling_assumptions and estimate_ate.
    """

    def __init__(self, treatment: tuple, treatment_values: tuple, control_values: tuple, adjustment_set: set,
                 outcomes: tuple, df: pd.DataFrame):
        self.treatment = treatment
        self.treatment_values = treatment_values
        self.control_values = control_values
        self.adjustment_set = adjustment_set
        self.outcomes = outcomes
        self.df = df
        self.modelling_assumptions = []

    @abstractmethod
    def add_modelling_assumptions(self):
        """
        Add modelling assumptions to the estimator. This is a list of strings which list the modelling assumptions that
        must hold if the resulting causal inference is to be considered valid.
        """
        pass

    @abstractmethod
    def _estimate_unit_effect(self) -> float:
        """
        Estimate the unit effect of the treatment on the outcomes. That is, the coefficient of the treatment variable
        in the linear regression equation.
        :return: The intercept and coefficient of the linear regression equation
        """
        pass

    def compute_confidence_intervals(self) -> [float, float]:
        """
        Estimate the 95% Wald confidence intervals for the effect of changing the treatment from control values to
        treatment values on the outcomes.
        :return: 95% Wald confidence intervals.
        """
        pass


class LinearRegressionEstimator(Estimator):

    """
    A Linear Regression Estimator is a parametric estimator which restricts the variables in the data to a linear
    combination of parameters and functions of the variables (note these functions need not be linear).
    """

    def add_modelling_assumptions(self):
        """
        Add modelling assumptions to the estimator. This is a list of strings which list the modelling assumptions that
        must hold if the resulting causal inference is to be considered valid.
        """
        self.modelling_assumptions += 'The variables in the data must fit a shape which can be expressed as a linear'\
                                      'combination of parameters and functions of variables. Note that these functions'\
                                      'do not need to be linear.'

    def add_squared_term_to_df(self, term_to_square: str):
        """ Add a squared term to the linear regression model and df.

        This enables the user to capture curvilinear relationships with a linear regression model, not just straight
        lines, while automatically adding the modelling assumption imposed by the addition of this term.

        :param term_to_square: The term (column in data and variable in DAG) which is to be squared.
        """
        new_term = str(term_to_square) + '^2'
        self.df[new_term] = self.df[term_to_square]**2
        self.adjustment_set.add(new_term)
        self.modelling_assumptions += f'Relationship between {self.treatment} and {self.outcomes} varies quadratically'\
                                      f'with {term_to_square}.'

    def estimate_average_treatment_effect(self) -> float:
        """ Estimate the average treatment effect of changing the treatment from control value to treatment value
        (intervention) on the outcomes.

        :return: The average treatment effect.
        """
        _, unit_effect = self._estimate_unit_effect()
        unit_effect = unit_effect[0, 0]
        return unit_effect*self.treatment_values - unit_effect*self.control_values

    def _estimate_unit_effect(self) -> ([float], [[float]]):
        """ Estimate the unit effect of the treatment on the outcome. That is, the coefficient of the linear regression
        equation/the change in outcome caused by a unit change in treatment.

        :return: The unit effect.
        """
        # 1. Reduce dataframe to contain only the necessary columns
        reduced_df = self.df.copy()
        necessary_cols = list(self.treatment) + list(self.adjustment_set)
        missing_rows = reduced_df[necessary_cols].isnull().any(axis=1)
        reduced_df = reduced_df[~missing_rows]

        # 2. Estimate the unit difference in outcome caused by unit difference in treatment
        treatment_and_adjustments_cols = reduced_df[list(self.treatment) + list(self.adjustment_set)]
        outcomes_col = reduced_df[list(self.outcomes)]
        regression = LinearRegression()
        regression.fit(treatment_and_adjustments_cols, outcomes_col)
        return regression.intercept_, regression.coef_
