from abc import ABC, abstractmethod
import statsmodels.api as sm
import pandas as pd


class Estimator(ABC):
    """ An estimator contains all of the information necessary to compute a causal estimate for the effect of changing
    a set of treatment variables to a set of values.

    All estimators must implement the abstract methods add_modelling_assumptions and estimate_ate.
    """

    def __init__(self, treatment: tuple, treatment_values: tuple, control_values: tuple, adjustment_set: set,
                 outcomes: tuple, df: pd.DataFrame, estimand: str = None):
        self.treatment = treatment
        self.treatment_values = treatment_values
        self.control_values = control_values
        self.adjustment_set = adjustment_set
        self.outcomes = outcomes
        self.df = df
        if estimand:
            self.estimand = estimand
        self.modelling_assumptions = []

    @abstractmethod
    def add_modelling_assumptions(self):
        """
        Add modelling assumptions to the estimator. This is a list of strings which list the modelling assumptions that
        must hold if the resulting causal inference is to be considered valid.
        """
        pass

    @abstractmethod
    def _run_linear_regression(self) -> float:
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

    def add_product_term_to_df(self, term_a: str, term_b: str):
        """ Add a product term to the linear regression model and df.

        This enables the user to capture interaction between a pair of variables in the model. In other words, while
        each covariate's contribution to the mean is assumed to be independent of the other covariates, the pair of
        product terms term_a*term_b a are restricted to vary linearly with each other.

        :param term_a: The first term of the product term.
        :param term_b: The second term of the product term.
        """
        new_term = str(term_a) + '*' + str(term_b)
        self.df[new_term] = self.df[term_a] * self.df[term_b]
        self.adjustment_set.add(new_term)
        self.modelling_assumptions += f'{term_a} and {term_b} vary linearly with each other.'

    def estimate_average_treatment_effect(self) -> float:
        """ Estimate the average treatment effect of changing the treatment from control value to treatment value
        (intervention) on the outcomes.

        :return: The average treatment effect and the 95% Wald confidence intervals.
        """
        params, [ci_low, ci_high] = self._run_linear_regression()
        unit_effect = params[list(self.treatment)].values[0]  # Unit effect is the partial coefficient of treatment
        return unit_effect*self.treatment_values - unit_effect*self.control_values, [ci_low, ci_high]

    def _run_linear_regression(self) -> pd.Series:
        """ Run linear regression of the treatment and adjustment set against the outcomes and return the results.

        :return: A pandas series containing the regression results for each parameter in the linear mean model and the
        95% Wald confidence intervals.
        """
        # 1. Reduce dataframe to contain only the necessary columns
        reduced_df = self.df.copy()
        necessary_cols = list(self.treatment) + list(self.adjustment_set) + list(self.outcomes)
        missing_rows = reduced_df[necessary_cols].isnull().any(axis=1)
        reduced_df = reduced_df[~missing_rows]

        # 2. Estimate the unit difference in outcome caused by unit difference in treatment
        treatment_and_adjustments_cols = reduced_df[list(self.treatment) + list(self.adjustment_set)]
        outcomes_col = reduced_df[list(self.outcomes)]
        regression = sm.OLS(outcomes_col, treatment_and_adjustments_cols)
        results = regression.fit()
        print(results.summary().tables[1])
        confidence_intervals = results.conf_int(alpha=0.05, cols=None)
        ci_low, ci_high = confidence_intervals[0][list(self.treatment)], confidence_intervals[1][list(self.treatment)]
        ci_low, ci_high = ci_low.values[0], ci_high.values[0]
        return results.params, [ci_low, ci_high]
