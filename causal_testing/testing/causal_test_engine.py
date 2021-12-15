import pandas as pd
import logging
from causal_testing.data_collection.data_collector import ExperimentalDataCollector, ObservationalDataCollector
from causal_testing.testing.causal_test_case import CausalTestCase
from causal_testing.testing.causal_test_outcome import CausalTestResult
from causal_testing.specification.causal_specification import CausalSpecification
from causal_testing.testing.estimators import Estimator

logger = logging.getLogger(__name__)


class CausalTestEngine:
    """
    Overarching workflow for Causal Testing. The CausalTestEngine proceeds in four steps.
    (1) Given a causal test case, specification, and (optionally) observational data, compute the causal estimand
        that, once estimated, yields the desired causal effect. This is essentially the recipe for a statistical
        procedure that, according to the assumptions encoded in the causal specification, estimates the casual effect
        of interest.
    (2) If using observational data, check whether the data is sufficient for estimating the causal effect of interest.
        If the data is insufficient, identify the missing data to guide the user towards un-exercised areas of the
        system-under-test. Else, if generating experimental data, run the model in the experimental conditions required
        to isolate the causal effect of interest.
    (3) Using the gathered data (whether observational or experimental), implement the statistical procedure prescribed
        by the causal estimand. For example, apply a linear regression model which includes a term for the set of
        variables which block (d-separate) all back-door paths. Return the causal estimate obtained following this
        procedure and, optionally, (depending on the estimator used) confidence intervals for this estimate. These are
        provided as an instance of the CausalTestResult class.
    (4) Define a test oracle procedure which uses the causal test results to determine whether the intervention has
        had the anticipated causal effect. This should assign a pass/fail value to the CausalTestResult.
    """

    def __init__(self, causal_test_case: CausalTestCase, causal_specification: CausalSpecification):
        self.causal_test_case = causal_test_case
        self.casual_dag, self.scenario = causal_specification.causal_dag, causal_specification.scenario
        self.scenario_execution_data_df = pd.DataFrame()

    def load_data(self, observational_data_path: str = None, n_repeats: int = 1, **kwargs):
        """ Load execution data corresponding to the causal test case into a pandas dataframe and return the minimal
        adjustment set.

        Data can be loaded in two ways:
            (1) Experimentally - the model is executed with the treatment and control input configurations under
                conditions that guarantee the observed change in outcome must be caused by the change in input
                (intervention).
            (2) Observationally - previous execution data is supplied in the form of a csv which is then filtered
                to remove any data corresponding to executions of a different scenario
                (i.e. not the scenario-under-test) and checked for positivity violations.

        After the data is loaded, both are treated in the same way and, provided the identifiability and modelling
        assumptions hold, can be used to estimate the causal effect for the causal test case.

        :param observational_data_path: An optional path to a csv containing observational data.
        :param n_repeats: An optional int which specifies the number of times to run a causal test case in the
        experimental case.
        :return self: Update the causal test case's execution data dataframe.
        :return minimal_adjustment_set: The smallest set of variables which can be adjusted for to obtain a causal
        estimate as opposed to a purely associational estimate.
        """

        if observational_data_path:
            observational_data_collector = ObservationalDataCollector(self.scenario, observational_data_path)
            scenario_execution_data_df = observational_data_collector.collect_data(**kwargs)
        else:
            experimental_data_collector = ExperimentalDataCollector(self.scenario,
                                                                    self.causal_test_case.control_input_configuration,
                                                                    self.causal_test_case.treatment_input_configuration,
                                                                    n_repeats=n_repeats)
            scenario_execution_data_df = experimental_data_collector.collect_data()

        self.scenario_execution_data_df = scenario_execution_data_df
        minimal_adjustment_sets = self.casual_dag.enumerate_minimal_adjustment_sets(
                self.causal_test_case.get_treatment_variables(),
                self.causal_test_case.get_outcome_variables()
            )
        minimal_adjustment_set = min(minimal_adjustment_sets, key=len)
        return minimal_adjustment_set

    def execute_test(self, estimator: Estimator, estimate_type: str = 'ate') -> CausalTestResult:
        """ Execute a causal test case and return the causal test result.

        Test case execution proceeds with the following steps:
        (1) Check that data has been loaded using the method load_data
        (2) Check loaded data for any positivity violations and warn the user if so
        (3) Instantiate the estimator with the values of the causal test case.
        (4) Using the estimator, estimate the average treatment effect of the changing the treatment from control value
            to treatment value on the outcome of interest, adjusting for the identified adjustment set.
        (5) Depending on the estimator used, compute 95% confidence intervals for the estimate.
        (6) Store results in an instance of CausalTestResults.
        (7) Apply test oracle procedure to assign a pass/fail to the CausalTestResult and return.

        :param estimator: A reference to an Estimator class.
        :param estimate_type: A string which denotes the type of estimate to return, ATE or CATE.
        :return causal_test_result: A CausalTestResult for the executed causal test case.
        """
        if self.scenario_execution_data_df.empty:
            raise Exception('No data has been loaded. Please call load_data prior to executing a causal test case.')
        estimator.df = self.scenario_execution_data_df
        if estimate_type == 'cate':
            if not hasattr(estimator, 'estimate_cates'):
                raise NotImplementedError(f'{estimator.__class__} has no CATE method.')
            else:
                cates_df = estimator.estimate_cates()
                return cates_df
        else:
            ate, confidence_intervals = estimator.estimate_ate()
            causal_test_result = CausalTestResult(estimator.treatment, estimator.outcome, estimator.treatment_values,
                                                  estimator.control_values, estimator.adjustment_set, ate,
                                                  confidence_intervals)
            # causal_test_result.apply_test_oracle_procedure(self.causal_test_case.expected_causal_effect)
        return causal_test_result

    # TODO (MF) I think that the test oracle procedure should go in here.
    # This way, the user can supply it as a function or something, which can be applied to the result of CI

    def _check_positivity_violation(self, variables_list):
        """ Check whether the dataframe has a positivity violation relative to the specified variables list.

        A positivity violation occurs when there is a stratum of the dataframe which does not have any data. Put simply,
        if we split the dataframe into covariate sub-groups, each sub-group must contain both a treated and untreated
        individual. If a positivity violation occurs, causal inference is still possible using a properly specified
        parametric estimator. Therefore, we should not throw an exception upon violation but raise a warning instead.

        :param variables_list: The list of variables for which positivity must be satisfied.
        :return: True if positivity is violated, False otherwise.
        """
        # TODO: Improve positivity checks to look for stratum-specific violations, not just missing variables in df
        if not set(variables_list).issubset(self.scenario_execution_data_df.columns):
            missing_variables = set(variables_list) - set(self.scenario_execution_data_df.columns)
            logger.warning(f'Positivity violation: missing data for variables {missing_variables}.\n'
                           f'Causal inference is only valid if a well-specified parametric model is used.\n'
                           f'Alternatively, consider restricting analysis to executions without the variables:'
                           f' {missing_variables}.')
            return True
        else:
            return False
