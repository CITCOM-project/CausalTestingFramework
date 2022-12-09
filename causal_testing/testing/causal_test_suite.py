class CausalTestSuite:
    """
    A CausalTestSuite is a class holding a dictionary, where the keys are base_test_cases representing the treatment
    and outcome Variables, and the values are test objects. Test Objects hold a causal_test_case_list which is a
    list of causal_test_cases which provide control and treatment values, and an iterator of Estimator Class References

    This structure can be fed into the CausalTestEngines execute_test_suite function which will iterate over all the
    base_test_case's and execute each causal_test_case with each iterator.
    """

    def __init__(
        self,
    ):
        self.test_suite = {}

    def add_test_object(self, base_test_case, causal_test_case_list, estimators, estimate_type: str = "ate"):
        test_object = {"tests": causal_test_case_list, "estimators": estimators, "estimate_type": estimate_type}
        self.test_suite[base_test_case] = test_object

    def get_single_test_object(self, base_test_case):
        return self.test_suite[base_test_case]
