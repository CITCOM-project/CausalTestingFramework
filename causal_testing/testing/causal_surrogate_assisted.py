from causal_testing.data_collection.data_collector import DataCollector
from causal_testing.testing.causal_test_suite import CausalTestSuite


class CausalSurrogateAssistedTestCase:
    
    def __init__(
            self, 
            test_suite: CausalTestSuite,
            # search_alogrithm: SearchAlgorithm,
            # simulator: Simulator,
        ):
        self.test_suite = test_suite

    def execute(self, data_collector: DataCollector, max_executions: int = 200):
        df = data_collector.collect_data()

        for _i in range(max_executions):

            # Build surrogate models based on df

            # Define surrogate model fitness function
            self.test_suite.execute_test_suite(df)

            # Multiobjective Metaheuristics to find candidate test case

            # Run candidate test case gainst simulator

            # Validate fault

            # If not valid, add to df and loop

            pass