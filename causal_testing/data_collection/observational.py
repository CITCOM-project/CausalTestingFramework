import pandas as pd
from causal_testing.data_collection.data_collector import DataCollector
from causal_testing.specification.causal_specification import Scenario


class Observational(DataCollector):

    def __init__(self, scenario):
        super().__init__()
        self.scenario = scenario

    def collect_data(self, csv_path):
        self.df = pd.read_csv(csv_path)
        self.filter_valid_data()

    def filter_valid_data(self):
        variables = set(self.scenario.keys())
        if not variables.issubset(self.df.columns):
            missing_variables = variables - set(self.df.columns)
            raise IndexError(f'Positivity violation: missing data for variables {missing_variables}.')