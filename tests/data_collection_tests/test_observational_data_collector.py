import unittest
import os
import pandas as pd
from causal_testing.data_collection.data_collector import ObservationalDataCollector
from causal_testing.specification.causal_specification import Scenario

from causal_testing.specification.variable import Input
from scipy.stats import uniform, rv_discrete


class TestObservationalDataCollector(unittest.TestCase):
    def setUp(self) -> None:
        self.observational_df_dir = "temp"
        if not os.path.exists(self.observational_df_dir):
            os.makedirs(self.observational_df_dir)
        self.observational_df_path = (
            f"{self.observational_df_dir}/observational_data.csv"
        )
        # Y = 3*X1 + X2*X3 + 10
        observational_df = pd.DataFrame(
            {"X1": [1, 2, 3, 4], "X2": [5, 6, 7, 8], "X3": [10, 20, 30, 40]}
        )
        observational_df["Y"] = observational_df.apply(
            lambda row: (3 * row.X1) + (row.X2 * row.X3) + 10, axis=1
        )
        observational_df.to_csv(self.observational_df_path)

    X1 = Input("X1", int, uniform(1, 4))
    X2 = Input("X2", int, rv_discrete(values=([7], [1])))
    X3 = Input("X3", int, uniform(10, 40))
    X4 = Input("X4", int, rv_discrete(values=([10], [1])))

    def test_all_variables_in_data(self):
        scenario = Scenario({"X1": self.X1, "X2": self.X2, "X3": self.X3})
        observational_data_collector = ObservationalDataCollector(scenario)
        df = observational_data_collector.collect_data(self.observational_df_path)
        assert not df.empty

    def test_not_all_variables_in_data(self):
        scenario = Scenario(
            {"X1": self.X1, "X2": self.X2, "X3": self.X3, "X4": self.X4}
        )
        observational_data_collector = ObservationalDataCollector(scenario)
        self.assertRaises(
            IndexError,
            observational_data_collector.collect_data,
            self.observational_df_path,
        )

    def tearDown(self) -> None:
        os.remove(self.observational_df_path)


if __name__ == "__main__":
    unittest.main()
