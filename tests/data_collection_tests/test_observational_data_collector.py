import unittest
import os
import shutil, tempfile
import pandas as pd
from causal_testing.data_collection.data_collector import ObservationalDataCollector
from causal_testing.specification.causal_specification import Scenario
from causal_testing.specification.variable import Input, Output, Meta
from scipy.stats import uniform, rv_discrete
from enum import Enum
import random


class TestObservationalDataCollector(unittest.TestCase):
    def setUp(self) -> None:
        class Color(Enum):
            RED = "RED"
            GREEN = "GREEN"
            BLUE = "BLUE"

        self.temp_dir_path = tempfile.mkdtemp()
        self.dag_dot_path = os.path.join(self.temp_dir_path, "dag.dot")
        self.observational_df_path = os.path.join(self.temp_dir_path, "observational_data.csv")
        # Y = 3*X1 + X2*X3 + 10
        self.observational_df = pd.DataFrame(
            {"X1": [1, 2, 3, 4], "X2": [5, 6, 7, 8], "X3": [10, 20, 30, 40], "Y2": ["RED", "GREEN", "BLUE", "BLUE"]}
        )
        self.observational_df["Y1"] = self.observational_df.apply(
            lambda row: (3 * row.X1) + (row.X2 * row.X3) + 10, axis=1
        )
        self.observational_df.to_csv(self.observational_df_path)
        self.observational_df["Y2"] = [Color[x] for x in self.observational_df["Y2"]]
        self.X1 = Input("X1", int, uniform(1, 4))
        self.X2 = Input("X2", int, rv_discrete(values=([7], [1])))
        self.X3 = Input("X3", int, uniform(10, 40))
        self.X4 = Input("X4", int, rv_discrete(values=([10], [1])))
        self.Y1 = Output("Y1", int)
        self.Y2 = Output("Y2", Color)

    def test_not_all_variables_in_data(self):
        scenario = Scenario({self.X1, self.X2, self.X3, self.X4})
        observational_data_collector = ObservationalDataCollector(scenario, self.observational_df)
        self.assertRaises(IndexError, observational_data_collector.collect_data)

    def test_all_variables_in_data(self):
        scenario = Scenario({self.X1, self.X2, self.X3, self.Y1, self.Y2})
        observational_data_collector = ObservationalDataCollector(scenario, self.observational_df)
        df = observational_data_collector.collect_data(index_col=0)
        assert df.equals(self.observational_df), f"\n{df}\nwas not equal to\n{self.observational_df}"

    def test_data_constraints(self):
        scenario = Scenario({self.X1, self.X2, self.X3, self.Y1, self.Y2}, {self.X1.z3 > 2})
        observational_data_collector = ObservationalDataCollector(scenario, self.observational_df)
        df = observational_data_collector.collect_data(index_col=0)
        expected = self.observational_df.loc[[2, 3]]
        assert df.equals(expected), f"\n{df}\nwas not equal to\n{expected}"

    def test_meta_population(self):
        def populate_m(data):
            data["M"] = data["X1"] * 2

        meta = Meta("M", int, populate_m)
        scenario = Scenario({self.X1, meta})
        observational_data_collector = ObservationalDataCollector(scenario, self.observational_df)
        observational_data_collector.collect_data()
        data = observational_data_collector.collect_data()
        assert all((m == 2 * x1 for x1, m in zip(data["X1"], data["M"])))

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir_path)


if __name__ == "__main__":
    unittest.main()
