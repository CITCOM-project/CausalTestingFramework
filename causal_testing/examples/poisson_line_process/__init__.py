"""
Set up the Poisson line process causal testing objects for easy import.
"""

from importlib import resources

import pandas as pd

from causal_testing.specification.causal_dag import CausalDAG

ROOT = "causal_testing.examples.poisson_line_process"

data_path = resources.files(ROOT).joinpath("data_random_1000.csv")
dag_path = resources.files(ROOT).joinpath("dag.dot")
tests_path = resources.files(ROOT).joinpath("causal_tests.json")
completed_tests_path = resources.files(ROOT).joinpath("causal_test_results.json")

dag = CausalDAG(dag_path)
df = pd.read_csv(data_path)
