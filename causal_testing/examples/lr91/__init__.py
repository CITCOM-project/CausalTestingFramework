"""
Set up the Luo-Rudy causal testing objects for easy import.
"""

from importlib import resources

import pandas as pd

from causal_testing.specification.causal_dag import CausalDAG

ROOT = "causal_testing.examples.lr91"

data_path = resources.files(ROOT).joinpath("normalised_results.csv")
dag_path = resources.files(ROOT).joinpath("dag.dot")

dag = CausalDAG(dag_path)
df = pd.read_csv(data_path)
