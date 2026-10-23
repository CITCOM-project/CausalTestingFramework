"""
Set up the Covasim doubling beta causal testing objects for easy import.
"""

from importlib import resources

import pandas as pd

from causal_testing.specification.causal_dag import CausalDAG

ROOT = "causal_testing.examples.covasim.doubling_beta"

data_path = resources.files(ROOT).joinpath("10k_observational_data.csv")
dag_path = resources.files(ROOT).joinpath("dag.dot")

dag = CausalDAG(dag_path)
df = pd.read_csv(data_path)
