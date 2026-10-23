import networkx as nx
import pandas as pd

from causal_testing.examples.covasim import doubling_beta, vaccinating_elderly
from causal_testing.specification.causal_dag import CausalDAG


def test_doubling_beta_load_data():
    """
    Test that the pre-loaded data is equivalent to loading the data path.
    """
    pd.testing.assert_frame_equal(doubling_beta.df, pd.read_csv(doubling_beta.data_path))


def test_doubling_beta_load_dag():
    """
    Test that the pre-loaded dag is equivalent to loading the dag path.
    """
    assert nx.utils.misc.graphs_equal(doubling_beta.dag, CausalDAG(doubling_beta.dag_path))


def test_vaccinating_elderly_load_data():
    """
    Test that the pre-loaded data is equivalent to loading the data path.
    """
    pd.testing.assert_frame_equal(vaccinating_elderly.df, pd.read_csv(vaccinating_elderly.data_path))


def test_vaccinating_elderly_load_dag():
    """
    Test that the pre-loaded dag is equivalent to loading the dag path.
    """
    assert nx.utils.misc.graphs_equal(vaccinating_elderly.dag, CausalDAG(vaccinating_elderly.dag_path))
