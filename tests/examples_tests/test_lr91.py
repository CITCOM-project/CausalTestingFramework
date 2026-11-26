import networkx as nx
import pandas as pd

from causal_testing.examples import lr91
from causal_testing.specification.causal_dag import CausalDAG


def test_load_data():
    """
    Test that the pre-loaded data is equivalent to loading the data path.
    """
    pd.testing.assert_frame_equal(lr91.df, pd.read_csv(lr91.data_path))


def test_load_dag():
    """
    Test that the pre-loaded dag is equivalent to loading the dag path.
    """
    assert nx.utils.misc.graphs_equal(lr91.dag, CausalDAG(lr91.dag_path))
