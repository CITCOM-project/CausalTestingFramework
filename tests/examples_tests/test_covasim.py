import networkx as nx
import pandas as pd

from causal_testing.examples.covasim import disease_variants, vaccinating_elderly
from causal_testing.specification.causal_dag import CausalDAG


def test_disease_variants_load_data():
    """
    Test that the pre-loaded data is equivalent to loading the data path.
    """
    pd.testing.assert_frame_equal(disease_variants.df, pd.read_csv(disease_variants.data_path))


def test_disease_variants_load_dag():
    """
    Test that the pre-loaded dag is equivalent to loading the dag path.
    """
    assert nx.utils.misc.graphs_equal(disease_variants.dag, CausalDAG(disease_variants.dag_path))


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
