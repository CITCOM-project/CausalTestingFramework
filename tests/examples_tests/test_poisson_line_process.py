import networkx as nx
import pandas as pd

from causal_testing.causal_testing_framework import CausalTestingFramework
from causal_testing.examples import poisson_line_process as plp
from causal_testing.specification.causal_dag import CausalDAG


def test_load_data():
    """
    Test that the pre-loaded data is equivalent to loading the data path.
    """
    pd.testing.assert_frame_equal(plp.df, pd.read_csv(plp.data_path))


def test_load_dag():
    """
    Test that the pre-loaded dag is equivalent to loading the dag path.
    """
    assert nx.utils.misc.graphs_equal(plp.dag, CausalDAG(plp.dag_path))


def test_load_tests():
    """
    Test that loading the JSON serialised tests is equivalent to generating them from scratch.
    """
    ctf_load = CausalTestingFramework(dag=plp.dag, df=plp.df)
    ctf_load.load_test_cases_from_json(plp.tests_path)

    ctf_generate = CausalTestingFramework(dag=plp.dag, df=plp.df)
    ctf_generate.generate_causal_tests()

    pd.testing.assert_frame_equal(ctf_generate.test_dataframe(), ctf_load.test_dataframe())


def test_load_completed_tests():
    """
    Test that loading the JSON serialised tests is equivalent to generating them from scratch.
    """
    ctf_load = CausalTestingFramework(dag=plp.dag, df=plp.df)
    ctf_load.load_test_cases_from_json(plp.completed_tests_path)

    ctf_generate = CausalTestingFramework(dag=plp.dag, df=plp.df)
    ctf_generate.generate_causal_tests()
    ctf_generate.run_tests(adequacy=True)

    pd.testing.assert_frame_equal(ctf_generate.test_dataframe(), ctf_load.test_dataframe())
