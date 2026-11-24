"""
Shared pytest fixtures for the main tests.
"""

import pytest


@pytest.fixture(name="dag_path")
def _dag_path():
    """
    Path to the example causal DAG.
    """
    return "tests/resources/data/dag.dot"


@pytest.fixture(name="data_path")
def _data_path():
    """
    Path to the example data.
    """
    return "tests/resources/data/data.csv"


@pytest.fixture(name="test_cases_path")
def _test_cases_path():
    """
    Path to the example causal test case configurations.
    """
    return "tests/resources/data/tests.json"
