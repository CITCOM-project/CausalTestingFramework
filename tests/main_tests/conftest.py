"""
Shared pytest fixtures for the main tests.
"""

import pytest


@pytest.fixture(name="dag_path")
def _dag_path(resources_dir):
    """
    Path to the example causal DAG.
    """
    return str(resources_dir / "dag.dot")


@pytest.fixture(name="data_path")
def _data_path(resources_dir):
    """
    Path to the example data.
    """
    return str(resources_dir / "data.csv")


@pytest.fixture(name="test_cases_path")
def _test_cases_path(resources_dir):
    """
    Path to the example causal test case configurations.
    """
    return str(resources_dir / "tests.json")


@pytest.fixture(name="include_edges_path")
def _include_edges_path(resources_dir):
    """
    Path to the edges that causal discovery must include.
    """
    return str(resources_dir / "include_edges.dot")


@pytest.fixture(name="exclude_edges_path")
def _exclude_edges_path(resources_dir):
    """
    Path to the edges that causal discovery must exclude.
    """
    return str(resources_dir / "exclude_edges.dot")
