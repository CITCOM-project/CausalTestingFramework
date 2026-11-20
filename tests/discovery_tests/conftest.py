"""
Shared pytest fixtures for the causal discovery tests.
"""

import pandas as pd
import pytest


@pytest.fixture(name="scarf_df")
def _scarf_df():
    """
    Fresh copy of the scarf dataset for each test.
    """
    return pd.read_csv("tests/resources/data/scarf_data.csv")
