"""
Shared pytest fixtures for tests.
"""

from pathlib import Path

import pandas as pd
import pytest

RESOURCES_DIR = Path(__file__).parent / "resources"


@pytest.fixture(name="resources_dir")
def _resources_dir() -> Path:
    """
    Directory containing data files used by the tests.
    """
    return RESOURCES_DIR


@pytest.fixture(name="scarf_df")
def _scarf_df(resources_dir):
    """
    Fresh copy of the scarf dataset for each test.
    """
    return pd.read_csv(resources_dir / "scarf_data.csv")
