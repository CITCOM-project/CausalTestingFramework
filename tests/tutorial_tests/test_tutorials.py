"""Tests that the tutorial notebooks in the documentation run without errors."""

import asyncio
import os
import pathlib
import sys
import warnings

import pytest

# The Jupyter environment must be configured before any Jupyter modules are imported
os.environ["JUPYTER_PLATFORM_DIRS"] = "1"

warnings.filterwarnings("ignore", category=DeprecationWarning, message=r"Jupyter is migrating.*")

if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# pylint: disable=wrong-import-position
import nbformat
from nbclient.client import NotebookClient

# pylint: enable=wrong-import-position

NOTEBOOK_DIR = pathlib.Path(__file__).parent.parent.parent / "docs" / "source" / "tutorials"
NOTEBOOK_FILES = sorted(list(NOTEBOOK_DIR.rglob("[!.]*/*.ipynb")))


@pytest.mark.parametrize("notebook_path", NOTEBOOK_FILES, ids=lambda p: p.name)
def test_tutorial_notebooks(notebook_path, monkeypatch):
    """Test that each tutorial notebook runs without errors."""
    # Change directory to the location of the notebook since there's no way to get the abspath from the notebook itself
    # monkeypatch restores the original working directory after the test
    monkeypatch.chdir(notebook_path.parent)
    client = NotebookClient(nbformat.read(notebook_path, as_version=4))
    client.execute()
