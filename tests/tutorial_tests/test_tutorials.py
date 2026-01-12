import os

os.environ["JUPYTER_PLATFORM_DIRS"] = "1"

import sys
import warnings
import asyncio

warnings.filterwarnings("ignore", category=DeprecationWarning, message=r"Jupyter is migrating.*")

if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import pathlib
import pytest
import nbformat
from nbclient.client import NotebookClient

NOTEBOOK_DIR = pathlib.Path(__file__).parent.parent.parent / "docs" / "source" / "tutorials"
NOTEBOOK_FILES = list(NOTEBOOK_DIR.rglob("[!.]*/*.ipynb"))


@pytest.mark.parametrize("notebook_path", NOTEBOOK_FILES, ids=lambda p: p.name)
def test_tutorial_notebooks(notebook_path):
    # Change directory to the location of the notebook since there's no way to get the abspath from the notebook itself
    os.chdir(os.path.dirname(notebook_path))
    client = NotebookClient(nbformat.read(notebook_path, as_version=4))
    client.execute()
