"""A library of helper methods for the causal testing framework tests."""
import sys
from os import mkdir
from os.path import join, dirname, realpath, exists
from shutil import rmtree


def create_temp_dir_if_non_existent():
    """Create a temporary directory in the current working directory if one does not exist already.

    Create a temporary directory named temp in the current working directory provided it does not already exist, and
    then return the path to the temporary directory (regardless of whether it existed previously or has just been
    created).

    :return: Path to the temporary directory.
    """
    temp_dir = join(dirname(realpath(sys.argv[0])), "temp/")
    if not exists(temp_dir):
        mkdir(temp_dir)
    return temp_dir


def remove_temp_dir_if_existent():
    """Remove a temporary directory from the current working directory if one exists."""
    temp_dir = join(dirname(realpath(sys.argv[0])), "temp/")
    if exists(temp_dir):
        rmtree(temp_dir)
