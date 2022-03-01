# Sphinx Documentation
This project uses the [Sphinx](https://www.sphinx-doc.org/en/master/) documentation generator with the [Sphinx docstring](https://sphinx-rtd-tutorial.readthedocs.io/en/latest/docstrings.html) format to produce documentation.

## Read the Docs
The documentation is accessible on [Read the Docs](https://causal-testing-framework.readthedocs.io/en/latest/). `.readthedocs.yaml` in the root project directory contains the configuration for the build environment within Read the Docs.

## Locally building
To build locally, the requirements in `docs/source/requirements.txt` will need to be installed.

Within `docs/`, run `make html` to create or update the .html files in the `docs/build` directory. 

Running `make clean` will clean the `build` folder.  