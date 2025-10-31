Getting Started
================

Installation
-----------------
* We currently support Python versions 3.10, 3.11, 3.12, and 3.13.

* The Causal Testing Framework can be installed through either the `Python Package Index (PyPI)`_ (recommended), or directly from source (recommended for contributors).

.. _Python Package Index (PyPI): https://dl.acm.org/doi/10.1145/3607184

.. note::
   We recommend you use a 64-bit OS (standard in most modern machines) as we have had reports of the installation crashing on legacy 32-bit Debian systems.

Method 1: Installing via pip
..............................

To install the Causal Testing Framework using :code:`pip` for the latest stable version::

    pip install causal-testing-framework

This provides all core functionality needed to perform causal testing, including support for causal DAGs, various estimation methods, and test case generation.

If you also want to install the framework with (optional) development packages/tools::

    pip install causal-testing-framework[dev]


Method 2: Installing via Source
...............................

To install from source::

    git clone https://github.com/CITCOM-project/CausalTestingFramework
    cd CausalTestingFramework

then, to install a specific release::

    git fetch --all --tags --prune
    git checkout tags/<tag> -b <branch>
    pip install . # For core API only
    pip install -e . # For editable install, useful for development work

e.g. version `1.0.0`::

    git fetch --all --tags --prune
    git checkout tags/1.0.0 -b version
    pip install .

or to install the latest development version::

    pip install .

To also install developer tools::

    pip install -e .[dev]

Verifying Your Installation
-----------------------------

After installation, verify that the framework is installed correctly in your environment::

    python -c "import causal_testing; print(causal_testing.__version__)"

Next Steps
-----------

* Check out the :doc:`tutorials` to learn how to use the framework.
* Read about :doc:`modules/causal_specification` to understand causal specifications and :doc:`modules/causal_testing` for the end-to-end causal testing process.
* Try the command-line interface for quick and simple testing::

    python -m causal_testing test --help
