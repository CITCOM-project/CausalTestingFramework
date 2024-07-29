Getting started
================

Requirements
---------------
* Python 3.9, 3.10, 3.11 and 3.12
* `Microsoft Visual C++ <https://docs.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist>`_ 14.0+ (Windows only).


Installation
-----------------
The Causal Testing Framework can be installed through either the `Python Package Index (PyPI)`_ (recommended), or directly from source.

.. _Python Package Index (PyPI): https://dl.acm.org/doi/10.1145/3607184

Method 1: Installing via pip
..............................

To install the Causal Testing Framework using :code:`pip` for the latest stable version::

    pip install causal-testing-framework

or if you want to install with development packages/tools::

    pip install causal-testing-framework[dev]


Method 2: Installing via source
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

