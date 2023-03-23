Installation
============

Requirements
------------
CausalTestingFramework requires python version 3.9 or later
If installing on Windows, ensure `Microsoft Visual C++ <https://docs.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist>`_ is version 14.0 or greater

Pygraphviz
----------

Pygraphviz can be installed through the conda-forge channel::

    conda install -c conda-forge pygraphviz


Alternatively, on Linux systems, this can be done with `sudo apt install graphviz libgraphviz-dev`.

Pip Install
-----------
To install the Causal Testing Framework using :code:`pip` for the latest stable version::

    pip install causal-testing-framework

To install with development packages/tools::

    pip install causal-testing-framework[dev]

Install from source
-------------------

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

