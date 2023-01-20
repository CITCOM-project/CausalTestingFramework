Installation
============

Requirements
------------
CausalTestingFramework requires python version 3.9 or later
If installing on Windows, ensure `Microsoft Visual C++ <https://docs.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist>`_ is version 14.0 or greater

Pygraphviz
----------

Pygraphviz can be installed through the conda-forge channel

.. code-block:: console

    conda install -c conda-forge pygraphviz


Alternatively, on Linux systems, this can be done with `sudo apt install graphviz libgraphviz-dev`.

Install from source
-------------------

In future it will be possible to install from PyPI, but for now...

.. code-block:: console

    git clone https://github.com/CITCOM-project/CausalTestingFramework
    cd CausalTestingFramework

then, to install a specific release:

.. code-block:: console    
    git fetch --all --tags --prune
    git checkout tags/<tag> -b <branch>
    pip install -e .

e.g. version `1.0.0`

.. code-block:: console    
    git fetch --all --tags --prune
    git checkout tags/1.0.0 -b version
    pip install -e .

or to install the latest development version:

.. code-block:: console    
    pip install -e .

Use 

.. code-block:: console

    pip install -e .[dev]

to also install developer tools.