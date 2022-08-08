Installation
============

Requirements
------------
CausalTestingFramework requires python version 3.9 or later
If installing on Windows, ensure `Microsoft Visual C++ <https://docs.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist>`_ is version 14.0 or greater

Pygraphviz
----------

Pygraphviz requires installation through the conda-forge channel

.. code-block:: console

    conda install -c conda-forge pygraphviz


Install from source
-------------------
.. code-block:: console

    git clone https://github.com/CITCOM-project/CausalTestingFramework
    cd CausalTestingFramework
    pip install -e .

Use 

.. code-block:: console

    pip install -e .[dev]

to also install developer tools.