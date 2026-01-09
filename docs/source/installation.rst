Getting Started
================

Installation
-----------------
* We currently support Python versions 3.10, 3.11, 3.12, and 3.13.

* The Causal Testing Framework can be installed through `conda-forge`_ (recommended), the `Python Package Index (PyPI)`_, or directly from source (recommended for contributors).

.. _conda-forge: https://anaconda.org/conda-forge/causal-testing-framework
.. _Python Package Index (PyPI): https://pypi.org/project/causal-testing-framework/

.. note::
   We recommend you use a 64-bit OS (standard in most modern machines) as we have had reports of the installation crashing on legacy 32-bit Debian systems.

Method 1: Installing via conda-forge (Recommended)
...................................................

**We recommend using conda or mamba for installation**, as they provide better dependency management and environment isolation, particularly for scientific computing workflows.

First, create a new conda environment with a supported Python version, e.g::

    conda create -n causal-testing-env python=3.13
    conda activate causal-testing-env

.. note::
   If you have `Miniforge <https://conda-forge.org/download/>`_ installed, you can replace :code:`conda` with :code:`mamba` in any of the commands below for faster package resolution.

Add the :code:`conda-forge` channel::

    conda config --add channels conda-forge
    conda config --set channel_priority strict

Install :code:`causal-testing-framework`::

    conda install causal-testing-framework


Method 2: Installing via pip
..............................

If you prefer using pip or need the development packages, you can install from PyPI.

To install the Causal Testing Framework using :code:`pip` for the latest stable version::

    pip install causal-testing-framework

This provides all core functionality needed to perform causal testing, including support for causal DAGs, various estimation methods, and test case generation.

If you also want to install the framework with (optional) development packages/tools::

    pip install causal-testing-framework[dev]


Method 3: Installing via Source (For Developers/Contributors)
...............................................................

If you're planning to contribute to the project or need an editable installation for development, you can install directly from source::

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
* Run the command for guidance on how to generate your causal tests directly from your input DAG::

    python -m causal_testing generate --help

* and the command on guidance on how to execute your causal tests::

    python -m causal_testing test --help


Using the CTF on DAFNI
======================

The Causal Testing Framework is also available to run on `DAFNI <https://www.dafni.ac.uk/>`_, allowing you to generate causal tests and evaluate causal effects from your input data and DAGs without installing the framework locally. This lets you integrate CTF into workflows with other models or datasets easily.

Data tab
--------

- Upload the required input files as a dataset. Typically, this includes:

  - ``dag.dot`` – the directed acyclic graph defining causal relationships between variables.
  - ``runtime_data.csv`` – the CSV file containing runtime input data.
  - ``causal_tests.json`` – optional; if provided, the framework will run tests directly. Otherwise, tests will be generated automatically.

  **Note:** Input files must remain in the ``data/inputs`` structure; this is required by the workflow.

Workflow tab
-------------

- Select the CTF workflow.
- In the Parameter sets section, click **Create**.
- In the page that opens:

  - Select the model in the workflow (typically ``causal-testing-framework``).
  - Complete the sections at the bottom:

    - **Parameters:** Set or confirm environment variables from the ``.env`` file (e.g., ``EXECUTION_MODE``, ``CAUSAL_TESTS``, ``CAUSAL_TEST_RESULTS``). These control whether tests are generated or executed, the filenames, estimator, effect type, and other runtime options.
    - **Datasets:** Click the icon and select the dataset containing your input files (``dag.dot``, ``runtime_data.csv``, ``causal_tests.json``). All input files will be placed in the required ``data/inputs`` directory when running the workflow.

- Unselect the model if needed, click **Continue**, and complete any required metadata such as the name of the parameter set.

Execute the workflow
----------------------

- Click **Execute workflow with parameter set**.
- If successful, the workflow will either generate ``causal_tests.json`` (if not provided) or run the tests and create ``causal_test_results.json`` in ``data/outputs``.
- After completion, you can view the results in the **Data tab** as a new output dataset.

Customisation and chaining
--------------------------

- You can create additional workflows to customise input parameters, filenames, or estimators.
- Multiple CTF workflows can also be chained to run sequential analyses or to integrate with other models and datasets, combining results for more complex causal testing scenarios.
