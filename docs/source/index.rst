Welcome to the Causal Testing Framework
==========================================

|status| |ci-tests| |code-cov| |docs| |deep-wiki| |python| |pypi| |conda-forge| |license| |joss| |doi|


Motivation
----------

From predicting the weather to simulating disease transmission, scientific software plays an increasingly pivotal role in developing scientific understanding that informs our everyday lives.
However, they are also some of the most difficult software systems to properly test.
They have large, complex input spaces, are computationally expensive to run, often rely on stochastic black-box components, and are applied in exploratory contexts where the expected outcomes are not known.
From a practical standpoint, the time and effort that can be dedicated to testing is often limited, especially in an academic context, making it especially important to maximise the efficiency of the limited number of test runs we are able to perform.

The Causal Testing Framework has two main workflows:

  - **Causal Testing** involves specifying the expected causal relationships and testing that the data conforms to this.
  - **Causal Discovery** involves infering the causal effects from the data and checking that the model is reasonable.

Causal Testing
--------------

 .. figure:: _static/images/testing-workflow.png
    :alt: Schematic diagram of the Causal Testing Workflow.
    :align: center

    **Figure:** Schematic diagram of the Causal Testing Workflow.

The Causal Testing Framework uses graphical :term:`causal inference` to specify and validate software behaviour by estimating the causal effects between variables.
This requires three main components:

#.
   :doc:`Causal Graph <../modules/causal_dag>`\ : This specifies the expected causal relationships between the variables in the form of a directed acyclic graph (DAG).
   The nodes in your DAG represent variables in your system, and edges between the variables represent the "flow of causality" such that an edge from X to Y represents the value of Y being caused (i.e. directly affected) by the value of X.

#.
  :doc:`Test Data <../modules/test_data>`\ : This is the data that will be used to estimate the causal effects between variables and evaluate your causal test cases.
  This takes the form of a table in which columns represent the variables in your DAG and each row represents a run of the system.

#.
   :doc:`Causal Tests <../modules/causal_tests>`\ : Each causal test case validates that the causal effect between the *treatment* and *outcome* variable that can be estimated from the :doc:`test data <../modules/test_data>` is as expected.
   The most basic causal test case simply validates the presence or absence of a causal effect.
   The Causal Testing Framework can automatically generate a suite of such tests from the causal DAG alone.
   You can then customise and refine these tests to suite your needs.

An example of this workflow can be seen in our :doc:`tutorials <tutorials/vaccinating_elderly/vaccinating_elderly_tutorial>`\.

Causal Discovery
----------------

.. figure:: _static/images/discovery-workflow.png
   :alt: Schematic diagram of the Causal Discovery Workflow.
   :align: center

   **Figure:** Schematic diagram of the Causal Discovery Workflow.

An alternative, more analytical approach involves using the Causal Testing Framework to automatically "discover" the causal relationships between variables from execution data.
This approach is well suited to users who do not have a concrete notion of the expected causal relationships between variables upfront, or are looking to gain an understanding of how an unfamiliar system works,
Note that we do not recommend using discovered DAGs for Causal Testing without careful manual inspection, since such graphs will trivially lead to passing test cases.


.. toctree::
   :maxdepth: 1

   installation

.. toctree::
  :hidden:
  :maxdepth: 1
  :caption: Quick Start

  quick_start/causal_testing
  quick_start/causal_discovery

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Module Descriptions

   /modules/causal_dag
   /modules/test_data
   /modules/causal_tests
   /modules/causal_inference
   /modules/causal_estimate
   /modules/test_oracle
   /modules/causal_discovery

.. toctree::
  :hidden:
  :maxdepth: 1
  :caption: Tutorials

  tutorials/vaccinating_elderly/vaccinating_elderly_tutorial
  tutorials/poisson_line_process/poisson_line_process_tutorial
  tutorials/visualising_causal_test_results/visualise_causal_test_results



.. toctree::
   :maxdepth: 2
   :caption: API
   :hidden:
   :titlesonly:

   /autoapi/index

.. toctree::
  :maxdepth: 1
  :caption: DAFNI integration

  dafni

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Glossary

   glossary

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Development

   /dev/version_release
   /dev/documentation
   /dev/actions_and_webhooks

.. toctree::
   :caption: Useful Links
   :hidden:
   :maxdepth: 2

   CITCoM Homepage <https://sites.google.com/sheffield.ac.uk/citcom/home>
   Paper <https://dl.acm.org/doi/10.1145/3607184>
   PyPI <https://pypi.org/project/causal-testing-framework/>
   Conda-forge <https://anaconda.org/channels/conda-forge/packages/causal-testing-framework/overview>
   Figshare <https://orda.shef.ac.uk/articles/software/CITCOM_Software_Release/24427516>
   DeepWiki <https://deepwiki.com/CITCOM-project/CausalTestingFramework>

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Credits

   credits

.. Define variables for our GH badges

.. |ci-tests| image:: https://github.com/CITCOM-project/CausalTestingFramework/actions/workflows/ci-tests.yaml/badge.svg
   :target: https://github.com/CITCOM-project/CausalTestingFramework/actions/workflows/ci-tests.yaml
   :alt: Continuous Integration Tests

.. |conda-forge| image:: https://img.shields.io/conda/v/conda-forge/causal-testing-framework.svg
   :target: https://anaconda.org/conda-forge/causal-testing-framework
   :alt: Conda Forge

.. |code-cov| image:: https://codecov.io/gh/CITCOM-project/CausalTestingFramework/branch/main/graph/badge.svg?token=04ijFVrb4a
   :target: https://codecov.io/gh/CITCOM-project/CausalTestingFramework
   :alt: Code coverage

.. |docs| image:: https://readthedocs.org/projects/causal-testing-framework/badge/?version=latest
   :target: https://causal-testing-framework.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation

.. |python| image:: https://img.shields.io/badge/dynamic/toml?url=https%3A%2F%2Fraw.githubusercontent.com%2FCITCOM-project%2FCausalTestingFramework%2Fmain%2Fpyproject.toml&query=%24.project%5B'requires-python'%5D&label=python
   :target: https://img.shields.io/badge/dynamic/toml?url=https%3A%2F%2Fraw.githubusercontent.com%2FCITCOM-project%2FCausalTestingFramework%2Fmain%2Fpyproject.toml&query=%24.project%5B'requires-python'%5D&label=python
   :alt: Python

.. |pypi| image:: https://img.shields.io/pypi/v/causal-testing-framework
   :target: https://pypi.org/project/causal-testing-framework/
   :alt: PyPI

.. |status| image:: https://www.repostatus.org/badges/latest/active.svg
   :target: https://www.repostatus.org/#active
   :alt: Status

.. |doi| image:: https://t.ly/FCT1B
   :target: https://orda.shef.ac.uk/articles/software/CITCOM_Software_Release/24427516
   :alt: DOI

.. |joss| image:: https://joss.theoj.org/papers/10.21105/joss.07739/status.svg
   :target: https://joss.theoj.org/papers/10.21105/joss.07739
   :alt: JOSS


.. |license| image:: https://img.shields.io/github/license/CITCOM-project/CausalTestingFramework
   :target: https://github.com/CITCOM-project/CausalTestingFramework
   :alt: License

.. |deep-wiki| image:: https://deepwiki.com/badge.svg
   :target: https://deepwiki.com/CITCOM-project/CausalTestingFramework
   :alt: DeepWiki
