Welcome to the Causal Testing Framework
==========================================

|status| |ci-tests| |code-cov| |docs| |python| |pypi| |doi| |license|

Motivation
----------

A common problem in computer science is to develop robust and reliable software systems that can perform correctly under various input configurations and maintain consistency across complex, physical scenarios. However, software systems, and more specifically computational models, can be difficult to test: they may contain hundreds of parameters, making testing all possible inputs computationally infeasible; some models may be inherently non-deterministic, producing different outputs for the same inputs due to randomness; or there may exist hidden causal relationships between input-output pairs, causing errors that only appear under specific combinations of input configurations.

The Framework
-------------

The Causal Testing Framework is composed of a :term:`causal inference`-driven architecture designed for functional black-box testing.
It leverages graphical causal inference (CI) techniques to specify and evaluate software behaviour from a black-box perspective.
Within this framework, causal directed acyclic graphs (DAGs) are used to represent the expected cause–effect relationships between
the inputs and outputs of the system under test, supported by mathematical foundations for designing statistical procedures that
enable causal inference. Each causal test case targets the causal effect of a specific intervention on the system under test--that is,
a deliberate modification to the input configuration expected to produce a corresponding change in one or more outputs.

.. toctree::
   :hidden:
   :caption: Home
.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Introduction

   background
   installation
   tutorials


.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Module Descriptions

   /modules/causal_specification
   /modules/estimators
   /modules/causal_testing

.. toctree::
   :maxdepth: 2
   :caption: API
   :hidden:
   :titlesonly:

   /autoapi/index

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
   Figshare <https://orda.shef.ac.uk/articles/software/CITCOM_Software_Release/24427516>
   PyPI <https://pypi.org/project/causal-testing-framework/>

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Credits

   credits

.. Define variables for our GH badges

.. |ci-tests| image:: https://github.com/CITCOM-project/CausalTestingFramework/actions/workflows/ci-tests.yaml/badge.svg
   :target: https://github.com/CITCOM-project/CausalTestingFramework/actions/workflows/ci-tests.yaml
   :alt: Continuous Integration Tests

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

.. |license| image:: https://img.shields.io/github/license/CITCOM-project/CausalTestingFramework
   :target: https://github.com/CITCOM-project/CausalTestingFramework
   :alt: License
