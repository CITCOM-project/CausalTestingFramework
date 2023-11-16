# Causal Testing Framework
### A Causal Inference-Driven Software Testing Framework


[![Project Status: Active – The project has reached a stable, usable state and is being actively developed.](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active) 
![example workflow](https://github.com/CITCOM-project/CausalTestingFramework/actions/workflows/ci-tests.yaml/badge.svg) 
[![codecov](https://codecov.io/gh/CITCOM-project/CausalTestingFramework/branch/main/graph/badge.svg?token=04ijFVrb4a)](https://codecov.io/gh/CITCOM-project/CausalTestingFramework) 
[![Documentation Status](https://readthedocs.org/projects/causal-testing-framework/badge/?version=latest)](https://causal-testing-framework.readthedocs.io/en/latest/?badge=latest)
![Dynamic TOML Badge](https://img.shields.io/badge/dynamic/toml?url=https%3A%2F%2Fraw.githubusercontent.com%2FCITCOM-project%2FCausalTestingFramework%2Fmain%2Fpyproject.toml&query=%24.project%5B'requires-python'%5D&label=python)
![PyPI - Version](https://img.shields.io/pypi/v/causal-testing-framework)
[![DOI](https://t.ly/FCT1B)](https://orda.shef.ac.uk/articles/software/CITCOM_Software_Release/24427516)
![GitHub License](https://img.shields.io/github/license/CITCOM-project/CausalTestingFramework)

Causal testing is a causal inference-driven framework for functional black-box testing. This framework utilises
graphical causal inference (CI) techniques for the specification and functional testing of software from a black-box
perspective. In this framework, we use causal directed acyclic graphs (DAGs) to express the anticipated cause-effect
relationships amongst the inputs and outputs of the system-under-test and the supporting mathematical framework to
design statistical procedures capable of making causal inferences. Each causal test case focuses on the causal effect of
an intervention made to the system-under test. That is, a prescribed change to the input configuration of the
system-under-test that is expected to cause a change to some output(s).

![Causal Testing Workflow](images/workflow.png)

## Installation

See the Read the Docs site for [installation
instructions](https://causal-testing-framework.readthedocs.io/en/latest/installation.html).

## Documentation

Further information on causal inference, the code, usage and more can be found on the [docs](https://causal-testing-framework.readthedocs.io/en/latest/)
