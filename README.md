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

### Requirements
- Python >= 3.9.

- Microsoft Visual C++ 14.0+ (Windows only).

To install the latest stable release of the Causal Testing Framework:

``pip install causal-testing-framework``

or if you want to install with the development packages/tools:

``pip install causal-testing-framework[dev]``

Alternatively, you can install directly via source:

```shell
git clone https://github.com/CITCOM-project/CausalTestingFramework
cd CausalTestingFramework
```
then to install a specific release:

```shell
git fetch --all --tags --prune
git checkout tags/<tag> -b <branch>
pip install . # For core API only
pip install -e . # For editable install, useful for development work
```

For more information on how to use the Causal Testing Framework, please refer to our [documentation](https://causal-testing-framework.readthedocs.io/en/latest/?badge=latest).  
## How to Cite
If you use our framework in your work, please cite the following:

``This research has used version X.Y.Z (software citation) of the 
Causal Testing Framework (paper citation).``

The paper citation should be the Causal Testing Framework [paper](https://dl.acm.org/doi/10.1145/3607184), 
and the software citation should be the specific Figshare [DOI](https://orda.shef.ac.uk/articles/software/CITCOM_Software_Release/24427516) of the version used in your work.

## Acknowledgements 

The Causal Testing Framework is supported by the UK's Engineering and Physical Sciences Research Council (EPSRC),
with the project name [CITCOM](https://gow.epsrc.ukri.org/NGBOViewGrant.aspx?GrantRef=EP/T030526/1) - "_Causal Inference for Testing of Computational Models_" 
under the grant EP/T030526/1.