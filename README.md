# Causal Testing Framework: A Causal Inference-Driven Software Testing Framework

![example workflow](https://github.com/CITCOM-project/CausalTestingFramework/actions/workflows/ci-tests.yaml/badge.svg) [![codecov](https://codecov.io/gh/CITCOM-project/CausalTestingFramework/branch/main/graph/badge.svg?token=04ijFVrb4a)](https://codecov.io/gh/CITCOM-project/CausalTestingFramework) [![Documentation Status](https://readthedocs.org/projects/causal-testing-framework/badge/?version=latest)](https://causal-testing-framework.readthedocs.io/en/latest/?badge=latest)

Causal testing is a causal inference-driven framework for functional black-box testing. This framework utilises
graphical causal inference (CI) techniques for the specification and functional testing of software from a black-box
perspective. In this framework, we use causal directed acyclic graphs (DAGs) to express the anticipated cause-effect
relationships amongst the inputs and outputs of the system-under-test and the supporting mathematical framework to
design statistical procedures capable of making causal inferences. Each causal test case focuses on the causal effect of
an intervention made to the system-under test. That is, a prescribed change to the input configuration of the
system-under-test that is expected to cause a change to some output(s).

![Causal Testing Workflow](images/workflow.png)


## Installation

See the readthedocs site for [installation
instructions](https://causal-testing-framework.readthedocs.io/en/latest/installation.html).

## Documentation

Further information on causal inference, the code, usage and more can be found on the [docs](https://causal-testing-framework.readthedocs.io/en/latest/)
