# Covasim Case Study: Doubling Beta (Infectiousness)
In this case study, we demonstrate how to use the causal testing framework with observational
data collected Covasim to conduct Statistical Metamorphic Testing (SMT) a posteriori. Here, we focus on a set of simple
modelling scenarios that investigate how the infectiousness of the virus (encoded as the parameter beta) affects the
cumulative number of infections over a fixed duration. We also run several causal tests that focus on increasingly
specific causal questions pertaining to more refined metamorphic properties and enabling us to learn more about the
model's behaviour without further executions.

More information about the case study can be found in Section 5.3
(Doubling Beta) of the paper.

## How to run
To run this case study:
1. Ensure all project dependencies are installed by running `pip install .` from the top
level of this directory (instructions are provided in the project README).
2. Change directory to `causal_testing/examples/covasim_/doubling_beta`.
3. Run the command `python test_beta.py`.

This will print out a series of test results covering a range of different causal questions that correspond to those
in Table 3 of the paper.
