# Luo-Rudy 1991 Case Study: APD90 Sensitivity Analysis
In this case study, we demonstrate how causal testing can be used to implement a sensitivity analysis approach measuring the
extent to which a series of conductance-based inputs effect a single output, APD90, in the seminal Luo-Rudy 1991
cardiac action potential model. As described in Section 5.2 of the paper, this involves running a series of causal test
cases that incrementally increase/decrease the treatment value for each input above/below its mean to quantify how much
a given change input causes a change in APD90.

Here we expect that increasing G_K, G_b, and G_K1, will cause APD90 to decrease, increasing G_si will
cause APD90 to increase, and increasing G_Na and G_Kp to have no significant effect on APD90. Further details
can be found in Section 5.2 of the paper.

## How to run
There are two versions of this case study:
1. `causal_test_max_conductances.py` which has a for loop to iteratively call the `causal_test_engine`
2. `causal_test_max_conductances_test_suite.py`, which uses the `causal_test_suite` object to interact with the `causal_test_engine`

To run this case study:
1. Ensure all project dependencies are installed by running `pip install .` in the top level directory
   (instructions are provided in the project README).
2. Change directory to `causal_testing/examples/lr91`.
3. Run the command `python test_max_conductances.py` or `python test_max_conductances_test_suite.py`

This should print a series of causal test results covering the effects of a range of different sized interventions made
to the inputs on APD90, and should also plot Figure 2 from the paper.
