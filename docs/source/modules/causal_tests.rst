Causal Test Cases
=================

A :term:`causal test case` asserts the expected change in an :term:`outcome variable` that applying an :term:`intervention` to the :term:`treatment variable` should cause.
Causal test cases can be as simple as `X has a causal effect on Y` or as complex as `Y should triple when we change X from 3 to 4 while Z is held constant at 8`.
Interested readers can find a formal definition and extended explanation in `this paper <https://dl.acm.org/doi/10.1145/3607184>`.

A key difference between causal testing and traditional testing is that, with causal testing, the test cases are completely separate entities from the data used to evaluate them.
In traditional testing, the two are almost synonymous.
For those already familiar with traditional testing techniques, this can be very difficult to conceptualise.
However, it is worth taking the time to fully appreciate this separation and the advantages it affords, the main such advantage being that it allows the same data (set of system executions) to be re-used to evaluate *multiple* causal tests.

Components of a test case
-------------------------

Causal test cases have three main components.

1.
  The :doc:`estimator <causal_estimate>` defines the form of the causal relationship under test and how the causal effect will be estimated.
  This is where the treatment and outcome variables are recorded, as well as any variables that must be :term:`adjusted <adjustment>` for to calculate an unbiassed effect estimate.

2.
  The :term:`effect measure` is the causal effect that will be estimated.
  One such metric is the :term:`average treatment effect` is the additive difference in the :term:`outcome variable` that we expect to observe as a result of our :term:`intervention`.
  Another such metric is the :term:`risk ratio`, which is the multiplicative difference.

3.
  The :doc:`expected effect <test_oracle>` is the value of the :term:`effect measure` that we expect to see.
  For example, we may expect our intervention to cause our outcome variable to decrease by 3, or become 4 times larger.
  Alternatively, we not be able to specify the causal effect so precisely and so could validate that the effect measure is positive or negative, or even just that there is some change or no change.

In addition to these three main components, you can also give each test case a name, for easy recognition.
Test cases can also be individually skipped, which can be useful if particular test cases are known to be problematic in some way, for example taking a long time to run.
