.. Glossary

Glossary
########

.. glossary::

  Adjustment
    The process of controlling, or "taking into account", variables other than the treatment and outcome in order to calculate an unbiassed effect estimate.

  Adjustment Set
    A set of variables that must be controlled or "taken into account" to calculate an unbiassed effect estimate.

  Causal inference
    Causal inference (:abbr:`CI (Causal Inference)`) is a family of statistical techniques designed to quantify and establish **causal** relationships in data.
    In contrast to purely statistical techniques that are driven by associations in data, CI incorporates knowledge about the data-generating mechanisms behind relationships in data to derive causal conclusions.

  Causal DAG
    A Directed Acyclic Graph depicting the direct causal relationships between variables, in which an edge ``X -> Y`` indicates that ``X`` directly causes ``Y``.
    That is, there exists an intervention on ``X`` which brings about a change in ``Y``.

  Causal Test Case
    A causal test case asserts an expected causal effect on an :term:`outcome variable` that results from an :term:`intervention` (change) on a :term:`treatment variable` `X`.
    See `this paper <https://dl.acm.org/doi/10.1145/3607184>` for a formal definition and extended explanation.
    Causal test cases can be as simple as `X has a causal effect on Y` or as complex as `Y should triple when we change X from 3 to 4 while Z is held constant at 8`.

  Confidence Intervals
    The range of values that are likely to contain the "true" causal effect value, with respect to a given significance level.
    For example, if we estimate the 95% confidence intervals, this can be interpreted as meaning that if the same data generation procedure were repeated 100 times from the same underlying population, approximately 95 of the resulting intervals would be expected to contain the true value.
    There are also alternative interpretations in the literature, which interested readers are invited to investigate in their own time.

  DAG
  Directed acyclic graph
    A directed acyclic graph (:abbr:`DAG (Directed Acyclic Graph)`) is a graphical representation used in causal inference to model and visualize relationships between variables.
    In a DAG, nodes represent variables, and directed edges between nodes indicate causal relationships, with the absence of cycles ensuring acyclicity.

  Effect Measure
  Effect Measures
    The effect measure to use, typically ATE, CATE, Risk Ratio, or Odds Ratio.

  ATE
  Average Treatment Effect
    The additive difference in the outcome between the control and treatment populations.

  CATE
  Conditional Average Treatment Effect
    The additive difference in the outcome between the control and treatment populations across different strata of the population.

  Risk Ratio
     The multiplicative difference in the outcome between the control and treatment populations.

  Odds Ratio
     The ratio of the odds of A in the presence of B and the odds of A in the absence of B.

  Identification
    The process of analysing a causal DAG to determine the variables which should be *adjusted for* in order to calculate an unbiassed causal effect of a treatment variable X on an outcome variable Y.
    Interested readers can find a more technical definition `here <https://miguelhernan.org/whatifbook>`_.

  Inestimable
    When a test case is evaluated with insufficient data to calculate a causal effect estimate at all, the test will return an *inestimable* outcome (rather than pass or fail).
    This is typically an indication that the data has violated the :doc:`positivity <modules/test_data>` assumption, fundamental for causal inference.

  Intervention
    An intervention ``delta : X -> X'`` is a function which manipulates the values of a subset of input valuations.

  Minimal Adjustment Set
    The smallest set of variables which must be controlled, or "adjusted for", to produce an unbiased estimate of causal effect.

  Outcome Variable
    The variable in a :term:`causal test case` that is being observed.
    This is also referred to as the "dependent variable" in some fields.

  Potential Outcome
    When we run a system under a particular configuration, we can observe one of several possible (of *potential*) outcomes.
    Interested readers can find a more technical definition `here <https://miguelhernan.org/whatifbook>`_.

  Test Oracle
    A test oracle determines whether the observed outcome is correct. In our framework, this is whether the expected causal effect matches the estimated causal effect.

  Test Adequacy
    A measurement of how well a given system has been tested.
    While all metrics are approximate, the common goal is that a better score should indicate a lower probability of failure.

  Treatment Variable
    The variable in a :term:`causal test case` that is being changed.
    This is also referred to as the "independent variable" in some fields.
