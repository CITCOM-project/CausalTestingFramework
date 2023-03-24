Glossary
============

We here define some of the key terms associated with causal inference and testing.

.. glossary::

    Intervention
      An intervention |delta| : X -> X' is a function which manipulates the values of a subset of input valuations.

    Causal DAG
      A Directed Acyclic Graph depicting the direct causal relationships between variables, in which an edge X -> Y indicates that X directly causes Y. That is, there exists an intervention on X which brings about a change in Y.

      Treatment
        The changed variable of interest (X).

      Outcome
        The observed variable of interest (Y).

    Scenario
      A modelling scenario M is a pair (X, C) where X is a non-strict subset of the model's input variables and C is a set of constraints over valuations of C, which may be empty.

    Causal Specification
      A causal specification is a pair S = (M, G) comprising a modelling scenario M and a causal DAG G capturing the causal relationships amongst the inputs and outputs of the SUT that are central to the modelling scenario.

    Causal Test Case
      A causal test case is a 4-tuple (M, X, |delta|, Y) that captures the expected causal effect, Y, of an intervention, |delta|, made to an input valuation, X, on some model outcome in the context of modelling scenario M.


    Estimate Type
      The effect measure to use, typically (C)ATE, Risk Ratio, or Odds Ratio

    ATE
      Average treatment effect: The additive difference in the outcome between the control and treatment populations.

    CATE
      Conditional ATE: The additive difference in the outcome between the control and treatment populations across different strata of the population.

    Risk ratio
      The multiplicative difference in the outcome between the control and treatment populations.

    Odds Ratio
      The ratio of the odds of A in the presence of B and the odds of A in the absence of B.

    Minimal Adjustment Set
      The smallest set of variables which must be controlled, or "adjusted for", to produce an unbiassed estimate of causal effect.

    Scenario Execution
      A software execution satisfying a given modelling scenario.
