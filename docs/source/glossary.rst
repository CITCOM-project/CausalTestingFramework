.. Glossary

Glossary
###################

.. glossary::

   Causal inference
    Causal inference (:abbr:`CI`) is a family of statistical techniques designed to quantify and establish **causal**
    relationships in data. In contrast to purely statistical techniques that are driven by associations in data, CI
    incorporates knowledge about the data-generating mechanisms behind relationships in data to derive causal conclusions.

    Causal DAG
      A Directed Acyclic Graph depicting the direct causal relationships between variables, in which an edge ``X -> Y`` indicates that ``X`` directly causes ``Y``. That is, there exists an intervention on ``X`` which brings about a change in ``Y``.

      Treatment Variable
        The changed variable of interest (``X``).

      Outcome Variable
        The observed variable of interest (``Y``).

    Causal Specification
      A causal specification is a pair ``S = (M, G)`` comprising a modelling scenario ``M`` and a causal DAG ``G`` capturing the causal relationships amongst the inputs and outputs of the SUT that are central to the modelling scenario.

    Causal Test Case
      A causal test case is a 4-tuple ``(M, X, |delta|, Y)`` that captures the expected causal effect, Y, of an intervention, ``delta``, made to an input valuation, ``X``, on some model outcome in the context of modelling scenario ``M``.

   Directed acyclic graph (DAG)
    A directed acyclic graph (:abbr:`DAG`) is a graphical representation used in causal inference to model and visualize relationships between variables.
    In a DAG, nodes represent variables, and directed edges between nodes indicate causal relationships, with the absence of cycles ensuring acyclicity.

   Estimate Type
    The effect measure to use, typically (C)ATE, Risk Ratio, or Odds Ratio

      ATE
      ~~~~
      **Average treatment effect** (:abbr:`ATE`): The additive difference in the outcome between the control and treatment populations.

      CATE
      ~~~~~~~~~~~
      **Conditional ATE** (:abbr:`CATE`): The additive difference in the outcome between the control and treatment populations across different strata of the population.

      Risk Ratio
      ~~~~~~~~~~~
      **Risk ratio**: The multiplicative difference in the outcome between the control and treatment populations.

      Odds Ratio
      ~~~~~~~~~~~
      **Odds Ratio**: The ratio of the odds of A in the presence of B and the odds of A in the absence of B.


   Minimal Adjustment Set
      The smallest set of variables which must be controlled, or "adjusted for", to produce an unbiased estimate of causal effect.

   Scenario
      A modelling scenario ``M`` is a pair ``(X, C)`` where ``X`` is a non-strict subset of the model's input variables and ``C`` is a set of constraints over valuations of ``C``, which may be empty.

   Scenario Execution
      A software execution satisfying a given modelling scenario.

   Intervention
      An intervention ``delta : X -> X'`` is a function which manipulates the values of a subset of input valuations.