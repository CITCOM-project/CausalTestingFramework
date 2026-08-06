Test Oracle
===========

As in traditional testing, the `oracle <https://en.wikipedia.org/wiki/Test_oracle>`_ is a procedure used to determine whether the observed behaviour is actually correct.
In causal testing, this represents checking that the causal effect estimated from the test data is what was expected.
The Causal Testing Framework supports several causal effects by default.
The most basic oracle procedure is to simply validate the presence or absence of a causal effect.
This requires nothing more than the edges of the causal DAG.
If you know the direction of a causal relationship (positive or negative), you can add a little more precision to your causal tests.
If you know precisely what the causal effect should be, you can check for a particular value, within a specified tolerance.

SomeEffect
----------

**Recommended use:** For validating the presence of a causal effect between two variables.
For additive :term:`effect measures` such as :term:`ATE`, :term:`CATE`, this involves checking that the :term:`confidence intervals` associated with a causal effect estimate do not contain zero.
For multiplicative :term:`effect measures` such as :term:`risk ratio` this involves checking that the :term:`confidence intervals` associated with a causal effect estimate do not contain one.

.. autoclass:: causal_testing.testing.causal_effect.SomeEffect
   :members:
   :undoc-members:
   :show-inheritance:
   :noindex:

NoEffect
----------

**Recommended use:** For validating the absence of a causal effect between two variables.
For additive :term:`effect measures` such as :term:`ATE`, :term:`CATE`, this involves checking that the :term:`confidence intervals` associated with a causal effect estimate contain zero.
For multiplicative :term:`effect measures` such as :term:`risk ratio` this involves checking that the :term:`confidence intervals` associated with a causal effect estimate contain one.

.. autoclass:: causal_testing.testing.causal_effect.NoEffect
  :members:
  :undoc-members:
  :show-inheritance:
  :noindex:

Positive
----------

**Recommended use:** For validating a positive causal effect.
For additive :term:`effect measures` such as :term:`ATE`, :term:`CATE`, this involves checking that the :term:`confidence intervals` associated with a causal effect estimate are both above zero.
For multiplicative :term:`effect measures` such as :term:`risk ratio` this involves checking that the :term:`confidence intervals` associated with a causal effect estimate are both above one.

.. autoclass:: causal_testing.testing.causal_effect.Positive
  :members:
  :undoc-members:
  :show-inheritance:
  :noindex:

Negative
----------

**Recommended use:** For validating a negative causal effect.
For additive :term:`effect measures` such as :term:`ATE`, :term:`CATE`, this involves checking that the :term:`confidence intervals` associated with a causal effect estimate are both below zero.
For multiplicative :term:`effect measures` such as :term:`risk ratio` this involves checking that the :term:`confidence intervals` associated with a causal effect estimate are both below one.

.. autoclass:: causal_testing.testing.causal_effect.Negative
  :members:
  :undoc-members:
  :show-inheritance:
  :noindex:

ExactValue
----------

**Recommended use:** For specifying a precise value for the expected causal effect.
Here, you can also specify arithmetic tolerance, categorical tollerance (the minimum proportion of categories that must exhibit the expected effect for the test to pass), and confidence interval limits.

.. autoclass:: causal_testing.testing.causal_effect.ExactValue
  :members:
  :undoc-members:
  :show-inheritance:
  :noindex:

Custom Causal Effects
---------------------

As with :ref:`custom estimators <modules/causal_estimate:Custom Estimators>`, you can also implement your own custom causal effects if the options above are not sufficient for your needs.
To do this, you can extend the :code:`CausalEffect` class and implement your own :code:`apply` method that takes an :doc:`effect estimate <../autoapi/causal_testing/estimation/effect_estimate/index>` and returns boolean :code:`True` if the test should pass and :code:`False` otherwise.
