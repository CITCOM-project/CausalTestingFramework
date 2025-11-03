Estimators Overview
===================

This page provides an overview on how to choose the most appropriate estimator for your workflow.


LinearRegressionEstimator
~~~~~~~~~~~~~~~~~~~~~~~~~

**Recommended use:** For continuous numerical outcomes (e.g. the number of people who are vaccinated).

.. autoclass:: causal_testing.estimation.linear_regression_estimator.LinearRegressionEstimator
   :members:
   :exclude-members: from_formula, regressor
   :undoc-members:
   :show-inheritance:
   :noindex:

LogisticRegressionEstimator
~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Recommended use:** For binary outcomes (yes/no, true/false, success/failure).

.. autoclass:: causal_testing.estimation.logistic_regression_estimator.LogisticRegressionEstimator
   :members:
   :exclude-members: from_formula, regressor
   :undoc-members:
   :show-inheritance:
   :noindex:

CubicSplineRegressionEstimator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Recommended use:** For continuous outcomes with non-linear relationships or changes in behaviour.
Useful when the relationship between treatment and outcome cannot be captured by a linear model.

.. autoclass:: causal_testing.estimation.cubic_spline_estimator.CubicSplineRegressionEstimator
   :members:
   :undoc-members:
   :show-inheritance:
   :noindex:


InstrumentalVariableEstimator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Recommended use:** When dealing with unmeasured confounding using instrumental variables.

.. autoclass:: causal_testing.estimation.instrumental_variable_estimator.InstrumentalVariableEstimator
   :members:
   :undoc-members:
   :show-inheritance:
   :noindex:

IPCWEstimator
~~~~~~~~~~~~~

**Recommended use:** For handling missing data or selection bias using inverse probability of censoring weighting (e.g. time-varying data).

.. autoclass:: causal_testing.estimation.ipcw_estimator.IPCWEstimator
   :members:
   :undoc-members:
   :show-inheritance:
   :noindex:

ExperimentalEstimator
~~~~~~~~~~~~~~~~~~~~~

**Recommended use:** For randomised controlled trials or experimental data where treatment assignment is randomised.
                     Directly runs the system under test multiple times with different configurations (e.g. you need to collect new data by executing your system multiple times).

.. autoclass:: causal_testing.estimation.experimental_estimator.ExperimentalEstimator
   :members:
   :undoc-members:
   :show-inheritance:
   :noindex:
