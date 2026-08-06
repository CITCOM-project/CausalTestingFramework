Causal Estimate
===============

This page provides an overview on how to choose the most appropriate estimator for your workflow.
When using the :code:`generate` command to generate causal tests from a causal DAG, the estimator used is chosen based on the datatype of your outcome variable:

* Linear regression is used for numerical variables
* Logistic regression is used for boolean variables
* Multinomial regression is used for categorical variables

In general, you won't need to change this.
However, if you have variables that are not recorded in your data, you may need to use the *instrumental variable estimator*.
This uses the concept of `instrumental variables <https://en.wikipedia.org/wiki/Instrumental_variables>`_ to :term:`adjust for <adjustment>` variables without needing to know their values.
To change the estimator, you will need change the name of the estimator in the JSON representation of the test cases produced by the :code:`generate` command.
Depending on which estimator you are using, you may also need to add values for different parameters.
See below for details.

Another useful customisation option for regression estimators is to modify the formula used for estimation.
By default, the linear, logistic, and multinomial regression estimators all use the formula :code:`Y ~ X + Z1 + Z2 + ...`, where :code:`Y` is the :term:`outcome variable`, :code:`X` is the :term:`treatment variable`, and :code:`Z1`, :code:`Z2`, etc. are the variables in the :term:`adjustment set`.
However, if you know in advance that your causal relationship is non-linear, it may be wise to refine this equation somewhat.
This is mostly quite intuitive, but does have a few quirks.
See the `patsy documentation <https://patsy.readthedocs.io/en/latest/formulas.html#the-formula-language>`_ for an explanation of the available operators.
An example can be seen in our :doc:`tutorials <../tutorials/poisson_line_process/poisson_line_process_tutorial>`.


LinearRegressionEstimator
-------------------------

**Recommended use:** For continuous numerical outcomes (e.g. the number of people who are vaccinated).

.. autoclass:: causal_testing.estimation.linear_regression_estimator.LinearRegressionEstimator
   :members:
   :exclude-members: from_formula, regressor
   :undoc-members:
   :show-inheritance:
   :noindex:

LogisticRegressionEstimator
---------------------------

**Recommended use:** For binary outcomes (yes/no, true/false, success/failure).

.. autoclass:: causal_testing.estimation.logistic_regression_estimator.LogisticRegressionEstimator
   :members:
   :exclude-members: from_formula, regressor
   :undoc-members:
   :show-inheritance:
   :noindex:

MultinomialRegressionEstimator
------------------------------

**Recommended use:** For categorical outcomes (e.g. colurs: Red, Green, Blue).

.. autoclass:: causal_testing.estimation.multinomial_regression_estimator.MultinomialRegressionEstimator
   :members:
   :exclude-members: from_formula, regressor
   :undoc-members:
   :show-inheritance:
   :noindex:

InstrumentalVariableEstimator
-----------------------------

**Recommended use:** When dealing with unmeasured confounding using instrumental variables.

.. autoclass:: causal_testing.estimation.instrumental_variable_estimator.InstrumentalVariableEstimator
   :members:
   :undoc-members:
   :show-inheritance:
   :noindex:

IPCWEstimator
-------------

**Recommended use:** For handling missing data or selection bias using inverse probability of censoring weighting (e.g. time-varying data).

.. autoclass:: causal_testing.estimation.ipcw_estimator.IPCWEstimator
   :members:
   :undoc-members:
   :show-inheritance:
   :noindex:

ExperimentalEstimator
---------------------

**Recommended use:** For randomised controlled trials or experimental data where treatment assignment is randomised.
                     Directly runs the system under test multiple times with different configurations (e.g. you need to collect new data by executing your system multiple times).

.. autoclass:: causal_testing.estimation.experimental_estimator.ExperimentalEstimator
   :members:
   :undoc-members:
   :show-inheritance:
   :noindex:

Custom Estimators
-----------------

If the above estimators are not sufficient for your needs, you can implement your own custom estimator by extending the :code:`Estimator` class and implementing the abstract :code:`add_modelling_assumptions` method and the estimation method for the causal effect measure you wish to calculate.
For example, if you wished to estimate the :term:`ATE` using the empirical mean of the recorded outcome under the control and treatment values, you would need to implement a method called :code:`estimate_ate`.
If you wished to estimate the risk ratio, you would need to call your method :code:`estimate_risk_ratio`.
The code for the :code:`EmpiricalMeanEstimator` is shown below.

..  code-block:: python

  import pandas as pd
  from causal_testing.estimation.abstract_estimator import Estimator
  from causal_testing.estimation.effect_estimate import EffectEstimate

  class EmpiricalMeanEstimator(Estimator):
      """
      Custom estimator class to estimate the causal effect based on the empirical mean.
      """

      def add_modelling_assumptions(self):
          """
          Add modelling assumptions to the estimator. This is a list of strings which list the modelling assumptions that
          must hold if the resulting causal inference is to be considered valid.
          """
          self.modelling_assumptions += "The data must contain runs with the exact configuration of interest."

      def estimate_ate(self, df: pd.DataFrame) -> EffectEstimate:
          """Estimate the outcomes under control and treatment.
          :param df: The data to use.
          :return: The empirical average treatment effect.
          """

          control_results = df.where(df[self.treatment_variable] == self.control_value)[self.outcome_variable].dropna()
          treatment_results = df.where(df[self.treatment_variable] == self.treatment_value)[
              self.outcome_variable
          ].dropna()

          def ate(sample1, sample2):
              return sample1.mean() - sample2.mean()

          bootstraps = bootstrap((treatment_results, control_results), ate, confidence_level=self.alpha)
          return EffectEstimate(
              type="ate",
              value=ate(treatment_results, control_results),
              ci_low=bootstraps.confidence_interval.low,
              ci_high=bootstraps.confidence_interval.high,
          )

Once you have implemented your estimator, you will need to register it as an extra entry point in your project's :code:`pyproject.toml` file so that the Causal Testing Framework can find it.
For example, if you had defined your :code:`EmpiricalMeanEstimator` class in a module called :code:`empirical_mean_estimator` in a folder called :code:`custom_estimators`, you would register it as follows.
You will also need to reinstall your project, e.g. with :code:`pip install -e .` each time you add a new estimator to your :code:`pyproject.toml`.
You do not need to reinstall each time you edit your project for source code edits.


..  code-block:: ini

 [project.entry-points."estimators"]
 CustomFlakefighter = "custom_estimators.empirical_mean_estimator:EmpiricalMeanEstimator"

Of course, for this to work, your module needs to be discoverable on your python path.
That is, you should be able to execute :code:`from custom_estimators.empirical_mean_estimator import EmpiricalMeanEstimator` successfully from within the current working directory.

You can also add your custom estimator to causal test cases specified in JSON.
To do so, you can simply set the :code:`estimator` property to the name of your estimator class and the :code:`estimate_type` property to the name of your causal effect measure.
In the above :code:`EmpiricalMeanEstimator` example, :code:`estimator` would be set to  :code:`"EmpiricalMeanEstimator"` and :code:`estimate_type` would be set to :code:`"ate"`.
