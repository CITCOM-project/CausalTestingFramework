Custom Estimators
=================

If the supported :ref:`estimators` are not sufficient for your needs, you can implement your own custom estimator by extending the :code:`Estimator` class and implementing the abstract :code:`add_modelling_assumptions` method and the estimation method for the causal effect measure you wish to calculate.
For example, if you wished to estimate the ATE using the empirical mean of the recorded outcome under the control and treatment values, you would need to implement a method called :code:`estimate_ate`.
If you wished to estimate the risk ratio, you would need to call your method :code:`estimate_risk_ratio`.
The code for the :code:`EmpiricalMeanEstimator` is shown below.

..  code-block:: python

  from causal_testing.estimation.abstract_estimator import Estimator
  from scipy.stats import bootstrap

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

      def estimate_ate(self) -> EffectEstimate:
          """Estimate the outcomes under control and treatment.
          :return: The empirical average treatment effect.
          """
          treatment_variable = self.base_test_case.treatment_variable.name
          outcome_variable = self.base_test_case.outcome_variable.name

          control_results = self.df.where(self.df[treatment_variable] == self.control_value)[outcome_variable].dropna()
          treatment_results = self.df.where(self.df[treatment_variable] == self.treatment_value)[
              outcome_variable
          ].dropna()

          def risk_ratio(sample1, sample2):
              return sample1.mean() - sample2.mean()

          bootstraps = bootstrap((treatment_results, control_results), risk_ratio, confidence_level=self.alpha)
          return EffectEstimate(
              type="risk_ratio",
              value=risk_ratio(treatment_results, control_results),
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
