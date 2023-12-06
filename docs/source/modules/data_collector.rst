Data Collection
===============

For causal testing, we require data for the scenario-under-test. This data can be collected in 2 ways: experimentally
and observationally.

Experimental Data Collector
****************************
- Experimental data collection involves running the system-under-test with two specific input configurations, one with the
  intervention and one without. We refer to these as the treatment and control configurations, respectively. The only
  difference between these two input configurations is the intervention and therefore the observed difference in outcome
  is the causal effect. If the system-under-test is non-deterministic, each input configuration should be ran multiple
  times to observe the difference in the distributions of outputs.

Observational Data Collector
*****************************

- Observational data collection involves collecting past execution data for the system-under-test that was not ran under
  the experimental conditions necessary to isolate the causal effect. Instead, we will use the causal knowledge encoded
  in the causal specification's causal DAG to identify and appropriately mitigate any sources of bias in the data. That
  way, we can still obtain the causal effect of the intervention but avoid running costly experiments.

- We cannot use any data as observational data, though. We need to ensure that the data is representative of the
  scenario-under-test. To achieve this, we filter any provided data using the defined constraints by checking whether the
  data for a variables falls within the specified distribution or meets the exact specified value.

- This package should contain methods which collect the data for causal inference. Users must implement these methods in a way that generates (experimental) or collects
  (observational) data for the scenario-under-test. For the observational case, we should also provide helper methods which filter the data.