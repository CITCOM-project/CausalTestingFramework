Test Data
=========

A key difference between causal testing and traditional testing is that causal testing operates statistically over *multiple runs* of a system.
This means that the process of collecting test data is completely separate from the process of evaluating test cases.
Furthermore, the process of causal :term:`identification`, enables pre-existing data to be used rather than having to collate a bespoke set of system runs without risking untrustworthy test outcomes due to biassed datasets.
The benefit of this is a potentially huge saving in computational cost.

The causal inference techniques that underpin the Causal Testing Framework make `three key assumptions <https://miguelhernan.org/whatifbook>`_ about the data.
In practice, any test data that records all relevant variables and achieves a good coverage of the input space should satisfy these assumptions.

1. **No unobserved confounding:** This means that all relevant variables have been recorded.
As discussed in the section on :doc:`causal graphs <../modules/causal_dag>`\ , it is important to record all relevant causal relationships in the causal DAG to facilitate causal :term:`identification`.
The Causal Testing Framework provides :doc:`special estimation techniques <../modules/causal_estimate>` that can adjust for unobserved variables in certain circumstances.

2. **Consistency:** Formally, this means that the observed output of a run of the system under a particular configuration matches the :term:`potential outcome` under that configuration.
Intuitively, this should be trivially true (especially for software systems) if the causal test case and the causal DAG are both well-specified.
For example, when examining the relationship between a person's weight and their risk of heart attach, the risk (and thus the potential outcomes) associated with different weight loss interventions may be very different.
In this example, we could add a node to our causal DAG which represents which (if any) weight loss interventions a person has undergone in order to facilitate proper :term:`identification` and :term:`adjustment`.

3. **Positivity:** Formally, this means that the probability of all relevant treatment values is non-zero.
Intuitively, this just means that the dataset needs to achieve good coverage of the input space.
This is of particular importance for binary and categorical inputs when several variables need to be adjusted for, as a lack of data can make it impossible to estimate a causal effect.
