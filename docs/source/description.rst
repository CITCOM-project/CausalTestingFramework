Framework Overview
=====================================

CTF Components
--------------

#.
   :doc:`Causal Specification <../modules/causal_specification>`\ : Before we can test software, we need to obtain an
   understanding of how it should behave in a particular use-case scenario. In addition, to apply graphical CI
   techniques for testing, we need a causal DAG which depicts causal relationships amongst inputs and outputs. To
   collect this information, users must create a *causal specification*. This comprises a set of scenarios which place
   constraints over input variables that capture the use-case of interest, a causal DAG corresponding to this scenario,
   and a series of high-level functional requirements that the user wishes to test. In causal testing, these
   requirements should describe how the model should respond to interventions (changes made to the input configuration).

#.
   :doc:`Causal Tests <../modules/causal_tests>`\ : With a causal specification in hand, we can now go about designing
   a series of test cases that interrogate the causal relationships of interest in the scenario-under-test. Informally,
   a causal test case is a triple (M, X, Delta, Y), where M is the modelling scenario, X is an input configuration,
   Delta is an intervention which should be applied to X, and Y is the expected *causal effect* of that intervention on
   some output of interest. Therefore, a causal test case states the expected causal effect (Y) of a particular
   intervention (Delta) made to an input configuration (X). For each scenario, the user should create a suite of causal
   tests. Once a causal test case has been defined, it is executed as follows:


   #. Using the causal DAG, identify an estimand for the effect of the intervention on the output of interest. That is,
      a statistical procedure capable of estimating the causal effect of the intervention on the output.
   #. Collect the data to which the statistical procedure will be applied (see Data collection below).
   #. Apply a statistical model (e.g. linear regression or causal forest) to the data to obtain a point estimate for
      the causal effect. Depending on the estimator used, confidence intervals may also be obtained at a specified
      confidence level e.g. 0.05 corresponds to 95% confidence intervals (optional).
   #. Return the casual test result including a point estimate and 95% confidence intervals, usually quantifying the
      average treatment effect (ATE).
   #. Implement and apply a test oracle to the causal test result - that is, a procedure that determines whether the
      test should pass or fail based on the results. In the simplest case, this takes the form of an assertion which
      compares the point estimate to the expected causal effect specified in the causal test case.

#.
   :doc:`Data Collection <../modules/data_collector>`\ : Data for the system-under-test can be collected in two
   ways: experimentally or observationally. The former involves executing the system-under-test under controlled
   conditions which, by design, isolate the causal effect of interest (accurate but expensive), while the latter
   involves collecting suitable previous execution data and utilising our causal knowledge to draw causal inferences (
   potentially less accurate but efficient). To collect experimental data, the user must implement a single method which
   runs the system-under-test with a given input configuration. On the other hand, when dealing with observational data,
   we automatically check whether the data is suitable for the identified estimand in two steps. First, confirm whether
   the data contains a column for each variable in the causal DAG. Second, we check
   for `positivity violations <https://www.youtube.com/watch?v=4xc8VkrF98w>`_. If there are positivity violations, we can
   provide instructions for an execution that will fill the gap (future work).

For more information on each of these steps, follow the link to their respective documentation.

Causal Inference Terminology
----------------------------

Here are some explanations for the causal inference terminology used above.


* Causal inference (CI) is a family of statistical techniques designed to quantify and establish **causal**
  relationships in data. In contrast to purely statistical techniques that are driven by associations in data, CI
  incorporates knowledge about the data-generating mechanisms behind relationships in data to derive causal conclusions.
* One of the key advantages of CI is that it is possible to answer causal questions using **observational data**. That
  is, data which has been passively observed rather than collected from an experiment and, therefore, may contain all
  kinds of bias. In a testing context, we would like to leverage this advantage to test causal relationships in software
  without having to run costly experiments.
* There are many forms of CI techniques with slightly different aims, but in this framework we focus on graphical CI
  techniques that use directed acyclic graphs to obtain causal estimates. These approaches used a causal DAG to explain
  the causal relationships that exist in data and, based on the structure of this graph, design statistical experiments
  capable of estimating the causal effect of a particular intervention or action, such as taking a drug or changing the
  value of an input variable.
