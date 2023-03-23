
Causal Testing
==============

This package contains the main components of the causal testing framework, causal tests and causal oracles, which utilise both the specification and data collection packages.

A causal test case is a triple ``(X, \Delta, Y)`` where ``X`` is an input configuration, ``\Delta`` is an intervention, and ``Y`` is the expected causal effect of applying ``\Delta`` to ``X``. Put simply, a causal test case states the expected change in an outcome that applying an intervention to X should cause. In this context, an intervention is simply a function which manipulates the input configuration of the scenario-under-test in a way that is expected to cause a change to some outcome.

For example, suppose we have an epidemiological computational model and we are testing the model in a classroom scenario. In particular, we are interested in how various precautions, such as hand washing and mask wearing, can prevent the spread of the virus in a classroom. Let us walk through the steps of causal testing.

Specification
-------------

In our causal specification, we define the scenario with the following constraints:


* ``n_people ~ Uniform(20, 30)`` (There are between 20 and 30 people in the classroom).
* ``environment = Grid(x,y ~ Uniform(20, 40))`` (The classroom is square grid of between 20x20 and 40x40 units).
* ``n_infected_t0 = 1`` (One person is infected initially).
* ``precaution = None`` (No precautions taken).
  We also specify the output we are interested in as ``n_infected_t5``\ , the number of people infected after five days of daily one hour lessons.

Then, we create a simple causal DAG which represents causality amongst these variables:

.. code-block::

   digraph CausalDAG {
     n_people -> n_infected_t5
     environment -> n_infected_t5
     n_people -> environment
     n_infected_t0 -> n_infected_t5
     environment -> precaution
     precaution -> n_infected_t5
   }

Causal Test Cases
-----------------

We then define a number of causal test cases to apply to the scenario-under-test. For example, supposing we expect mask wearing and hand washing to have a preventative effect:


* ``mask_wearing_test = (X={precaution = None}, \Delta = {precaution = Mask}, Y = {-20% < n_infected_t5 < -10% })`` (Mask wearing is expected to result in between 10% and 20% fewer infections).
* ``hand_washing_test = (X={precaution = None}, \Delta = {precaution = Hand Washing}, Y = {-40% < n_infected_t5 < -25%})`` (Hand washing is expected to result in between 25% and 40% fewer infections).

Data Collection
---------------

To run these test cases experimentally, we need to execute both ``X`` and ``\Delta(X)`` - that is, with and without the interventions. Since the only difference between these test cases is the intervention, we can conclude that the observed difference in ``n_infected_t5`` was caused by the interventions. While this is the simplest approach, it can be extremely inefficient at scale, particularly when dealing with complex software such as computational models.

To run these test cases observationally, we need to collect *valid* observational data for the scenario-under-test. This means we can only use executions with between 20 and 30 people, a square environment of size betwen 20x20 and 40x40, and where a single person was initially infected. In addition, this data must contain executions both with and without the intervention. Next, we need to identify any sources of bias in this data and determine a procedure to counteract them. This is achieved automatically using graphical causal inference techniques that identify a set of variables that can be adjusted to obtain a causal estimate. Finally, for any categorical biasing variables, we need to make sure we have executions corresponding to each category otherwise we have a positivity violation (i.e. missing data). In the worst case, this at least guides the user to an area of the system-under-test that should be executed.

Causal Inference
----------------

After collecting either observational or experimental data, we now need to apply causal inference. First, as described above, we use our causal graph to identify a set of adjustment variables which mitigate all bias in the data. Next, we use statistical models to adjust for these variables (implementing the statistical procedure necessary to isolate the causal effect) and obtain the desired causal estimate. Depending on the statistical model used, we can also generate 95% confidence intervals (or confidence intervals at any confidence level for that matter).

In our example, the causal DAG tell us it is necessary to adjust for ``environment`` in order to obtain the causal effect of ``precaution`` on ``n_infected_t5``. Supposing the relationship is linear, we can employ a linear regression model of the form ``n_infected_t5 ~ p0*precaution + p1*environment`` to carry out this adjustment. If we use experimental data, only a single environment is used by design and therefore the adjustment has no impact. However, if we use observational data, the environment may vary and therefore this adjustment will look at the causal effect within different environments and then provide a weighted average, which turns out to be the partial coefficient ``p0``.

Test Oracle Procedure
---------------------

After conducting causal inference, all that remains is to ascertain whether the causal effect is expected or not. In our example, this is simply a case of checking whether the causal effect on ``n_infected_t5`` falls within the specified range. However, in the future, we may wish to implement more complex oracles.
