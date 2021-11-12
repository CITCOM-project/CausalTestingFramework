# Causal Specification

A causal specification comprises two components: a series of scenario requirements for the system-under-test and a corresponding causal DAG.

Each causal test case focuses on a particular scenario or use-case of the system-under-test that involves some intervention. For that scenario, the user must construct a series of requirements and constraints which characterise the scenario-under-test. A constraint is simply a mapping from a subset of inputs to a value or distribution.

For example, in an epidemiological model, a scenario might be a simulation which focuses on the effect of a vaccine on a specific number of people. The causal specification should place constraints over the values or distributions of inputs parameters that characterise this simulation, such as a `pop_size=10000` or `pop_size ~ normal(mean=10000, var=1000)`, and `vaccine = pfizer`. The user will then write high-level requirements surrounding this scenario, such as `The vaccine should cause the number of deaths, infections, and serious cases to decrease`.

Additionally, the user must construct a causal DAG which represents the anticipated cause-effect relationships amongst the inputs and outputs relevant to the scenario. This simply involves creating nodes for each input and output and adding edges where one variable is expected to cause another. For example, `vaccine --> infections` encodes the belief that the vaccine should cause a change to infections.

Collectively, the components of the causal specification provide both contextual information in the form of constraints and requirements, as well as causal information in the form of a causal DAG. Later on, these components will be used to design statistical experiments that can answer causal questions about the scenrio-under-test, such as `Does the pfizer vaccine cause a decrease in the number of infections, deaths, and serious cases?`
