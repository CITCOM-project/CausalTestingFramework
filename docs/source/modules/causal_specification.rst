
Causal Specification
====================

In causal testing, our units of interest are specific usage **scenarios** of the system-under-test. For example, when testing an epidemiological computational model, one scenario could focus on the simulation of the spread of a virus in a crowded indoors space. For this scenario, our causal specification will describe how a number of interventions should **cause** some outputs to change e.g. opening a window should reduce the spread of the virus by some factor.

In order to isolate the causal effect of the defined interventions, the user needs to express the anticipated cause-effect relationships amongst the inputs and outputs involved in the scenario. This is achieved using a causal DAG, a simple dot and arrow graph that does not contain any cycles where nodes are random variables that represent the inputs and outputs in the scenario-under-test, and edges represent causality. For example, ``window --> infection_prob`` encodes the belief that opening or closing the window should cause the probability of infection to change.

A causal specification is simply the combination of these components: a series of requirements for the scenario-under-test and a causal DAG representing causality amongst the inputs and outputs. We will discuss these in more detail below.

Scenario Requirements
---------------------

Each scenario is defined as a series of constraints placed over a set of input variables. A constraint is simply a mapping from an input variable to a specific value or distribution that characterises the scenario in question. For example, a scenario simulating the spread of a virus in a crowded indoors space would likely place a constraint over the size of room, the number of windows, and the number of people in the room.

Requirements for this scenario should describe how a particular intervention (e.g. opening the window, changing the number of people, changing the size of the room etc.) is expected to cause a particular outcome (number of infections, deaths, R0 etc.) to change. The way these requirements are expressed is up to the user, however, it is essential that they focus on the expected effect of an intervention.

Causal DAG
----------

In order to apply CI techniques, we need to capture causality amongst the inputs and outputs in the scenario-under-test. Therefore, for each scenario, the user must define a causal DAG. While there is generally no guidance/algorithm that can be followed to create a causal DAG, there are a couple of requirements that should be satisfied.


#. The DAG must contain all inputs and outputs involved in the scenario-under-test.
#. If there are any other variables which are not directly involved but are expected to have a causal relationship with the variables in the scenario-under-test, these should also be added to the graph. For example, the size of the room might be partially caused by the simulated location (house styles, average wealth etc.), in which case location should be added to the DAG with an edge to room size and any other variables it is deemed to influence.
#. If in doubt, add an edge. It is a stronger assumption to exclude an edge (X and Y are independent) than to include one (X has some potentially negligiable causal effect on Y).

Collectively, the components of the causal specification provide both contextual information in the form of constraints and requirements, as well as causal information in the form of a causal DAG. Later on, these components will be used to design statistical experiments that can answer causal questions about the scenrio-under-test, such as ``Does opening a window impair the viruses ability to spread?``
