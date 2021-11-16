# Causal Testing: A Causal Inference-Driven Framework for Functional Black-Box Testing

Causal testing is a causal inference-driven framework for functional black-box testing. But what does this mean?

![Causal Testing Workflow](images/workflow.pdf)
## Causal Inference
- Causal inference (CI) is a family of statistical techniques designed to quanitfy and establish **causal** relationships in data. In contrast to purely statistical techniques that are driven by associations in data, CI encorporates knowledge about the data-generating mechanisms behind relationships in data to derive causal conclusions. 
- One of the key advantages of CI is that it is possible to answer causal questions using **observational data**. That is, data which has been passively observed rather than collected from an experiment and, therefore, may contain all kinds of bias. In a testing context, we would like to leverage this advantage to test causal relationships in software without having to run costly experiments.
- There are many forms of CI techniques with slightly different aims, but in this framework we focus on graphical CI techniques that use directed acyclic graphs to obtain causal estimates. These approaches used a causal DAG to explain the causal relationships that exist in data and, based on the structure of this graph, design statistical experiments capable of estimating the causal effect of a particular intervention or action, such as taking a drug or changing the value of an input variable.

## Testing
- Functional testing is form of software testing that focuses on testing functional requirements (i.e. how the system-under-test should behave).
- Black-box testing if any form of testing that treats the system-under-test as a black-box, focusing on the inputs and outputs instead of the inner-workings.

## Causal Testing
This framework utilises existing graphical CI techniques for the specification and functional testing of software from a black-box perspective. Specifically, we use causal DAGs to express the anticipated cause-effect relationships amongst the inputs and outputs of the system-under-test and the accomanying mathematical techniques to design statistical experiments capable of making causal inferences. Each causal test case focuses on the causal effect of an intervention made to the system-under test. That is, a prescribed change to the input configuration of the system-under-test that is expected to cause a change to some subset of outputs. 

The causal testing framework has three core components:

1. [Causal specification](causal_testing/specification/README.md): To apply CI to test software, we need some functional requirements to test and a causal DAG which captures causal relationships amongst inputs and outputs. These functional requirments should describe how particular interventions are expected to cause a change to the behaviour of the system-under-test. For example, for the function `y=x^2 + 10`, increasing the input `x` from 2 to 4 should increase the output `y` from 14 to 26.
2. [Data collection](causal_testing/data_collection/README.md): Next, we need data from the system-under-test corresponding to the functional requirements outlined in the causal specification. Broadly speaking, we can collect this in two different ways: experimentally or observationally. The former involves executing the system-under-test under controlled conditions which, by design, isolate the causal effect of interest (accurate but expensive), while the latter involves collecting suitable previous execution data and utilising our causal knowledge to draw causal inferences (potentially less accurate but efficient).
3. [Causal tests](causal_testing/testing/README.md): With this information, we can apply CI techniques in two steps: identification and estimation.
  - Identification involves analysing the structure of our causal DAG to identify any sources of bias that must be controlled for to reach a causal conclusion. The outcome of identification is an estimand which describes a statistical procedure capable of isolating the causal effect of interest (assuming the assumptions made in the causal DAG are correct). 
  - Estimation is the process of applying conventional statistical models and methods to estimate the identified estimand. The outcome of estimation is a causal estimate - that is, the expected change caused by the intervention on the outcome of interest. In addition, some estimators can produce confidence intervals which gauge the range in which the true estimate is most likely to lie (narrower is better). The causal estimate and/or the confidence intervals are then used to ascertain the correctness of the system-under-test (test oracle procedure).

For more information on each of these steps, follow the link to their respective documentation.
