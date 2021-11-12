# Causal Testing: A Causal Inference-Driven Framework for Functional Black-Box Testing

This repository contains the framework for causal testing, a causal inference-driven framework for functional black-box testing. But what does this mean?

- Causal inference (CI) is a family of statistical techniques designed to quanitfy and establish causal relationships in data. There are many forms of CI techniques with slightly different aims, but in this framework we focus on graphical CI techniques that use directed acyclic graphs to obtain causal estimates. These approaches used a causal DAG to explain the causal relationships that exist in data and, based on the structure of this graph, design statistical experiments capable of estimating the causal effect of a particular intervention or action, such as taking a drug or changing the value of an input variable.

- Functional black-box testing is form of software testing that focuses on testing functional requirements (i.e. how the system-under-test should behave) without considering the inner-workings of the system-under-test. Instead, we focus on the relationships that exist amongst the input and output variable of the system-under-test.

This framework utilises existing graphical CI techniques for the specification and testing of complex software from a black-box perspective. Specifically, we use causal DAGs to express the anticipated cause-effect relationships amongst the inputs and outputs of the system-under-test and the accomanying mathematical techniques to design statistical test cases capable of making causal inferences. Each causal test case focuses on the effect of an intervention made to the system-under test. That is, a prescribed change to the input configuration of the system-under-test that is expected to cause a change to some subset of outputs. 

The causal testing framework has three core components:

1. Causal specification: To apply CI to test software, we need some functional requirements to test and a causal DAG which captures causal relationships amongst inputs and outputs.
2. Data collection: Next, we need data from the system-under-test corresponding to the functional requirements outlined in the causal specification.
3. Causal tests: With this information, we can apply CI to the causal DAG to design a statistical experiment which can yield a causal effect. Then, we can implement this experiment on the collected data with appropriate statistical methods to yield a causal estimate. Finally, the causal estimate is compared against the expectation to determine whether the test passes or fails.

For more information on each of these steps, follow the link to their respective documentation.
