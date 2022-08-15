# Causal Testing Framework: A Causal Inference-Driven Software Testing Framework

 ![example workflow](https://github.com/CITCOM-project/CausalTestingFramework/actions/workflows/ci-tests.yaml/badge.svg) [![codecov](https://codecov.io/gh/CITCOM-project/CausalTestingFramework/branch/main/graph/badge.svg?token=04ijFVrb4a)](https://codecov.io/gh/CITCOM-project/CausalTestingFramework) [![Documentation Status](https://readthedocs.org/projects/causal-testing-framework/badge/?version=latest)](https://causal-testing-framework.readthedocs.io/en/latest/?badge=latest)

Causal testing is a causal inference-driven framework for functional black-box testing. This framework utilises graphical causal inference (CI) techniques for the specification and functional testing of software from a black-box perspective. In this framework, we use causal directed acyclic graphs (DAGs) to express the anticipated cause-effect relationships amongst the inputs and outputs of the system-under-test and the supporting mathematical framework to design statistical procedures capable of making causal inferences. Each causal test case focuses on the causal effect of an intervention made to the system-under test. That is, a prescribed change to the input configuration of the system-under-test that is expected to cause a change to some output(s).

![Causal Testing Workflow](images/workflow.png)

The causal testing framework has three core components:

1. [Causal specification](causal_testing/specification/README.md): Before we can test software, we need to obtain an understanding of how it should behave in a particular use-case scenario. In addition, to apply graphical CI techniques for testing, we need a causal DAG which depicts causal relationships amongst inputs and outputs. To collect this information, users must create a _causal specification_. This comprises a set of scenarios which place constraints over input variables that capture the use-case of interest, a causal DAG corresponding to this scenario, and a series of high-level functional requirements that the user wishes to test. In causal testing, these requirements should describe how the model should respond to interventions (changes made to the input configuration).

2. [Causal tests](causal_testing/testing/README.md): With a causal specification in hand, we can now go about designing a series of test cases that interrogate the causal relationships of interest in the scenario-under-test. Informally, a causal test case is a triple (M, X, Delta, Y), where M is the modelling scenario, X is an input configuration, Delta is an intervention which should be applied to X, and Y is the expected _causal effect_ of that intervention on some output of interest. Therefore, a causal test case states the expected causal effect (Y) of a particular intervention (Delta) made to an input configuration (X). For each scenario, the user should create a suite of causal tests. Once a causal test case has been defined, it is executed as follows:
    1. Using the causal DAG, identify an estimand for the effect of the intervention on the output of interest. That is, a statistical procedure capable of estimating the causal effect of the intervention on the output.
    2. Collect the data to which the statistical procedure will be applied (see Data collection below).
    3. Apply a statistical model (e.g. linear regression or causal forest) to the data to obtain a point estimate for the causal effect. Depending on the estimator used, confidence intervals may also be obtained at a specified confidence level e.g. 0.05 corresponds to 95% confidence intervals (optional).
    4. Return the casual test result including a point estimate and 95% confidence intervals, usally quantifying the average treatment effect (ATE).
    5. Implement and apply a test oracle to the causal test result - that is, a procedure that determines whether the test should pass or fail based on the results. In the simplest case, this takes the form of an assertion which compares the point estimate to the expected causal effect specified in the causal test case.

3. [Data collection](causal_testing/data_collection/README.md): Data for the system-under-test can be collected in two ways: experimentally or observationally. The former involves executing the system-under-test under controlled conditions which, by design, isolate the causal effect of interest (accurate but expensive), while the latter involves collecting suitable previous execution data and utilising our causal knowledge to draw causal inferences (potentially less accurate but efficient). To collect experimental data, the user must implement a single method which runs the system-under-test with a given input configuration. On the other hand, when dealing with observational data, we automatically check whether the data is suitable for the identified estimand in two steps. First, confirm whether the data contains a column for each variable in the causal DAG. Second, we check for [positivity violations](https://www.youtube.com/watch?v=4xc8VkrF98w). If there are positivity violations, we can provide instructions for an execution that will fill the gap (future work).

For more information on each of these steps, follow the link to their respective documentation.

## Causal Inference Terminology

Here are some explanations for the causal inference terminology used above.

- Causal inference (CI) is a family of statistical techniques designed to quantify and establish **causal** relationships in data. In contrast to purely statistical techniques that are driven by associations in data, CI incorporates knowledge about the data-generating mechanisms behind relationships in data to derive causal conclusions.
- One of the key advantages of CI is that it is possible to answer causal questions using **observational data**. That is, data which has been passively observed rather than collected from an experiment and, therefore, may contain all kinds of bias. In a testing context, we would like to leverage this advantage to test causal relationships in software without having to run costly experiments.
- There are many forms of CI techniques with slightly different aims, but in this framework we focus on graphical CI techniques that use directed acyclic graphs to obtain causal estimates. These approaches used a causal DAG to explain the causal relationships that exist in data and, based on the structure of this graph, design statistical experiments capable of estimating the causal effect of a particular intervention or action, such as taking a drug or changing the value of an input variable.

## Installation

To use the causal testing framework, clone the repository, `cd` into the root directory, and run `pip install -e .`. More detailled installation instructions can be found in the [online documentation](https://causal-testing-framework.readthedocs.io/en/latest/installation.html).

## Usage

There are currently two methods of using the Causal Testing Framework, through the [JSON Front End](https://causal-testing-framework.readthedocs.io/en/latest/json_front_end.html) or directly as described below.

The causal testing framework is made up of three main components: Specification, Testing, and Data Collection. The first step is to specify the (part of the) system under test as a modelling `Scenario`. Modelling scenarios specify the observable variables and any constraints which exist between them. We currently support three types of variable:

- `Input` variables are input parameters to the system.
- `Output` variables are outputs from the system.
- `Meta` variables are not directly observable but are relevant to system testing, e.g. a model may take a `location` parameter and expand this out into `average_age` and `household_size` variables "under the hood". These parameters can be made explicit by instantiating them as metavariables.

To instantiate a scenario, simply provide a set of variables and an optional set of constraints, e.g.

```{python}
from causal_testing.specification.variable import Input, Output, Meta
from causal_testing.specification.scenario import Scenario

x = Input("x", int)  # Define an input with name "x" of type int
y = Output("y", float)  # Define an output with name "y" of type float
z = Meta("y", int)  # Define a meta with name "z" of type int

modelling_scenario = Scenario({x, y, z}, {x > z, z < 3})  # Define a scenario with the three variables and two constraints
```

Note that scenario constraints are primarily intended to help specify the region of the input space under test in a manner consistent with the Category Partition Method. It is not intended to serve as a test oracle. Use constraints sparingly and with caution to avoid introducing data selection bias. We use Z3 to handle constraints. For help with this, check out [their documentation](https://ericpony.github.io/z3py-tutorial/guide-examples.htm).

Having fully specified the modelling scenario, we are now ready to test. Causal tests are, essentially [metamorphic tests](https://en.wikipedia.org/wiki/Metamorphic_testing) which are executed using statistical causal inference. A causal test expresses the change in a given output that we expect to see when we change a particular input in a particular way, e.g.

```{python}
from causal_testing.testing.causal_test_case import CausalTestCase
from causal_testing.testing.causal_test_outcome import Positive

causal_test_case = CausalTestCase(
    control_input_configuration={x: 0},  # Our unmodified (or 'control') configuration in which input x is 0
    treatment_input_configuration={x: 1}, # Our modified (or 'treatment') configuration in which input x is 1
    expected_causal_effect=Positive,  # We expect to see a positive change as a result of this
    outcome_variables=[y],  # We expect to observe that positive change in variable y
)
```

Before we can run our test case, we first need data. There are two ways to acquire this: 1. run the model with the specific input configurations we're interested in, 2. use data from previous model runs. For a small number of specific tests where accuracy is critical, the first approach will yield the best results. To do this, you need to instantiate the `ExperimentalDataCollector` class.

Where there are many test cases using pre-existing data is likely to be faster. If the program's behaviour can be estimated statistically, the results should still be reliable as long as there is enough data for the estimator to work as intended. This will vary depending on the program and the estimator. To use this method, simply instantiate the `ObservationalDataCollector` class with the modelling scenario and a path to the CSV file containing the runtime data, e.g.

```{python}
data_csv_path = 'results/data.csv'
data_collector = ObservationalDataCollector(modelling_scenario, data_csv_path)
```

The actual running of the tests is done using the `CausalTestEngine` class. This is still a work in progress and may change in the future to improve ease of use, but currently proceeds as follows.

```{python}
causal_test_engine = CausalTestEngine(causal_test_case, causal_specification, data_collector)  # Instantiate the causal test engine
minimal_adjustment_set = causal_test_engine.load_data(data_csv_path, index_col=0)  # Calculate the adjustment set
treatment_vars = list(causal_test_case.treatment_input_configuration)
minimal_adjustment_set = minimal_adjustment_set - set([v.name for v in treatment_vars])  # Remove the treatment variables from the adjustment set. This is necessary for causal inference to work properly.
```

Whether using fresh or pre-existing data, a key aspect of causal inference is estimation. To actually execute a test, we need an estimator. We currently support two estimators: linear regression and causal forest. These can simply be instantiated as per the [documentation](https://causal-testing-framework.readthedocs.io/en/latest/autoapi/causal_testing/testing/estimators/index.html).

```{python}
from causal_testing.testing.estimators import LinearRegressionEstimator
estimation_model = LinearRegressionEstimator("x",), 0, 1, minimal_adjustment_set, ("y",), causal_test_engine.scenario_execution_data_df)
```

We can now execute the test using the estimation model. This returns a causal test result, from which we can extract various information. Here, we simply assert that the observed result is (on average) what we expect to see.

```{python}
causal_test_result = causal_test_engine.execute_test(estimation_model)
test_passes = causal_test_case.expected_causal_effect.apply(causal_test_result)
assert test_passes, "Expected to see a positive change in y."
```
