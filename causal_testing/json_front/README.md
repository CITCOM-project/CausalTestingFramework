# JSON Causal Testing Framework Frontend

The JSON frontend allows Causal Tests and parameters to be specified in JSON to allow for tests to be quickly written
whilst retaining the flexibility of the Causal Testing Framework (CTF). 

An example is provided in `examples/poisson` which will be walked through in this README to better understand
the framework

`examples/poisson/causal_test_setup.py` contains python code written by the user to implement scenario specific features
such as:
1. Custom Estimators
2. Causal Variable specification
3. Causal test case outcomes
4. Meta constraint functions
5. Mapping JSON distributions, effects, and estimators to python objects

`examples/poisson/causal_tests.json` is the JSON file that allows for the easy specification of multiple causal tests.
Each test requires:
1. Test name
2. Mutations
3. Estimator
4. Estimate_type
5. Effect modifiers
6. Expected effects
7. Skip: boolean that if set true the test won't be executed and will be skipped

To run the JSON frontend example from the root directory of the project, use 
`python examples/poisson/causal_test_setup.py --directory_path="examples/poisson"`

A failure flag `-f` can be specified to stop the framework running if a test is failed
`python examples/poisson/causal_test_setup.py -f --directory_path="examples/poisson"`

