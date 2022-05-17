# JSON Causal Testing Framework Frontend

The JSON frontend allows Causal Tests and parameters to be specified in JSON to allow for tests to be quickly written
whilst retaining the flexibility of the Causal Testing Framework (CTF). 

An example is provided in `examples/poisson` which will be walked through in this README to better understand
the framework


`process_causal_tests_json.py` contains the code to read and parse the user generated JSON and custom python functions and run them as tests
 on CTF. This file should not need editing between scenarios

`examples/poisson/causal_test_setup.py` contains python code written by the user to implement scenario specific features
such as:
1. Custom Estimators
2. Causal test case outcomes
3. Meta constraint functions
4. Mapping JSON distributions, effects, and estimators to python objects

`examples/poisson/causal_tests.json` is the JSON file that allows for the easy specification of multiple causal tests
it requires:
1. Input, Output, and Meta causal variables
2. A list of constraints
3. Any tests to be executed

To run the JSON frontend example, use `python json_frontend/process_causal_tests_json.py --directory_path="json_frontend/examples/poisson"`

A failure flag `-f` can be specified to stop the framework running if a test is failed
`python json_frontend/process_causal_tests_json.py -f --directory_path="json_frontend/examples/poisson"`

