JSON Frontend
======================================
The JSON frontend allows Causal Tests and parameters to be specified in JSON to allow for tests to be quickly written
whilst retaining the flexibility of the Causal Testing Framework (CTF).

basic workflow
--------------
The basic workflow of using the JSON frontend is as follows:

#. Specify your test cases in the JSON format (more details below)
#. Create your DAG in a dot file
#. Initialise the JsonUtility class in python with a path of where you want the outputs saved
#. Set the paths pointing the Json class to your json file, dag file and optionally your data file (see data section below) using the :func:`causal_testing.json_front.json_class.JsonUtility.set_paths` method
#. Run the :func:`causal_testing.json_front.json_class.JsonUtility.setup` method providing your scenario
#. Run the :func:`causal_testing.json_front.json_class.JsonUtility.run_json_tests` method, which will execute the test cases provided by the JSON file.

Example Walkthrough
-------------------
An example is provided in `examples/poisson` which will be walked through in this README to better understand
the framework

run_causal_tests.py
*******************
`examples/poisson/run_causal_tests.py <https://github.com/CITCOM-project/CausalTestingFramework/blob/main/examples/poisson/run_causal_tests.py>`_
contains python code written by the user to implement scenario specific features
such as:

#. Custom Estimators
#. Causal Variable specification
#. Causal test case outcomes
#. Meta constraint functions
#. Mapping JSON distributions, effects, and estimators to python objects

Use case specific information is also declared here such as the paths to the relevant files needed for the tests.

causal_tests.json
*****************
`examples/poisson/causal_tests.json <https://github.c#om/CITCOM-project/CausalTestingFramework/blob/main/examples/poisson/causal_tests.json>`_ contains python code written by the user to implement scenario specific features
is the JSON file that allows for the easy specification of multiple causal tests. Tests can be specified two ways; firstly by specifying a mutation lke in the example tests with the following structure:
Each test requires:

#. name
#. mutations
#. estimator
#. estimate_type
#. effect_modifiers
#. expected_effects
#. skip: boolean that if set true the test won't be executed and will be skipped

The second method of specifying a test is to specify the test in a concrete form with the following structure:

#. name
#. treatment_variable
#. control_value
#. treatment_value
#. estimator
#. estimate_type
#. expected_effect
#. skip

Run Commands
************
This example uses the Argparse utility built into the JSON frontend, which allows the frontend to be run from a commandline interface as shown here.

To run the JSON frontend example from the root directory of the project, use::

    python examples\poisson\run_causal_tests.py --data_path="examples\poisson\data.csv" --dag_path="examples\poisson\dag.dot" --json_path="examples\poisson\causal_tests.json

A failure flag `-f` can be specified to stop the framework running if a test is failed::

    python examples\poisson\run_causal_tests.py -f --data_path="examples\poisson\data.csv" --dag_path="examples\poisson\dag.dot" --json_path="examples\poisson\causal_tests.json

There are two main outputs of this frontend, both are controlled by the logging module. Firstly outputs are printed to stdout (terminal).
Secondly a log file is produced, by default a file called `json_frontend.log` is produced in the directory the script is called from.

The behaviour of where the log file is produced and named can be altered with the --log_path argument::

    python examples\poisson\run_causal_tests.py -f --data_path="examples\poisson\data.csv" --dag_path="examples\poisson\dag.dot" --json_path="examples\poisson\causal_tests.json --log_path="example_directory\logname.log"


Runtime Data
----------

There are currently 2 methods to inputting your runtime data into the JSON frontend:

#. Providing one or more file paths to `.csv` files containing your data
#. Setting a dataframe to the .data attribute of the JsonUtility instance, this must be done before the setup method is called.




