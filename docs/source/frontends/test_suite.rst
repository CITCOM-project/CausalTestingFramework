.. module:: causal_testing
Test Suite
======================================
The test_suite feature allows for the effective running of multiple causal_test_cases using a logical structure.
This structure is defined by the parameters in the class: :class:`causal_testing.testing.causal_test_suite`.

A current limitation of the Test Suite is that it requires references to the estimator class, not instances (objects) of
estimator classes, which prevents the usage of some of the features of an estimator.

The test_suite class is an extension of the Python UserDict_, meaning it simulates a standard Python dictionary where
any dictionary method can be used. The class also features a setter to make adding new test cases quicker and less
error prone :meth:`causal_testing.testing.causal_test_suite.CausalTestSuite.add_test_object`.

The suite's dictionary structure is at the top level a :class:`causal_testing.testing.base_test_case` as the key and
the value is a test object in the format of another dictionary:

.. code-block:: python

    test_object = {"tests": causal_test_case_list, "estimators": estimators_classes, "estimate_type": estimate_type}

Each ``base_test_case`` contains the treatment and outcome variables, and only causal_test_cases testing this relationship
should be placed in the test object for that ``base_test_case``.


Following this, users can similarly execute a suite of causal tests and return the results in a list by executing the
class's :meth:`causal_testing.testing.causal_test_suite.CausalTestSuite.execute_test_suite` method.



