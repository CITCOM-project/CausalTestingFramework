Causal Testing
==============

This quick start guide shows how to carry set up and run causal tests using the command line interface.
This is the simplest way to interact with the Causal Testing Framework as you do not need to write any code.
The result will be a JSON file containing a set of pass/fail test outcomes for each of the specified causal relationships that you can then use to explore the system.

Step 1: Prepare the DAG
-----------------------

The first step is to specify the expected causal relationships between your variables using a directed acyclic graph (DAG).
To do this, we use the `DOT language <https://graphviz.org/docs/layouts/dot>`_, which provides an intuitive text-based representation of DAGs.
The syntax is very lightweight: an edge from node :code:`X` to :code:`Y` is specified as :code:`X -> Y;`.
If you would prefer to use a visual editor, you can use `Dagitty <https://dagitty.net/dags.html>`_ and copy the *Model code*.

A simple example is shown below.
The first line specifies that the graph is a :code:`digraph` (directed graph), and names it :code:`expected_relationships`.
The next three lines list the variables :code:`X`, :code:`Y`, and :code:`Z`.
The next line lists a single edge :code:`X -> Y`, indicating that :code:`X` should cause :code:`Y`.
:code:`Z` has no incoming or outgoing edges, so should be independent of both :code:`X` and :code:`Y`.

.. code-block:: graphviz

   digraph expected_relationships {
    X;
    Y;
    Z;

    X -> Y;
   }

Step 2: Prepare the Causal Test Cases
-------------------------------------

Having prepared the causal DAG, you can then use the Causal Testing Framework automatically convert the specified causal relationships to test cases.
Each DAG implicitly encodes two types of causal relationship: causal dependences (i.e. the edges of the DAG) and causal *in*\ dependences (i.e. the non-edges) of the graph.
Assuming you have saved your DAG in a text file called :code:`dag.dot` in the current working directory, you can generate the corresponding causal test cases using the following command from your command shell::

  causal-testing generate --dag-path dag.dot -output tests.json

This will output a JSON file containing the causal test cases to :code:`tests.json`.
While these test cases are executable "out of the box", they can be fully customised to suit your needs.
To do this, you can either manually edit :code:`tests.json` (be careful as your changes will be overwritten if you regenerate the test cases), or by providing additional configuration options to the :code:`generate` command above (run :code:`causal-testing generate --help` to see the full list).

Step 3: Prepare the Test Data
-----------------------------

Causal test cases are evaluated statistically with respect to a *set* of system executions.
This set of executions is specified as a table of values in which each column corresponds to a variable in your DAG, and each row represents a valuation of those variables for a single run of the system.
For an example, check out our interactive :doc:`tutorial <../tutorials/vaccinating_elderly/vaccinating_elderly_tutorial>`.

A major strength of the Causal Testing Framework is that the specification of the expected causal relationships is completely separate from the collection of test data, so you can evaluate the same tests on multiple different datasets with very little additional effort.
This means that, if you already have some data from previous runs of the system, you can get to testing straight away without needing to run system under test again.
The framework supports `several file formats <https://pandas.pydata.org/pandas-docs/stable/user_guide/io.html>`_, including CSV, excel, parquet, and even HTML.

Step 4: Evaluate the Test Cases
-------------------------------

We now have everything we need to evaluate the causal tests: the DAG, the data, and the test cases themselves.
Assuming your data is saved in a single CSV file called :code:`data.csv`, you can execute your causal tests with the following command::

  causal-testing test --dag dag.dot --data-paths data.csv --test-config tests.json --output test_results.json

This will execute your causal test cases and produce a file called :code:`test_results.json` that will contain your causal test results.
There are various configuration options at this stage.
Run :code:`causal-testing test --help` to see them all.
