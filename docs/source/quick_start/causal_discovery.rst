Causal Discovery
================

This quick start guide shows how to carry out causal discovery using the commandline interface.
The result will be a causal DAG that shows the causal relationships within the system.

Step 1: Prepare the Data
------------------------

This is the exact same process as step 3 in the :doc:`causal testing quick start guide <causal_testing>`.
To perform causal discovery, you will need to collect form a table of values in which each column will correspond to a variable in your DAG, and each row represents a valuation of those variables for a single run of the system.
This need not be collected specially --- you can easily use pre-existing data if you have it.

Step 2: Run the Causal Discovery
--------------------------------

The data is all you need to run causal discovery.
Assuming your data is saved in a single CSV file called :code:`data.csv`, you can run causal discovery with the following command::

  causal-testing discover --data paths data.csv --output dag.dot

This will create a `textual representation <https://graphviz.org/docs/layouts/dot>`_ of the inferred DAG in a file called :code:`dag.dot` in the same format used for causal testing.
There are various configuration options available here, including the causal discovery technique used, and options to provide known relationships and independences.
For full details, run :code:`causal-testing discover --help`.

.. note::
   There may be more than one DAG that explains a given dataset.
   For best results, we recommend generating multiple DAGs with different random seeds and examining commonalities between them.

.. warning::
   Causal discovery should **not** be seen as an easy way to create specifications for causal testing!
   While causal discovery can be very to understand the causal relationships between variables in a system, it critical to check that the inferred relationships are **sensible and meaningful**.

Step 3: Evaluate your DAG
-------------------------

The construction of a DAG is an iterative process.
As with any data-driven technique, the output of causal discovery is entirely dependent on the data that the algorithm is given.
If the dataset is small, the resulting DAG may be overfitted to the dataset, meaning the corresponding test outcomes are highly dependent on a few individual points.
To help mitigate this risk, we provide an evaluation function to help you investigate this::

    causal-testing evaluate --data paths data.csv --dag-path dag.dot --output output.csv

This will iteratively resample the data and evaluate causal tests to check the presence and absence of the causal relationships that the DAG specifies.
The result will be a CSV file saved to :code:`output.csv` which gives confidence intervals for the number of passing, failing, and :term:`inestimable` tests, as well as the results with the full dataset.
Narrower confidence intervals indicate that the DAG gives *stable* test outcomes, and so is less likely to be overfitted to the dataset.
