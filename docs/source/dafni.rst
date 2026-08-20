Using the CTF on DAFNI
======================

The Causal Testing Framework is also available to run on `DAFNI <https://www.dafni.ac.uk/>`_, allowing you to generate causal tests and evaluate causal effects from your input data and DAGs without installing the framework locally. This lets you integrate CTF into workflows with other models or datasets easily.

Data tab
--------

- Upload the required input files as a dataset. Typically, this includes:

  - ``dag.dot`` – the directed acyclic graph defining causal relationships between variables.
  - ``runtime_data.csv`` – the CSV file containing runtime input data.
  - ``causal_tests.json`` – optional; if provided, the framework will run tests directly. Otherwise, tests will be generated automatically.

  **Note:** Input files must remain in the ``data/inputs`` structure; this is required by the workflow.

Workflow tab
-------------

- Select the CTF workflow.
- In the Parameter sets section, click **Create**.
- In the page that opens:

  - Select the model in the workflow (typically ``causal-testing-framework``).
  - Complete the sections at the bottom:

    - **Parameters:** Set or confirm environment variables from the ``.env`` file (e.g., ``EXECUTION_MODE``, ``CAUSAL_TESTS``, ``CAUSAL_TEST_RESULTS``). These control whether tests are generated or executed, the filenames, estimator, effect type, and other runtime options.
    - **Datasets:** Click the icon and select the dataset containing your input files (``dag.dot``, ``runtime_data.csv``, ``causal_tests.json``). All input files will be placed in the required ``data/inputs`` directory when running the workflow.

- Unselect the model if needed, click **Continue**, and complete any required metadata such as the name of the parameter set.

Execute the workflow
----------------------

- Click **Execute workflow with parameter set**.
- If successful, the workflow will either generate ``causal_tests.json`` (if not provided) or run the tests and create ``causal_test_results.json`` in ``data/outputs``.
- After completion, you can view the results in the **Data tab** as a new output dataset.

Customisation and chaining
--------------------------

- You can create additional workflows to customise input parameters, filenames, or estimators.
- Multiple CTF workflows can also be chained to run sequential analyses or to integrate with other models and datasets, combining results for more complex causal testing scenarios.
