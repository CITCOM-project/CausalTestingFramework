# Causal Testing Framework on DAFNI

- This directory contains the containerisation files of the causal testing framework using Docker, which is used
  to upload the framework onto [DAFNI](https://www.dafni.ac.uk).
- It is **not** recommended to install the causal testing framework using Docker, and should only be installed
  using [conda-forge](https://anaconda.org/channels/conda-forge/packages/causal-testing-framework/overview) or [PyPI](https://pypi.org/project/causal-testing-framework/).

### Directory Hierarchy

- `data` contains two folders (structure is critical for DAFNI workflows):
  - `inputs` contains all input files that are uploaded to DAFNI.
    - `causal_tests.json` is a JSON file containing generated causal tests. If it exists, the framework can automatically run tests without regenerating.
    - `dag.dot` is a DOT file defining the directed acyclic graph (DAG). Causal variables are stored in node metadata as key-value pairs using the syntax:  
      `node [datatype="int", typestring="input"]`  
      - `datatype` specifies the variable's data type (e.g., `"int"`, `"str"`).  
      - `typestring` specifies whether the variable is an `"input"` or `"output"`.
    - `runtime_data.csv` contains the input runtime data for testing.
  - `outputs` is the folder where `causal_test_results.json` is created after running tests.

### Workflow

- The `entrypoint.sh` script now supports auto-detection:  
  - If `causal_tests.json` exists in `data/inputs`, the script automatically runs the test mode.  
  - If it does not exist, the script generates the causal tests first.  
  - The user can still override this behaviour by setting `EXECUTION_MODE` explicitly in the `.env` file.
- Filenames for `causal_tests.json` and `causal_test_results.json` are now configurable through environment variables (`CAUSAL_TESTS` and `CAUSAL_TEST_RESULTS`) in the `.env` file.
- Input/output directories are fixed as `data/inputs` and `data/outputs` to comply with DAFNI requirements.
- The script now reads all configuration parameters (estimator, effect type, threads, verbosity, query filters, adequacy metrics, etc.) **from the `.env` file**, keeping the Docker image and container clean and flexible.

### Docker files

- `model_definition.yaml` is the model metadata required for DAFNI.
- `Dockerfile` builds the container image and uses `entrypoint.sh` as the main entrypoint. All paths and options are now configurable via `.env`.
- `docker-compose.yaml` allows building and running the container with a single command. The `.env` file is required here to define all environment variables.
- `.dockerignore` specifies files to exclude from the Docker image.
- `.env` provides all configurable environment variables for the workflow (execution mode, filenames, estimator options, DAG/effects configuration, and runtime options). This is only needed if using `docker-compose`.