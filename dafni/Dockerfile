# Define the Python version neded for CTF
FROM python:3.10-slim

## Prevents Python from writing pyc files
ENV PYTHONDONTWRITEBYTECODE=1
#
## Keeps Python from buffering stdout and stderr to avoid the framework
## from crashing without emitting any logs due to buffering
ENV PYTHONUNBUFFERED=1

#Label maintainer
LABEL maintainer="Dr. Farhad Allian - The University of Sheffield"

# Create a folder for the source code/outputs
RUN mkdir -p ./causal_testing
RUN mkdir -p ./data/outputs

# Copy the source code and test files from build into the container
COPY --chown=nobody  ../causal_testing ./causal_testing
COPY --chown=nobody ./dafni/main_dafni.py ./
COPY --chown=nobody ./dafni/data/inputs ./data/inputs

# Install core dependencies using PyPi
RUN pip install causal-testing-framework --no-cache-dir

#For local testing purposes
ENV VARIABLES_PATH=./data/inputs/variables.json \
    CAUSAL_TESTS=./data/inputs/causal_tests.json \
    DATA_PATH=./data/inputs/runtime_data.csv \
    DAG_PATH=./data/inputs/dag.dot

# Define the entrypoint/commands
CMD python main_dafni.py --variables_path $VARIABLES_PATH --dag_path $DAG_PATH --data_path $DATA_PATH --tests_path $CAUSAL_TESTS
