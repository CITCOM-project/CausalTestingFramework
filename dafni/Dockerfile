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

# Copy the source code and test files from build into the container
COPY --chown=nobody ./causal_testing /usr/src/app/
COPY --chown=nobody ./dafni/inputs /usr/src/app/inputs/
COPY --chown=nobody ./dafni/main_dafni.py /usr/src/app/

# Change the working directory
WORKDIR /usr/src/app/

# Install core dependencies using PyPi
RUN pip install causal-testing-framework --no-cache-dir

# Use the necessaary environment variables for the script's inputs
ENV VARIABLES=./inputs/variables.json \
    CAUSAL_TESTS=./inputs/causal_tests.json \
    DATA_PATH=./inputs/simulated_data.csv \
    DAG_PATH=./inputs/dag.dot

# Define the entrypoint/commands
CMD python main_dafni.py --variables_path $VARIABLES_PATH --dag_path $DAG_PATH --data_path $DATA_PATH --tests_path $CAUSAL_TESTS
