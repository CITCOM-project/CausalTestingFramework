# Define the Python version neded for CTF
FROM python:3.12-slim

# Install core dependencies using PyPi
RUN pip install causal-testing-framework --no-cache-dir

COPY ./docker_entrypoint.sh ./docker_entrypoint.sh
RUN chmod +x ./docker_entrypoint.sh

## Prevents Python from writing pyc files
ENV PYTHONDONTWRITEBYTECODE=1

## Keeps Python from buffering stdout and stderr to avoid the framework
## from crashing without emitting any logs due to buffering
ENV PYTHONUNBUFFERED=1

ENTRYPOINT [ "./docker_entrypoint.sh" ]
