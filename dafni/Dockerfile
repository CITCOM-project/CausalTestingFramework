FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1

ENV PYTHONUNBUFFERED=1

LABEL maintainer="Dr. Neil Walkinshaw - The University of Sheffield"

WORKDIR /

# Copy source code
COPY causal_testing /causal_testing

# Set PYTHONPATH
ENV PYTHONPATH="/causal_testing:${PYTHONPATH}"

# Install dependencies
RUN pip install --no-cache-dir causal-testing-framework

# Copy entrypoint
COPY dafni/entrypoint.sh /entrypoint.sh

RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]