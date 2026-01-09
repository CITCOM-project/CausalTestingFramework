#!/bin/sh
set -e

INPUT_DIR="data/inputs"
OUTPUT_DIR="data/outputs"

# --------------------------
# DEBUG ENVIRONMENT VARIABLES
# --------------------------
echo "=== DEBUG: Environment Variables ==="
echo "ADEQUACY: '$ADEQUACY'"
echo "BOOTSTRAP_SIZE: '$BOOTSTRAP_SIZE'"
echo "EXECUTION_MODE: '$EXECUTION_MODE'"
echo "VERBOSE: '$VERBOSE'"
echo "===================================="

# --------------------------
# Discover the DAGs first (.dot)
# --------------------------
DAG_FILES=$(find "$INPUT_DIR/dag-data" -name "*.dot" 2>/dev/null || true)

if [ -z "$DAG_FILES" ]; then
    echo "ERROR: No .dot DAG file found in $INPUT_DIR/dag-data"
    exit 1
fi

set -- $DAG_FILES
if [ "$#" -ne 1 ]; then
    echo "ERROR: Expected exactly one DAG (.dot) file, found $#"
    exit 1
fi

DAG_PATH="$1"
echo "Using DAG file: $DAG_PATH"

# --------------------------
# Discover runtime CSV data (can be multiple)
# --------------------------
DATA_PATHS=$(find "$INPUT_DIR/runtime-data" -name "*.csv" 2>/dev/null || true)

if [ -z "$DATA_PATHS" ]; then
    echo "ERROR: No runtime CSV files found in $INPUT_DIR/runtime-data"
    exit 1
fi

echo "Found CSV files: $DATA_PATHS"

# Causal tests path for checking if it exists in inputs
CAUSAL_TESTS_INPUT_PATH="$INPUT_DIR/causal-tests/$CAUSAL_TESTS"
# Causal tests path for writing (generate mode)
CAUSAL_TESTS_OUTPUT_PATH="$OUTPUT_DIR/$CAUSAL_TESTS"
# Results path (test mode)
CAUSAL_TEST_RESULTS_PATH="$OUTPUT_DIR/$CAUSAL_TEST_RESULTS"

# --------------------------
# Auto-detect mode
# --------------------------
if [ "$EXECUTION_MODE" = "auto" ]; then
    if [ -d "$INPUT_DIR/causal-tests" ] && [ -f "$CAUSAL_TESTS_INPUT_PATH" ]; then
        EXECUTION_MODE="test"
        echo "Auto mode: causal tests found in inputs -> running TEST mode"
    else
        EXECUTION_MODE="generate"
        echo "Auto mode: no causal tests found in inputs -> running GENERATE mode"
    fi
else
    echo "Execution mode explicitly set to: $EXECUTION_MODE"
fi

# --------------------------
# Generate mode
# --------------------------
if [ "$EXECUTION_MODE" = "generate" ]; then
    echo "Running causal_testing GENERATE..."
    echo "Will write causal tests to: $CAUSAL_TESTS_OUTPUT_PATH"

    python -m causal_testing generate \
        -D "$DAG_PATH" \
        -o "$CAUSAL_TESTS_OUTPUT_PATH" \
        -e "$ESTIMATOR" \
        -T "$EFFECT_TYPE" \
        -E "$ESTIMATE_TYPE" \
        -t "$THREADS" \
        $([ "$IGNORE_CYCLES" = "true" ] && echo "-i")

# --------------------------
# Test mode
# --------------------------
elif [ "$EXECUTION_MODE" = "test" ]; then
    if [ ! -f "$CAUSAL_TESTS_INPUT_PATH" ]; then
        echo "ERROR: Causal tests file not found at $CAUSAL_TESTS_INPUT_PATH"
        exit 1
    fi

    echo "Running causal_testing TEST..."
    echo "Using causal tests from: $CAUSAL_TESTS_INPUT_PATH"

    # DEBUG: Show which branch we're taking
    echo "=== DEBUG: Adequacy Check ==="
    if [ "$ADEQUACY" = "true" ]; then
        echo "ADEQUACY is TRUE - will pass -a -b $BOOTSTRAP_SIZE"
    else
        echo "ADEQUACY is FALSE - will NOT pass -a -b flags"
    fi
    echo "============================="

    # Build command with adequacy flags only when ADEQUACY is true
    if [ "$ADEQUACY" = "true" ]; then
        echo "DEBUG: Executing WITH adequacy flags"
        python -m causal_testing test \
            -D "$DAG_PATH" \
            -d $DATA_PATHS \
            -t "$CAUSAL_TESTS_INPUT_PATH" \
            -o "$CAUSAL_TEST_RESULTS_PATH" \
            $([ "$IGNORE_CYCLES" = "true" ] && echo "-i") \
            $([ "$VERBOSE" = "true" ] && echo "-v") \
            $([ -n "$QUERY" ] && [ "$QUERY" != "None" ] && echo "-q '$QUERY'") \
            -a -b $BOOTSTRAP_SIZE \
            $([ "$SILENT" = "true" ] && echo "-s") \
            $([ "$BATCH_SIZE" != "0" ] && echo "--batch-size $BATCH_SIZE")
    else
        echo "DEBUG: Executing WITHOUT adequacy flags"
        python -m causal_testing test \
            -D "$DAG_PATH" \
            -d $DATA_PATHS \
            -t "$CAUSAL_TESTS_INPUT_PATH" \
            -o "$CAUSAL_TEST_RESULTS_PATH" \
            $([ "$IGNORE_CYCLES" = "true" ] && echo "-i") \
            $([ "$VERBOSE" = "true" ] && echo "-v") \
            $([ -n "$QUERY" ] && [ "$QUERY" != "None" ] && echo "-q '$QUERY'") \
            $([ "$SILENT" = "true" ] && echo "-s") \
            $([ "$BATCH_SIZE" != "0" ] && echo "--batch-size $BATCH_SIZE")
    fi
fi

echo "Execution completed successfully"