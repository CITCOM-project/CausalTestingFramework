#!/bin/sh
set -e

# Auto-detect mode if EXECUTION_MODE=auto
if [ "$EXECUTION_MODE" = "auto" ]; then
    if [ -f "$CAUSAL_TESTS" ]; then
        EXECUTION_MODE="test"
        echo "Auto mode: causal_tests.json found -> running TEST mode"
    else
        EXECUTION_MODE="generate"
        echo "Auto mode: No causal tests found -> running GENERATE mode"
    fi
else
    echo "Execution mode explicitly set to: $EXECUTION_MODE"
fi

# --------------------------
# Generate mode
# --------------------------
if [ "$EXECUTION_MODE" = "generate" ]; then
    echo "Running causal_testing GENERATE..."
    python -m causal_testing generate \
        --dag-path "$DAG_PATH" \
        --output "$CAUSAL_TESTS" \
        --estimator "$ESTIMATOR" \
        --effect-type "$EFFECT_TYPE" \
        --estimate-type "$ESTIMATE_TYPE" \
        $( [ "$IGNORE_CYCLES" = "true" ] && echo "--ignore-cycles" ) \
        --threads "$THREADS"

# --------------------------
# Test mode
# --------------------------
elif [ "$EXECUTION_MODE" = "test" ]; then
    if [ ! -f "$CAUSAL_TESTS" ]; then
        echo "Error: Causal tests file not found at $CAUSAL_TESTS"
        exit 1
    fi

    echo "Running causal_testing TEST..."
    python -m causal_testing test \
        --dag-path "$DAG_PATH" \
        --data-paths "$DATA_PATH" \
        --test-config "$CAUSAL_TESTS" \
        --output "$CAUSAL_TEST_RESULTS" \
        $( [ "$IGNORE_CYCLES" = "true" ] && echo "--ignore-cycles" ) \
        $( [ "$VERBOSE" = "true" ] && echo "--verbose" ) \
        $( [ -n "$QUERY" ] && echo "--query" "$QUERY" ) \
        $( [ "$ADEQUACY" = "true" ] && echo "--adequacy --adequacy-bootstrap-size $BOOTSTRAP_SIZE" ) \
        $( [ "$SILENT" = "true" ] && echo "--silent" ) \
        $( [ "$BATCH_SIZE" != "0" ] && echo "--batch-size $BATCH_SIZE" )
fi