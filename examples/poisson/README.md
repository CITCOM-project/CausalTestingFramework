# Poisson Line Process Case Study: Statistical Metamorphic Testing
Here we demonstrate how the same test suite as in `poisson-line-process` can be coded using the JSON front end.

## How to run
To run this case study:
1. Ensure all project dependencies are installed by running `pip install .` in the top level directory
   (instructions are provided in the project README).
2. Change directory to `causal_testing/examples/poisson`.
3. Run the command `python test_run_causal_tests.py --data_path data.csv --dag_path dag.dot --json_path causal_tests.json`

This should print a series of causal test results and produce two CSV files. `intensity_num_shapes_results_random_1000.csv` corresponds to table 1, and `width_num_shapes_results_random_1000.csv` relates to our findings regarding the relationship of width and `P_u`.
