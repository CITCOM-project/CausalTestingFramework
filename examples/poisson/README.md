# Poisson Line Process Case Study: Statistical Metamorphic Testing
Here we demonstrate how the same test suite as in `poisson-line-process` can be coded using the JSON front end.

## How to run
To run this case study:
1. Ensure all project dependencies are installed by running `pip install .` in the top level directory
   (instructions are provided in the project README).
2. Change directory to `causal_testing/examples/poisson`.
3. Run the command `python run_causal_tests.py --data_path data.csv --dag_path dag.dot --json_path causal_tests.json`
