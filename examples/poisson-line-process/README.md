# Poisson Line Process Case Study: Statistical Metamorphic Testing
In this case study, we demonstrate how causal testing can be used to implement statistical metamorphic testing as per Guderlei & Mayer 2007. As described in section 5.1 of the paper, this involves running a series of causal test cases that incrementally change the width and height of the sampling window. We then show how statistical estimation can produce similar results using only a fraction of the data.

## How to run
To run this case study:
1. Ensure all project dependencies are installed by running `pip install .` in the top level directory
   (instructions are provided in the project README).
2. Change directory to `causal_testing/examples/poisson-line-process`.
3. Run the command `python example_pure_python.py` to demonstrate causal testing using pure python.

This should print a series of causal test results and produce two CSV files. `intensity_num_shapes_results_random_1000.csv` corresponds to table 1, and `width_num_shapes_results_random_1000.csv` relates to our findings regarding the relationship of width and `P_u`.

## Running using the main entrypoint
You should be able to run the main entrypoint by simply running the following command from within this directory:

```
python -m causal_testing --dag_path dag.dot --data_paths data/random/data_random_1000.csv --test_config causal_tests.json --output results/test_results.json
```
