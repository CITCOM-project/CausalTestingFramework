from causal_testing.specification.causal_dag import CausalDAG
from causal_testing.specification.scenario import Scenario
from causal_testing.specification.variable import Input, Output
from causal_testing.specification.causal_specification import CausalSpecification
from causal_testing.testing.causal_test_case import CausalTestCase
from causal_testing.testing.causal_test_outcome import Positive
from causal_testing.testing.intervention import Intervention
from causal_testing.testing.causal_test_engine import CausalTestEngine
from causal_testing.testing.estimators import LinearRegressionEstimator

# 1. Read in the Causal DAG
causal_dag = CausalDAG('./dag.dot')

# 2. Create variables
pop_size = Input('pop_size', int)
pop_infected = Input('pop_infected', int)
n_days = Input('n_days', int)
cum_infections = Output('cum_infections', int)
cum_deaths = Output('cum_deaths', int)
location = Input('location', str)
variants = Input('variants', str)
avg_age = Input('avg_age', float)
beta = Input('beta', float)

# 3. Create scenario by applying constraints over a subset of the input variables
scenario = Scenario(variables={pop_size, pop_infected, n_days, cum_infections, cum_deaths,
                               location, variants, avg_age, beta},
                    constraints={pop_size.z3 == 10000, pop_infected.z3 == 100, n_days.z3 == 200})

# 4. Construct a causal specification from the scenario and causal DAG
causal_specification = CausalSpecification(scenario, causal_dag)

# 5. Create a causal test case
causal_test_case = CausalTestCase(control_input_configuration={beta: 0.016},
                                  expected_causal_effect=Positive,
                                  outcome_variables={cum_infections},
                                  intervention=Intervention((beta,), (2 * 0.016,), ), )

# 6. Create an instance of the causal test engine
causal_test_engine = CausalTestEngine(causal_test_case, causal_specification)

# 7. Obtain the minimal adjustment set for the causal test case from the causal DAG
minimal_adjustment_set = causal_test_engine.load_data('./observational_data.csv', index_col=0)

# 8. Define an estimator that adjusts for the variables in the minimal adjustment set to obtain a causal estimate
linear_regression_estimator = LinearRegressionEstimator(('beta',), 2 * 0.016, 0.016, minimal_adjustment_set,
                                                        ('cum_infections',))
# 9. Define an estimator that does not adjust for the variables in the minimal adjustment set (not causal)
linear_regression_estimator_no_adjustment = LinearRegressionEstimator(('beta',), 2 * 0.016, 0.016, set(),
                                                                      ('cum_infections',))

# 10. Execute the test case and compare the results
causal_test_result = causal_test_engine.execute_test(linear_regression_estimator, 'ate')
association_test_result = causal_test_engine.execute_test(linear_regression_estimator_no_adjustment, 'ate')
print(causal_test_result)
print(association_test_result)

