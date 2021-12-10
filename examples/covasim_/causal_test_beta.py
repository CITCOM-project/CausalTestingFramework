from causal_testing.specification.causal_dag import CausalDAG
from causal_testing.specification.scenario import Scenario
from causal_testing.specification.variable import Input, Output
from causal_testing.specification.causal_specification import CausalSpecification

# 1. Read in the Causal DAG
causal_dag = CausalDAG('./dag.dot')

# 2. Create variables
pop_size = Input('pop_size', int)
pop_infected = Input('pop_infected', int)
n_days = Input('n_days', int)
c_infections = Output('cum_infections', int)
c_deaths = Output('cum_deaths', int)

# 3. Create scenario by applying constraints over the input variables
scenario = Scenario(variables={pop_size, pop_infected, n_days},
                    constraints={pop_size == 10000, pop_infected == 100, n_days == 200})

# 4. Construct a causal specification from the scenario and causal DAG
causal_specification = CausalSpecification(scenario, causal_dag)
