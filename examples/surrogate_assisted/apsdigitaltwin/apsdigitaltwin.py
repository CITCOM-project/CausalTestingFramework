from causal_testing.data_collection.data_collector import ObservationalDataCollector
from causal_testing.specification.causal_dag import CausalDAG
from causal_testing.specification.causal_specification import CausalSpecification
from causal_testing.specification.scenario import Scenario
from causal_testing.specification.variable import Input, Output
from causal_testing.testing.causal_surrogate_assisted import CausalSurrogateAssistedTestCase, SimulationResult, Simulator
from causal_testing.testing.surrogate_search_algorithms import GeneticSearchAlgorithm
from examples.apsdigitaltwin.util.model import Model, OpenAPS, i_label, g_label, s_label

import pandas as pd
import numpy as np

from dotenv import load_dotenv


class APSDigitalTwinSimulator(Simulator):
    def __init__(self, constants, profile_path) -> None:
        super().__init__()

        self.constants = constants
        self.profile_path = profile_path

    def run_with_config(self, configuration) -> SimulationResult:
        min_bg = 200
        max_bg = 0
        end_bg = 0
        end_cob = 0
        end_iob = 0
        open_aps_output = 0
        violation = False

        open_aps = OpenAPS(profile_path=self.profile_path)
        model_openaps = Model([configuration["start_cob"], 0, 0, configuration["start_bg"], configuration["start_iob"]], self.constants)
        for t in range(1, 121):
            if t % 5 == 1:
                rate = open_aps.run(model_openaps.history, output_file=f"./openaps_temp", faulty=True)
                if rate == -1:
                    violation = True
                open_aps_output += rate
                for j in range(5):
                    model_openaps.add_intervention(t + j, i_label, rate / 5.0)
            model_openaps.update(t)

            min_bg = min(min_bg, model_openaps.history[-1][g_label])
            max_bg = max(max_bg, model_openaps.history[-1][g_label])

            end_bg = model_openaps.history[-1][g_label]
            end_cob = model_openaps.history[-1][s_label]
            end_iob = model_openaps.history[-1][i_label]

        data = {
            "start_bg": configuration["start_bg"],
            "start_cob": configuration["start_cob"],
            "start_iob": configuration["start_iob"],
            "end_bg": end_bg,
            "end_cob": end_cob,
            "end_iob": end_iob,
            "hypo": min_bg,
            "hyper": max_bg,
            "open_aps_output": open_aps_output,
        }

        return SimulationResult(data, violation)

if __name__ == "__main__":
    load_dotenv()

    search_bias = Input("search_bias", float, hidden=True)
    
    start_bg = Input("start_bg", float)
    start_cob = Input("start_cob", float)
    start_iob = Input("start_iob", float)
    open_aps_output = Output("open_aps_output", float)
    hyper = Output("hyper", float)

    constraints = {
        start_bg >= 70, start_bg <= 180, 
        start_cob >= 100, start_cob <= 300, 
        start_iob >= 0, start_iob <= 150
    }

    scenario = Scenario(
        variables={
            search_bias,
            start_bg,
            start_cob,
            start_iob,
            open_aps_output,
            hyper,
        },
        constraints = constraints
    )

    dag = CausalDAG("./dag.dot")
    specification = CausalSpecification(scenario, dag)

    ga_config = {
        "parent_selection_type": "tournament",
        "K_tournament": 4,
        "mutation_type": "random",
        "mutation_percent_genes": 50,
        "mutation_by_replacement": True,
        "suppress_warnings": True,
    }

    ga_search = GeneticSearchAlgorithm(config=ga_config)

    constants = []
    with open("constants.txt", "r") as const_file:
        constants = const_file.read().replace("[", "").replace("]", "").split(",")
        constants = [np.float64(const) for const in constants]
        constants[7] = int(constants[7])

    simulator = APSDigitalTwinSimulator(constants, "./util/profile.json")
    data_collector = ObservationalDataCollector(scenario, pd.read_csv("./data.csv"))
    test_case = CausalSurrogateAssistedTestCase(specification, ga_search, simulator)

    print(test_case.execute(data_collector))