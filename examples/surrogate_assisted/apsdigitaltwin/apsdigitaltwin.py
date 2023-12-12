from causal_testing.data_collection.data_collector import ObservationalDataCollector
from causal_testing.specification.causal_dag import CausalDAG
from causal_testing.specification.causal_specification import CausalSpecification
from causal_testing.specification.scenario import Scenario
from causal_testing.specification.variable import Input, Output
from causal_testing.surrogate.causal_surrogate_assisted import CausalSurrogateAssistedTestCase, SimulationResult, Simulator
from causal_testing.surrogate.surrogate_search_algorithms import GeneticSearchAlgorithm
from examples.surrogate_assisted.apsdigitaltwin.util.model import Model, OpenAPS, i_label, g_label, s_label

import pandas as pd
import numpy as np
import os
import multiprocessing as mp

import random
from dotenv import load_dotenv


class APSDigitalTwinSimulator(Simulator):
    def __init__(self, constants, profile_path, output_file = "./openaps_temp") -> None:
        super().__init__()

        self.constants = constants
        self.profile_path = profile_path
        self.output_file = output_file

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
                rate = open_aps.run(model_openaps.history, output_file=self.output_file)
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

        violation = max_bg > 200 or min_bg < 50

        return SimulationResult(data, violation, None)

def main(file):
    random.seed(123)
    np.random.seed(123)

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
    const_file_name = file.replace("datasets", "constants").replace("_np_random_random_faulty_scenarios", ".txt")
    with open(const_file_name, "r") as const_file:
        constants = const_file.read().replace("[", "").replace("]", "").split(",")
        constants = [np.float64(const) for const in constants]
        constants[7] = int(constants[7])

    simulator = APSDigitalTwinSimulator(constants, "./util/profile.json", f"./{file}_openaps_temp")
    data_collector = ObservationalDataCollector(scenario, pd.read_csv(f"./{file}.csv"))
    test_case = CausalSurrogateAssistedTestCase(specification, ga_search, simulator)

    res, iter, df = test_case.execute(data_collector)
    with open(f"./outputs/{file.replace('./datasets/', '')}.txt", "w") as out:
        out.write(str(res) + "\n" + str(iter))
    df.to_csv(f"./outputs/{file.replace('./datasets/', '')}_full.csv")

    print(f"finished {file}")

if __name__ == "__main__":
    load_dotenv()

    all_traces = os.listdir("./datasets")
    while len(all_traces) > 0:
        num = 1
        if num > len(all_traces):
            num = len(all_traces)

        with mp.Pool(processes=num) as pool:
            pool_vals = []
            while len(pool_vals) < num and len(all_traces) > 0:
                data_trace = all_traces.pop()
                if data_trace.endswith(".csv"):
                    if len(pd.read_csv(os.path.join("./datasets", data_trace))) >= 300:
                        pool_vals.append(f"./datasets/{data_trace[:-4]}")

            pool.map(main, pool_vals)