from causal_testing.testing.causal_surrogate_assisted import SimulationResult, Simulator
from examples.apsdigitaltwin.util.model import Model, OpenAPS, i_label, g_label, s_label


class APSDigitalTwinSimulator(Simulator):
    def __init__(self, constants) -> None:
        super().__init__()

        self.constants = constants

    def run_with_config(self, configuration) -> SimulationResult:
        min_bg = 200
        max_bg = 0
        end_bg = 0
        end_cob = 0
        end_iob = 0
        open_aps_output = 0
        violation = False

        open_aps = OpenAPS()
        model_openaps = Model([configuration[1], 0, 0, configuration[0], configuration[2]], self.constants)
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
            "start_bg": configuration[0],
            "start_cob": configuration[1],
            "start_iob": configuration[2],
            "end_bg": end_bg,
            "end_cob": end_cob,
            "end_iob": end_iob,
            "hypo": min_bg,
            "hyper": max_bg,
            "open_aps_output": open_aps_output,
        }

        return SimulationResult(data, violation)

