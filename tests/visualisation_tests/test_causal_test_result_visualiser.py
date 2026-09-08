import os
import unittest
from itertools import cycle
from tempfile import TemporaryDirectory

import pandas as pd

from causal_testing.causal_testing_framework import CausalTestingFramework
from causal_testing.estimation.effect_estimate import EffectEstimate
from causal_testing.specification.causal_dag import CausalDAG
from causal_testing.testing.causal_test_result import CausalTestResult, TestOutcome
from causal_testing.visualisation.visualisation_plotter import VisualisationPlotter


class TestVisualiser(unittest.TestCase):
    def test_results_dag(self):
        dag = CausalDAG()
        dag.add_edges_from([("A", "B"), ("C", "D"), ("E", "F")])
        dag.datatypes = {node: float for node in dag.nodes}
        test_cases = dag.generate_causal_tests()

        test_result_cycle = cycle([TestOutcome.PASS, TestOutcome.FAIL, TestOutcome.INESTIMABLE])
        effect_estimate_cycle = cycle(
            [
                EffectEstimate(
                    effect_measure="ate", effect_estimate=pd.Series(5), ci_low=pd.Series(4), ci_high=pd.Series(6)
                ),  # Positive
                EffectEstimate(
                    effect_measure="ate", effect_estimate=pd.Series(5), ci_low=pd.Series(-4), ci_high=pd.Series(6)
                ),  # No effect
                EffectEstimate(
                    effect_measure="ate", effect_estimate=pd.Series(-5), ci_low=pd.Series(-6), ci_high=pd.Series(-4)
                ),  # Negative
            ]
        )
        for test in test_cases:
            test.result = CausalTestResult(
                effect_estimate=next(effect_estimate_cycle),
                outcome=next(test_result_cycle),
            )
        ctf = CausalTestingFramework(dag=dag, test_cases=test_cases)
        vp = VisualisationPlotter(ctf)

        with TemporaryDirectory() as tmp:
            vp.results_dag(output_file=os.path.join(tmp, "dag.dot"))
            dag2 = CausalDAG(os.path.join(tmp, "dag.dot"), ignore_cycles=True)
            self.assertEqual(dag.nodes, dag2.nodes)
