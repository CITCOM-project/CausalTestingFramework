from causal_testing.visualisation.visualisation_plotter import VisualisationPlotter
from causal_testing.causal_testing_framework import CausalTestingFramework

import holoviews as hv
import panel as pn

hv.extension("bokeh")


DAG_PATH = "dag.dot"
RESULT_CONFIG = "causal_test_results.json"

framework = CausalTestingFramework()
framework.setup(dag_path=DAG_PATH, test_cases_path=RESULT_CONFIG)

visualisation_plotter = VisualisationPlotter(framework)
dag = visualisation_plotter.interactive_results_dag()
pn.panel(dag).show()
