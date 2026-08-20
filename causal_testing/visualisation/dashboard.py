"""
This module implements the Dashboard class to provide a panel dashboard to visualise causal test results.
"""

import panel as pn
import param

from causal_testing.causal_testing_framework import CausalTestingFramework
from causal_testing.visualisation.visualisation_plotter import VisualisationPlotter

pn.extension(design="material", sizing_mode="stretch_width")


class Dashboard(param.Parameterized):
    """
    Class to contain the main app and plotting functionality.
    """

    def __init__(self, ctf: CausalTestingFramework):
        super().__init__()
        self.plotter = VisualisationPlotter(ctf)

    def sidebar(self) -> pn.Param:
        """
        :returns: Parameters to go in the sidebar.
        """
        return pn.Param(self.param)

    def main_panel(self) -> pn.Row:
        """
        Main panel with content.
        """
        return pn.Row(self.plotter.interactive_results_dag(), self.plotter.test_outcome_adjacency())

    def serve(self):
        """
        Serve the dashboard.
        """
        page_content = pn.template.MaterialTemplate(
            title="Causal Testing Framework",
            site="Test Results",
            sidebar=self.sidebar,
            main=[self.main_panel],
        )
        pn.serve(page_content)
