"""
This module implements the Dashboard class to provide a panel dashboard to visualise causal test results.
"""

import io

import networkx as nx
import panel as pn
import param

from causal_testing.causal_testing_framework import CausalTestingFramework, data_readers, read_dataframe
from causal_testing.specification.causal_dag import CausalDAG
from causal_testing.testing.causal_test_result import TestOutcome
from causal_testing.visualisation.visualisation_plotter import VisualisationPlotter

pn.extension(design="material", sizing_mode="stretch_width")


class Dashboard(param.Parameterized):
    """
    Class to contain the main app.
    """

    df = param.DataFrame(default=None)
    ctf = param.ClassSelector(class_=CausalTestingFramework, default=CausalTestingFramework(test_cases=[]))
    adequacy = param.Boolean(default=False)

    def __init__(self):
        super().__init__()
        self.plotter = VisualisationPlotter(self.ctf)

        # DAG
        self.dag_file_input = pn.widgets.FileInput(accept=".dot,.gv", mime_type="text/vnd.graphviz", label="DAG file")
        self.dag_file_input.param.watch(self._parse_dot_file, "value", onlychanged=True)

        # Data
        self.data_file_input = pn.widgets.FileInput(accept=",".join(data_readers), label="Data file")
        self.data_file_input.param.watch(self._parse_data_file, "value", onlychanged=True)

        # Generate causal tests
        self.generate_tests = pn.widgets.Button(label="Generate Tests")
        self.generate_tests.param.watch(self._generate_tests, "value", onlychanged=True)

        # Run causal tests
        self.run_tests = pn.widgets.Button(label="Run Tests")
        self.run_tests.param.watch(self._run_tests, "value", onlychanged=True)

    def _parse_dot_file(self, event):
        """Parses uploaded DOT bytes into a CausalDAG and initialises the CTF with it."""
        if not event.new:
            return

        parsed_multigraph = nx.nx_pydot.read_dot(io.StringIO(event.new.decode("utf-8")))
        dag = CausalDAG()
        dag.update(nx.DiGraph(parsed_multigraph))
        self.ctf.dag = dag
        self.param.trigger("ctf")

    def _parse_data_file(self, event):
        """Parses uploaded data bytes into a pandas DataFrame and initialises the CTF with it"""
        self.ctf.df = read_dataframe(file_path=self.data_file_input.filename, content=io.BytesIO(event.new))
        self.ctf.dag.datatypes = self.ctf.df.dtypes

    def _generate_tests(self, _):
        """Generates causal test cases from a DAG."""
        self.ctf.test_cases = self.ctf.dag.generate_causal_tests()
        self.param.trigger("ctf")

    def _run_tests(self, _):
        self.ctf.run_tests(silent=True, adequacy=self.adequacy)
        self.param.trigger("ctf")

    def key_stats(self) -> pn.Row:
        """
        Key figures about the test suite: Total, Passing, Failing, Inestimable
        """
        if not self.ctf.test_cases:
            return pn.pane.Markdown("Generate or load test cases.")

        test_df = self.ctf.test_dataframe()

        num_tests = pn.indicators.Number(
            name="Test Cases",
            value=len(test_df),
            colors=[(0, "black")],
            styles={"background": "#f8f9fa", "padding": "15px", "border-radius": "8px"},
            sizing_mode="stretch_width",
        )

        if "result.outcome" not in test_df:
            return pn.Row(num_tests, pn.pane.Markdown("Run test cases."))

        totals = {outcome: (test_df["result.outcome"] == outcome.name).sum() for outcome in TestOutcome}

        def format_result(outcome: TestOutcome) -> str:
            return f"{{value}} <span style='font-size: 0.5em;'>({(totals[outcome]/len(test_df))*100:.1f}%)</span>"

        return pn.Row(
            num_tests,
            pn.indicators.Number(
                name="Passing tests",
                value=totals[TestOutcome.PASS],
                colors=[(1, "red")],  # Color red if everything fails
                default_color="green",
                format=format_result(TestOutcome.PASS),
                styles={"background": "#f8f9fa", "padding": "15px", "border-radius": "8px"},
                sizing_mode="stretch_width",
            ),
            pn.indicators.Number(
                name="Failing tests",
                value=totals[TestOutcome.FAIL],
                styles={"background": "#f8f9fa", "padding": "15px", "border-radius": "8px"},
                colors=[(1, "green")],  # Color red if anything fails
                default_color="red",
                format=format_result(TestOutcome.FAIL),
                sizing_mode="stretch_width",
            ),
            pn.indicators.Number(
                name="Inestimable tests",
                value=totals[TestOutcome.INESTIMABLE],
                colors=[(1, "green")],  # Color green if everything is estimable
                default_color="orange",
                format=format_result(TestOutcome.INESTIMABLE),
                styles={"background": "#f8f9fa", "padding": "15px", "border-radius": "8px"},
                sizing_mode="stretch_width",
            ),
        )

    def sidebar(self) -> pn.Param:
        """
        :returns: Parameters to go in the sidebar.
        """
        return pn.Column(
            self.dag_file_input,
            self.data_file_input,
            self.generate_tests,
            pn.Param(self.param.adequacy, widgets={"adequacy": pn.widgets.Switch}),
            self.run_tests,
        )

    @pn.depends("ctf")
    def main_panel(self) -> pn.Row:
        """
        Main panel for content.
        """
        if self.ctf.dag is None:
            return pn.pane.Markdown("Please select the causal DAG.")

        results_dag = self.plotter.interactive_results_dag(
            width=800,
            height=450,
        )

        results = pn.Row(results_dag)
        if any(test.result for test in self.ctf.test_cases):
            results.append(
                self.plotter.test_outcome_adjacency(
                    xrotation=45,
                    width=450,
                    height=450,
                ),
            )

        content = pn.Column(
            self.key_stats(),
            results,
        )
        if self.adequacy:
            content.append(
                pn.Row(
                    self.plotter.data_adequacy_heatmap(
                        xrotation=45,
                        width=500,
                        height=380,
                    ),
                    self.plotter.dag_adequacy_heatmap(
                        xrotation=45,
                        width=500,
                        height=380,
                    ),
                ),
            )
        return content

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


if __name__ == "__main__":
    Dashboard().serve()
