"""
This module implements the Dashboard class to provide a panel dashboard to visualise causal test results.
"""

import io
import json

import holoviews as hv
import networkx as nx
import panel as pn
import panel_material_ui as pmui
import param
from bokeh.io import export_png

from causal_testing.causal_testing_framework import CausalTestingFramework, data_readers, read_dataframe
from causal_testing.specification.causal_dag import CausalDAG
from causal_testing.testing.causal_test_result import TestOutcome
from causal_testing.visualisation.visualisation_plotter import VisualisationPlotter

pn.extension("codeeditor")
pn.extension(design="material", sizing_mode="stretch_width", notifications=True)

pn.config.raw_css.append(
    """
    .test_suite_stats bk-panel-models-markup-HTML {
        background: rgb(248, 249, 250);
        padding: 15px;
        border-radius: 8px;
        width: calc(100% - 20px);
    }


    @media screen {
        .test_suite_stat_title {
            font-size: 18pt;
        }
        .test_suite_stat_value {
            font-size: 54pt;
        }
        .print_only {
            display: none;
        }
    }

    @media print {
        @page {
            size: A4 portrait;
            margin: 10mm;
        }

    /* 1. Reset all parent wrappers and template containers */
        html, body,
        #container, #content, .main, .main-content,
        .bk-root, .bk-root *,
        .template-container, .pn-template {
            height: auto !important;
            flex: none !important;
        }

        .page_break_before {
            break-before: page !important;
            page-break-before: always !important;
        }


        .screen_only {
            display: none;
        }

        #sidebar,
        button,
        .bk-header {
            display: none !important;
        }

        :root {
            --sidebar-width: 0px !important;
        }

        .bk-toolbar {
            display: none !important;
        }

        .test_suite_stats bk-panel-models-markup-HTML {
            width: 25% !important;
            margin: 0 !important;
        }
    }
    """
)


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
        self.dag_file_input = pmui.FileInput(accept=".dot,.gv", mime_type="text/vnd.graphviz", label="DAG file")
        self.dag_file_input.param.watch(self._load_dag_file, "value", onlychanged=True)
        # Data
        self.data_file_input = pmui.FileInput(accept=",".join(data_readers), label="Data file")
        self.data_file_input.param.watch(self._load_data_file, "value", onlychanged=True)

        # Tests
        self.test_file_input = pmui.FileInput(accept=".json", label="Test file")
        self.test_file_input.param.watch(self._load_test_file, "value", onlychanged=True)

        # Generate causal tests
        self.generate_tests = pmui.Button(
            label="Generate", sizing_mode="fixed", align="end", height=37, width=97, disabled=True
        )
        self.generate_tests.param.watch(self._generate_tests, "value", onlychanged=True)

        # Edit causal tests
        self.test_editor = pn.widgets.CodeEditor(language="json", value="")
        self.update_tests = pmui.Button(label="Update", sizing_mode="fixed", align="end", height=37, width=120)
        self.update_tests.param.watch(self._update_tests, "value", onlychanged=True)
        self.download_tests = pmui.FileDownload(
            callback=self._download_tests,
            filename="causal_tests.json",
            button_type="primary",
            label="Download",
            height=37,
            width=120,
            sizing_mode="fixed",
        )

        # Run causal tests
        self.run_tests = pmui.Button(label="Run Tests", color="primary", disabled=True)
        self.run_tests.param.watch(self._run_tests, "value", onlychanged=True)

    def _load_dag_file(self, event):
        """Parses uploaded DOT bytes into a CausalDAG and initialises the CTF with it."""
        parsed_multigraph = nx.nx_pydot.read_dot(io.StringIO(event.new.decode("utf-8")))
        dag = CausalDAG()
        dag.update(nx.DiGraph(parsed_multigraph))
        self.ctf.dag = dag
        self.param.trigger("ctf")
        self.run_tests.disabled = not self.ctf.ready_to_run()
        self.generate_tests.disabled = False

    def _load_test_file(self, event):
        self.ctf.test_cases = [self.ctf.create_causal_test(test) for test in json.load(io.BytesIO(event.new))]
        if any(test.result is not None and test.result.adequacy is not None for test in self.ctf.test_cases):
            self.adequacy = True

        self.param.trigger("ctf")
        self.run_tests.disabled = not self.ctf.ready_to_run()

    def _load_data_file(self, event):
        """Parses uploaded data bytes into a pandas DataFrame and initialises the CTF with it"""
        self.ctf.df = read_dataframe(file_path=self.data_file_input.filename, content=io.BytesIO(event.new))
        self.ctf.dag.datatypes = self.ctf.df.dtypes
        self.run_tests.disabled = not self.ctf.ready_to_run()

    def _generate_tests(self, _):
        """Generates causal test cases from a DAG."""
        try:
            self.ctf.test_cases = self.ctf.dag.generate_causal_tests()
            self.param.trigger("ctf")
            self.run_tests.disabled = not self.ctf.ready_to_run()
        except ValueError as e:
            pn.state.notifications.error(str(e), duration=0)

    def _run_tests(self, _):
        total_steps = len(self.ctf.test_cases)
        original_label = self.run_tests.label
        theme_var = f"var(--bs-{self.run_tests.button_type}, var(--panel-primary-color, #2085ec))"
        self.run_tests.disabled = True
        for i, test_case in enumerate(self.ctf.test_cases):
            pct = int((i / total_steps) * 100)
            self.run_tests.label = f"Processing... {pct}%"
            self.run_tests.stylesheets = [
                f"""
            button {{
                background-image: linear-gradient(to right, {theme_var} {pct}%, #e0e0e0 {pct}%) !important;
                background-color: transparent !important;
                border-color: #ccc !important;
            }}
            """
            ]
            test_case.execute_test(
                self.ctf.df, suppress_estimation_errors=True, adequacy=self.adequacy, bootstrap_size=100
            )
        self.run_tests.disabled = False
        self.run_tests.stylesheets = [
            """
        button {{
            background: #2085ec;
        }}
    """
        ]
        self.run_tests.label = original_label
        self.param.trigger("ctf")

    def _test_suite_stat(
        self,
        name: str,
        value: int,
        percentage: float = None,
        colors: list[tuple[int, str]] = None,
        default_color: str = None,
    ):
        color = "" if default_color is None else f"color: {default_color};"
        if value is not None and colors is not None:
            for threshold, c in colors:
                if value < threshold:
                    color = f"color: {c};"
                    break
        if percentage is not None:
            value_html = f"""
            <span>{value if value is not None else "-"}
                <span style="font-size: 0.5em;">({percentage:.1f}%)</span>
            </span>
            """
        else:
            value_html = f"""<span>{value if value is not None else "-"}</span>"""

        return pn.pane.HTML(
            f"""
        <div class="bk-panel-models-markup-HTML">
            <div style="width: 100%; min-width: 0px; visibility: visible; {color}">
                <div class="test_suite_stat_title">{name}</div>
                <div class="test_suite_stat_value">{value_html}</div>
            </div>
        </div>
        """
        )

    def test_suite_stats(self) -> pn.GridBox:
        """
        Key figures about the test suite: Total, Passing, Failing, Inestimable
        """
        test_df = self.ctf.test_dataframe()
        if "result.outcome" not in test_df:
            test_df["result.outcome"] = None

        num_tests = self._test_suite_stat(
            name="Test Cases",
            value=None if test_df.empty else len(test_df),
            colors=[(0, "black")],
        )

        totals = {outcome: (test_df["result.outcome"] == outcome.name).sum() for outcome in TestOutcome}

        return pn.GridBox(
            num_tests,
            self._test_suite_stat(
                name="Passing tests",
                value=None if test_df.empty or test_df["result.outcome"].isnull().any() else totals[TestOutcome.PASS],
                colors=[(1, "red")],  # Color red if everything fails
                default_color="green",
                percentage=(
                    None
                    if test_df.empty or test_df["result.outcome"].isnull().any()
                    else (totals[TestOutcome.PASS] / len(test_df)) * 100
                ),
            ),
            self._test_suite_stat(
                name="Failing tests",
                value=None if test_df.empty or test_df["result.outcome"].isnull().any() else totals[TestOutcome.FAIL],
                colors=[(1, "green")],  # Color red if anything fails
                default_color="red",
                percentage=(
                    None
                    if test_df.empty or test_df["result.outcome"].isnull().any()
                    else (totals[TestOutcome.FAIL] / len(test_df)) * 100
                ),
            ),
            self._test_suite_stat(
                name="Inestimable tests",
                value=(
                    None
                    if test_df.empty or test_df["result.outcome"].isnull().any()
                    else totals[TestOutcome.INESTIMABLE]
                ),
                colors=[(1, "green")],  # Color green if everything is estimable
                default_color="orange",
                percentage=(
                    None
                    if test_df.empty or test_df["result.outcome"].isnull().any()
                    else (totals[TestOutcome.INESTIMABLE] / len(test_df)) * 100
                ),
            ),
            ncols=4,
            sizing_mode="stretch_width",
            css_classes=["test_suite_stats"],
        )

    def sidebar(self) -> pn.Param:
        """
        :returns: Parameters to go in the sidebar.
        """
        return pn.Column(
            self.dag_file_input,
            self.data_file_input,
            pn.Row(self.test_file_input, self.generate_tests),
            pn.Param(
                self.param.adequacy,
                widgets={
                    "adequacy": pmui.Switch,
                    # "styles": {"transform": "scale(1.5)", "transform-origin": "left center"},
                },
            ),
            self.run_tests,
        )

    def _update_tests(self, _):
        self.ctf.test_cases = [self.ctf.create_causal_test(test) for test in json.loads(self.test_editor.value)]

    def _download_tests(self):
        return io.BytesIO(self.test_editor.value.encode("utf-8"))

    @pn.depends("ctf")
    def test_editor_panel(self) -> pn.Column:
        """
        Panel to allow users to edit causal test cases.
        """
        if self.ctf and self.ctf.test_cases:
            self.test_editor.value = json.dumps([test.to_dict() for test in self.ctf.test_cases], indent=2)
            return pn.Column(
                pn.Row(self.test_editor),
                pn.Row(self.update_tests, self.download_tests),
            )
        return None

    @pn.depends("ctf", "adequacy")
    def main_panel(self) -> pn.Column:
        """
        Main panel for content.
        """
        content = pn.Column(self.test_suite_stats())

        if self.ctf.dag is None:
            content.append(pn.pane.Markdown("Please select the causal DAG."))
        else:
            results = pn.FlexBox(
                pn.pane.PNG(
                    export_png(
                        hv.render(self.plotter.interactive_results_dag(frame_height=250, width=600), backend="bokeh"),
                        webdriver=None,
                    ),
                    css_classes=["print_only"],
                ),
                pn.pane.HoloViews(
                    self.plotter.interactive_results_dag(frame_height=300),
                    styles={"flex": "1 1 800px"},
                    css_classes=["screen_only"],
                ),
                flex_direction="row",
                flex_wrap="wrap",
                sizing_mode="stretch_width",
            )

            if any(test.result for test in self.ctf.test_cases):
                content.append(
                    pn.pane.Markdown(
                        """
                # Test Outcomes

                Passing causal tests are shown in green.
                Failing tests are shown in red.
                Inestimable tests are shown in yellow.
                These occur the test data violates the
                [positivity](https://causal-testing-framework.readthedocs.io/en/latest/modules/test_data.html)
                assumption, especially for categorical variables, where the data is split based on variable values.

                In the adjacency matrix, the treatments and outcomes have been ordered such that failing tests appear
                closer to the bottom left corner and passing tests appear closer to the top right corner.
                This makes it easier to spot patterns in the data, e.g. variables involved in many failing tests, or
                clusters of failing tests.
                """
                    )
                )
                results.append(
                    pn.pane.HoloViews(
                        self.plotter.test_outcome_adjacency(
                            xrotation=45,
                            frame_height=300,
                        ),
                        styles={"flex": "1 1 400px"},
                        css_classes=["test_outcome_adjacency"],
                    ),
                )
            content.append(results)

            if self.adequacy:
                content.append(
                    pn.pane.Markdown(
                        """
            # Test Adequacy

            [Causal test adequacy](https://causal-testing-framework.readthedocs.io/en/latest/modules/test_adequacy.html)
            essentially gives an indication as to whether the causal test cases have been evaluated with sufficient
            data to make the test outcomes trustworthy.
            For Data Adequacy, values close to zero are desirable.
            Larger values indicate an "unstable" causal estimate, i.e. individual data points have a large effect on
            the estimate (and therefore the test outcome), indicating that the test outcome may not be trustworthy
            and more data is desirable.
            Values smaller than zero indicate a "suspiciously stable" causal estimate, indicating that the dataset
            may not capture the full stochasticity of the model (or that the model is deterministic).

            DAG adequacy indicates how well the DAG fits the dataset, and how "stable" the test outcomes are,
            i.e. how affected the outcomes are by individual data points.
            Higher percentage pass rates indicate more reliable test outcomes.
                """
                    )
                )
                content.append(
                    pn.FlexBox(
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

    def build_template(self):
        """
        Set up the dashboard view.
        """
        return pn.template.MaterialTemplate(
            title="Causal Testing Framework",
            site="Test Results",
            sidebar=self.sidebar(),
            main=[pn.Tabs(("Main", self.main_panel), ("Test Editor", self.test_editor_panel))],
        )


def dashboard_session():
    """
    Set up the dashboard session instance.
    """
    dashboard = Dashboard()
    return dashboard.build_template()


def serve_dashboard():
    """
    Serve the dashboard.
    """
    pn.serve(dashboard_session, port=5006, show=False)


if __name__ == "__main__":
    Dashboard().serve()
