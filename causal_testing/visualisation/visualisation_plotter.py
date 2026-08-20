"""
This module implements the visualisation plot generator to generate plots from a CausalTestingFramework instance to help
visualise the causal test results.
"""

import holoviews as hv
import networkx as nx
import numpy as np
import pandas as pd
from bokeh.models import Div, HoverTool
from bokeh.palettes import RdYlGn

from causal_testing.causal_testing_framework import CausalTestingFramework
from causal_testing.testing.causal_test_result import TestOutcome
from causal_testing.visualisation.geometry import edge_spline, node_width, sort_df_by_median_split, style_graph_hook


class VisualisationPlotter:
    """
    Class to generate plots to visualise CausalTestingFramework test results.
    """

    def __init__(self, ctf: CausalTestingFramework):
        self.ctf = ctf

    def results_dag(
        self,
        output_file: str = None,
        view_independences: bool = True,
        colours: dict[TestOutcome, str] = None,
    ) -> nx.DiGraph:
        """
        View causal test results as a graph.

        :param output_file: Optional output file to write to (.dot).
        :param view_independences: Whether to display failed independence tests (defaults to True).
        :param colours: Optional dictionary of colours to display the test outcomes.
                        By default, pass=green, fail=red, inestimable=orange.
        """
        default_colours = {TestOutcome.PASS: "green", TestOutcome.INESTIMABLE: "orange", TestOutcome.FAIL: "red"}

        if colours is not None:
            colours = default_colours | colours
        else:
            colours = default_colours

        result_dag = nx.DiGraph()
        result_dag.add_nodes_from(self.ctf.dag.nodes)
        result_dag.add_edges_from(self.ctf.dag.edges)

        for test in self.ctf.test_cases:
            if test.result:
                effect_estimate = pd.concat(
                    [
                        test.result.effect_estimate.ci_low,
                        test.result.effect_estimate.effect_estimate,
                        test.result.effect_estimate.ci_high,
                    ],
                    axis=1,
                )
                effect_estimate.columns = ["ci_low", "estimate", "ci_high"]
                if (test.treatment_variable, test.outcome_variable) in result_dag.edges:
                    result_dag[test.treatment_variable][test.outcome_variable]["label"] = test.result.effect_direction()
                    result_dag[test.treatment_variable][test.outcome_variable]["color"] = colours[test.result.outcome]
                    result_dag[test.treatment_variable][test.outcome_variable]["fontcolor"] = colours[
                        test.result.outcome
                    ]

                elif view_independences and test.result.outcome != TestOutcome.PASS:
                    result_dag.add_edge(test.treatment_variable, test.outcome_variable, ignore_cycles=True)
                    result_dag[test.treatment_variable][test.outcome_variable]["style"] = "dashed"
                    result_dag[test.treatment_variable][test.outcome_variable]["label"] = test.result.effect_direction()
                    result_dag[test.treatment_variable][test.outcome_variable]["color"] = colours[test.result.outcome]
                    result_dag[test.treatment_variable][test.outcome_variable]["fontcolor"] = colours[
                        test.result.outcome
                    ]

        if output_file is not None:
            nx.drawing.nx_pydot.write_dot(result_dag, output_file)

        return result_dag

    def data_adequacy_heatmap(self) -> hv.HeatMap:
        """
        Visualise data adequacy as an adjacency matrix heatmap of the kurtosis.
        """
        adequacy = pd.json_normalize(map(lambda t: t.to_dict(), self.ctf.test_cases))

        for col in [
            "effect_estimate.effect_estimate",
            "effect_estimate.ci_low",
            "effect_estimate.ci_high",
            "adequacy.kurtosis",
        ]:
            columns = [c for c in adequacy.columns if c.startswith(f"result.{col}.")]
            adequacy[f"result.{col}"] = adequacy[columns].bfill(axis=1).iloc[:, 0]
            adequacy = adequacy.drop(columns=columns)
        adequacy = sort_df_by_median_split(adequacy, value_col="result.adequacy.kurtosis")

        # Get data bounds
        vmin = adequacy["result.adequacy.kurtosis"].min()
        vmax = adequacy["result.adequacy.kurtosis"].max()

        # Calculate zero position (0.0 to 1.0)
        zero_ratio = (0 - vmin) / (vmax - vmin)

        # Generate the colour samples from the negative and positive colourmaps
        num_samples = 1000
        n_neg = int(num_samples * zero_ratio)
        n_pos = num_samples - n_neg

        neg_colors = hv.plotting.util.process_cmap("blues_r", provider="bokeh", ncolors=n_neg)
        pos_colors = hv.plotting.util.process_cmap("YlOrRd", provider="bokeh", ncolors=n_pos)
        asymmetric_cmap = neg_colors + pos_colors

        # Render
        return hv.HeatMap(
            adequacy,
            kdims=[
                ("estimator.treatment_variable", "Treatment variable"),
                ("estimator.outcome_variable", "Outcome variable"),
            ],
            vdims=[("result.adequacy.kurtosis", "Kurtosis")],
        ).opts(
            cmap=asymmetric_cmap,
            clim=(vmin, vmax),
            clipping_colors={"NaN": "grey"},  # Grey out invalid tests
            colorbar=True,
            xrotation=90,
            width=600,
            height=500,
            tools=["hover"],
            xlabel="Treatment variable",
            ylabel="Outcome variable",
            clabel="Causal test adequacy",
            title="Data Adequacy",
        )

    def dag_adequacy_heatmap(self) -> hv.HeatMap:
        """
        Visualise dag adequacy as an adjacency matrix heatmap of the percentage of passing test cases.
        """
        adequacy = pd.json_normalize(map(lambda t: t.to_dict(), self.ctf.test_cases))

        # Turn passing test cases into a percentage
        adequacy["result.adequacy.passing"] = (
            adequacy["result.adequacy.passing"] / adequacy["result.adequacy.bootstrap_size"]
        ) * 100

        return hv.HeatMap(
            sort_df_by_median_split(adequacy, value_col="result.adequacy.passing"),
            kdims=[
                ("estimator.treatment_variable", "Treatment variable"),
                ("estimator.outcome_variable", "Outcome variable"),
            ],
            vdims=[("result.adequacy.passing", "Passing (%)")],
        ).opts(
            cmap="RdYlGn",
            clim=(0, 100),
            clipping_colors={"NaN": "grey"},  # Grey out invalid tests
            colorbar=True,
            xrotation=90,
            width=600,
            height=500,
            tools=["hover"],
            xlabel="Treatment variable",
            ylabel="Outcome variable",
            clabel="Percentage passing test cases",
            title="DAG Adequacy",
        )

    def test_outcome_adjacency(self) -> hv.HeatMap:
        """
        Visualise causal test results as an adjacency matrix.
        """
        results = pd.json_normalize(map(lambda t: t.to_dict(), self.ctf.test_cases))
        results["result.outcome.value"] = results["result.outcome"].apply(lambda x: TestOutcome[x].value)

        red = RdYlGn[11][0]
        yellow = RdYlGn[11][7]
        green = RdYlGn[11][10]

        colour_map = {"FAIL": red, "INESTIMABLE": yellow, "PASS": green}

        def add_discrete_legend(plot, _):
            legend_html = f"""
            <div style="text-align: center; font-family: sans-serif; font-size: 14px; padding: 4px;">
                <span style="color: {red}; font-weight: bold;">■ Pass</span>
                <span style="color: {yellow}; font-weight: bold;">■ Inestimable</span>
                <span style="color: {green}; font-weight: bold;">■ Fail</span>
            </div>
            """
            div = Div(text=legend_html)
            plot.state.add_layout(div, "above")

        # Apply to your HeatMap
        return hv.HeatMap(
            sort_df_by_median_split(results, value_col="result.outcome.value", vdims=["result.outcome"]),
            kdims=[
                ("estimator.treatment_variable", "Treatment variable"),
                ("estimator.outcome_variable", "Outcome variable"),
            ],
            vdims=[("result.outcome", "Outcome")],
        ).opts(
            cmap=colour_map,
            clipping_colors={"NaN": "grey"},
            xrotation=90,
            width=500,
            height=500,
            tools=["hover"],
            xlabel="Treatment variable",
            ylabel="Outcome variable",
            hooks=[add_discrete_legend],
            title="Test Outcomes",
        )

    def interactive_results_dag(self) -> hv.Overlay:
        """
        Generate an interactive holoview graph of the causal DAG showing failing tests.

        :returns: Inveractive holoviews graph.
        """
        results = self.results_dag()
        for test in self.ctf.test_cases:
            effect_estimate = pd.concat(
                [
                    test.result.effect_estimate.ci_low,
                    test.result.effect_estimate.effect_estimate,
                    test.result.effect_estimate.ci_high,
                ],
                axis=1,
            )
            effect_estimate.columns = ["ci_low", "estimate", "ci_high"]
            try:
                results[test.treatment_variable][test.outcome_variable]["title"] = effect_estimate.to_html()
            except KeyError:
                continue

        # Use DOT to do the layout
        agraph = nx.nx_agraph.to_agraph(results)
        agraph.layout(prog="dot")

        node_positions = {}
        for node in agraph.nodes():
            x, y = map(float, node.attr["pos"].split(","))
            node_positions[node.name] = (x, y)

        # Build the edges
        edges_df = pd.DataFrame([{"source": u, "target": v} | data for u, v, data in results.edges(data=True)])

        edges_df["trimmed_path"] = edges_df[["source", "target"]].apply(
            lambda row: edge_spline(
                dot_pos=agraph.get_edge(row["source"], row["target"]).attr["pos"],
                target_node_centre=node_positions[row["target"]],
                target_node_width=node_width(row["target"]) / 2,
            ),
            axis=1,
        )
        edges_df[["arrow_starts_x", "arrow_starts_y"]] = pd.DataFrame(
            [trimmed_path[-2] for trimmed_path in edges_df["trimmed_path"]], index=edges_df.index
        )
        edges_df[["arrow_ends_x", "arrow_ends_y"]] = pd.DataFrame(
            [trimmed_path[-1] for trimmed_path in edges_df["trimmed_path"]], index=edges_df.index
        )

        nodes_df = pd.DataFrame(
            [(x, y, node_id) for node_id, (x, y) in node_positions.items()], columns=["x", "y", "node_id"]
        )

        # Build the graph from the nodes and edges
        graph = hv.Graph(
            (
                edges_df,
                hv.Nodes(
                    nodes_df,
                    kdims=["x", "y", "node_id"],
                ),
                hv.EdgePaths(edges_df["trimmed_path"].tolist()),
            ),
            kdims=["source", "target"],
            vdims=[c for c in edges_df.columns if c not in ["source", "target", "trimmed_path"]],
        ).opts(
            edge_line_dash="style",
            edge_line_width=1.5,
            edge_color="color",
            edge_hover_line_color="color",
            width=900,
            height=450,
            hooks=[style_graph_hook],
            xaxis=None,
            yaxis=None,
            tools=[
                HoverTool(
                    tooltips="""
                <div style="padding: 6px; border: 1px solid #ccc; font-family: sans-serif;">
                    <strong>Treatment:</strong> @source<br>
                    <strong>Outcome:</strong> @target<br>
                    <strong>Causal Effect:</strong> <br/> @title{safe}<br>
                </div>
            """
                )
            ],
            inspection_policy="edges",
        )

        # Label layers
        node_labels = hv.Labels(nodes_df, kdims=["x", "y"], vdims=["node_id"]).opts(
            text_font_size="9pt",
            text_color="black",
            text_align="center",
            text_baseline="middle",
            yoffset=0,
        )

        edge_labels = hv.Labels(
            pd.concat(
                [
                    pd.DataFrame(
                        # Stack the x and y elements of the middle index of each trimmed path
                        np.vstack(edges_df["trimmed_path"].apply(lambda path: path[len(path) // 2]).values),
                        columns=["x", "y"],
                    ),
                    edges_df["label"],
                ],
                axis=1,
            ),
            kdims=["x", "y"],
            vdims=["label"],
        ).opts(
            text_font_size="9pt",
            text_color="darkblue",
            text_align="center",
            text_baseline="middle",
        )

        return graph * node_labels * edge_labels
