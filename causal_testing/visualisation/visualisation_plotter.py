"""
This module implements the visualisation plot generator to generate plots from a CausalTestingFramework instance to help
visualise the causal test results.
"""

import holoviews as hv
import networkx as nx
import numpy as np
import pandas as pd
import pydot
from bokeh.models import Div, HoverTool
from bokeh.palettes import RdYlGn
from holoviews.plotting.bokeh.graphs import GraphPlot

from causal_testing.specification.causal_dag import CausalDAG
from causal_testing.testing.causal_test_result import TestOutcome
from causal_testing.visualisation.geometry import add_colorbar_annotations, edge_spline, node_width, style_graph_hook

hv.extension("bokeh")


class VisualisationPlotter:
    """
    Class to generate plots to visualise CausalTestingFramework test results.

    :ivar dag: The causal DAG.
    :ivar df: The causal testing data.
    :ivar colour_map: Dictionary mapping TestOutcomes PASS, FAIL, and INESTIMABLE test outcomes to colours.
    """

    def __init__(self, dag: CausalDAG = None, df: pd.DataFrame = None, colour_map: dict[TestOutcome, str] = None):
        self.dag = dag
        self.df = None
        self.results = False
        if df is not None:
            self.update_df(df)

        self.colour_map = (
            colour_map
            if colour_map is not None
            else {
                TestOutcome.PASS: RdYlGn[11][0],
                TestOutcome.INESTIMABLE: RdYlGn[11][7],
                TestOutcome.FAIL: RdYlGn[11][9],
            }
        )

    def update_df(self, df: pd.DataFrame):
        """
        Update and preformat the data.

        :param df: The new dataframe.
        """

        # Pre-format the data
        if "result.outcome" in df:
            df["result.outcome.value"] = df["result.outcome"].apply(lambda x: TestOutcome[x].value)

        self.xticks = list(enumerate(df["estimator.treatment_variable"].unique()))
        self.yticks = list(enumerate(df["estimator.outcome_variable"].unique()))

        df["treatment_variable_inx"] = df["estimator.treatment_variable"].map({v: k for k, v in self.xticks})
        df["outcome_variable_inx"] = df["estimator.outcome_variable"].map({v: k for k, v in self.yticks})

        for col in [
            "effect_estimate.effect_estimate",
            "effect_estimate.ci_low",
            "effect_estimate.ci_high",
            "adequacy.kurtosis",
        ]:
            prefix = f"result.{col}."
            columns = [c for c in df.columns if c.startswith(prefix)]
            if columns:
                self.results = True
                df[f"result.{col}"] = df[columns].apply(
                    lambda row, p=prefix: {k.replace(p, ""): v for k, v in row.dropna().to_dict().items()},
                    axis=1,
                )
                df = df.drop(columns=columns)
                if "kurtosis" in col:
                    # Causal test adequacy is displayed as a single number, but categorical treatments get one value
                    # for each value, e.g. if we've got a variable `Colour` that can be red, blue, or green, we get
                    # a kurtosis value for each, which we need to aggregate.
                    # I'm taking this as a max for now, but it'd be nice to do something more meaningful.
                    df[f"result.{col}"] = df[f"result.{col}"].apply(lambda row: max(row.values(), default=None))
        self.df = df

    def results_dag(
        self,
        view_independences: bool = True,
        html: bool = False,
        layout_engine: str = None,
    ) -> nx.DiGraph:
        """
        View causal test results as a graph.

        :param view_independences: Whether to display failed independence tests (defaults to True).
        :param html: Whether to include html representations of the causal effect. (Defaults to false)
        :param layout_engine: The layout engine to use. (Defaults to None for concise output)
                              See https://graphviz.org/docs/layouts/ for a list of supported engines.
        """
        result_dag = nx.DiGraph(ignore_cycles=True)
        result_dag.add_nodes_from(self.dag.nodes)
        result_dag.add_edges_from(self.dag.edges)

        if self.df is not None and self.results:
            # Add in edges for non-passing independence tests
            if view_independences:
                result_dag.add_edges_from(
                    filter(
                        lambda edge: edge not in result_dag.edges,
                        self.df.loc[
                            self.df["result.outcome"] != TestOutcome.PASS.name,
                            ["estimator.treatment_variable", "estimator.outcome_variable"],
                        ].itertuples(index=False),
                    ),
                    style="dashed",
                )

            for _, test in self.df.iterrows():
                treatment_variable = test["estimator.treatment_variable"]
                outcome_variable = test["estimator.outcome_variable"]

                if (treatment_variable, outcome_variable) in result_dag.edges or (
                    view_independences and test["result.outcome"] != TestOutcome.PASS.name
                ):
                    result_dag[treatment_variable][outcome_variable]["label"] = test["result.effect_direction"]
                    result_dag[treatment_variable][outcome_variable]["color"] = self.colour_map[
                        getattr(TestOutcome, test["result.outcome"])
                    ]
                    result_dag[treatment_variable][outcome_variable]["fontcolor"] = self.colour_map[
                        getattr(TestOutcome, test["result.outcome"])
                    ]
                    if (
                        html
                        and "result.effect_estimate.effect_estimate" in self.df
                        and "result.effect_estimate.ci_low" in self.df
                        and "result.effect_estimate.ci_low" in self.df
                    ):
                        effect_estimate = pd.DataFrame(
                            {
                                "ci_low": test["result.effect_estimate.ci_low"],
                                "estimate": test["result.effect_estimate.effect_estimate"],
                                "ci_high": test["result.effect_estimate.ci_high"],
                            }
                        )
                        result_dag[test["estimator.treatment_variable"]][test["estimator.outcome_variable"]][
                            "title"
                        ] = f"<{effect_estimate.to_html()}>"

        if layout_engine:
            result_dag = nx.drawing.nx_pydot.from_pydot(
                pydot.graph_from_dot_data(
                    nx.drawing.nx_pydot.to_pydot(result_dag).create_dot(prog=layout_engine).decode("utf-8")
                )[0]
            )

        return result_dag

    def add_discrete_legend(self, plot: GraphPlot, element: hv.Graph):
        """
        Add a pass/fail/inestimable legend to plots.

        :param plot: The current plot figure.
        :param element: The Graph element.
        """
        elements = [
            f'<span style="color: {self.colour_map[TestOutcome.PASS]};">■ Pass</span>',
            f'<span style="color: {self.colour_map[TestOutcome.INESTIMABLE]};">■ Inestimable</span>',
            f'<span style="color: {self.colour_map[TestOutcome.FAIL]};">■ Fail</span>',
        ]
        if isinstance(element, hv.Graph):
            elements += [
                '<span style= "margin-left: 5ex;">— Expected dependent</span>',
                "<span>--- Expected indepednent</span>",
            ]
        legend_html = (
            '<div style="text-align: center; font-family: sans-serif; font-size: 14px; padding: 4px;">'
            + "\n".join(elements)
            + "</div>"
        )

        div = Div(text=legend_html)
        plot.state.add_layout(div, "above")

    def interactive_results_dag(self, **kwargs) -> hv.Overlay:
        """
        Generate an interactive holoview graph of the causal DAG showing failing tests.

        :returns: Inveractive holoviews graph.
        """
        results = self.results_dag(html=True, layout_engine="dot")

        node_positions = {}
        for node, attributes in results.nodes(data=True):
            x, y = map(float, attributes["pos"].strip('"').split(","))
            node_positions[node] = (x, y)

        # Build the edges
        edges_df = pd.DataFrame(
            [
                {"source": u, "target": v} | {k: v.strip('"') for k, v in data.items()}
                for u, v, data in results.edges(data=True)
            ]
        )

        edges_df["trimmed_path"] = edges_df[["source", "target"]].apply(
            lambda row: edge_spline(
                dot_pos=results[row["source"]][row["target"]]["pos"],
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

        hooks = [style_graph_hook]
        if self.df is not None and self.results:
            hooks.append(self.add_discrete_legend)

        if "label" in edges_df.columns:
            edges_df["label"] = edges_df["label"].fillna("")

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
            edge_line_dash="style" if "style" in edges_df else "solid",
            edge_line_width=1.5,
            edge_color="color" if "color" in edges_df else "black",
            edge_hover_line_color="color",
            hooks=hooks,
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
                ),
                "fullscreen",
            ],
            inspection_policy="edges",
            **kwargs,
        )

        # Label layers
        edges_df["label"] = "" if "label" not in edges_df else edges_df["label"]
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

    def _get_split_category_order(self, category_col: str, value_col: str) -> list:
        """
        Partitions categories into two groups relative to overall_median:
        - Group median < overall_median: sorted by category min (ascending).
        - Group median >= overall_median: sorted by category max (ascending).

        :param category_col: The column to group by.
        :param value_col: The column containing the value to display.

        :returns: sorted list of the values in category_col.
        """

        stats = self.df.groupby(category_col)[value_col].agg(["median", "min", "max"]).reset_index()

        # Sort lower half by min value, upper half by max value
        lower_order = (
            stats[stats["median"] < stats["median"].median()]
            .sort_values(by="min", ascending=True)[category_col]
            .tolist()
        )
        upper_order = (
            stats[stats["median"] >= stats["median"].median()]
            .sort_values(by="max", ascending=True)[category_col]
            .tolist()
        )

        return lower_order + upper_order

    def sort_df_by_median_split(
        self,
        value_col: str,
        treatment_col: str = "estimator.treatment_variable",
        outcome_col: str = "estimator.outcome_variable",
    ) -> pd.DataFrame:
        """
        Sorts treatment and outcome variables relative to the overall median.

        :param value_col: The column containing the value to display.
        :param treatment_col: The column containing the data to display on the x-axis.
        :param outcome_col: The column containing the data to display on the y-axis.

        :returns: Sorted dataframe containing the treatment_col, outcome_col, and vdims.
        """

        # Fill missing (treatment, outcome) combinations with empty rows
        # We need this to ensure that it's possible to obtain the correct ordering in the heatmap
        df = (
            self.df.copy()
            .set_index([treatment_col, outcome_col])
            .reindex(
                pd.MultiIndex.from_product(
                    [
                        self.df[treatment_col].dropna().unique(),
                        self.df[outcome_col].dropna().unique(),
                    ],
                    names=[treatment_col, outcome_col],
                )
            )
            .reset_index()
        )

        # Apply ordered categoricals so HoloViews maps the axes to these index positions
        df[treatment_col] = pd.Categorical(
            df[treatment_col], categories=self._get_split_category_order(treatment_col, value_col), ordered=True
        )
        df[outcome_col] = pd.Categorical(
            df[outcome_col], categories=self._get_split_category_order(outcome_col, value_col), ordered=True
        )

        df = df.sort_values(by=[treatment_col, outcome_col]).dropna(subset=[value_col])

        # Need to convert the values back to strings, otherwise holoviz thinks they're not unique
        df[treatment_col] = df[treatment_col].astype(str)
        df[outcome_col] = df[outcome_col].astype(str)

        return df

    def data_adequacy_heatmap(self, **kwargs) -> hv.HeatMap:
        """
        Visualise data adequacy as an adjacency matrix heatmap of the kurtosis.
        """

        data = self.sort_df_by_median_split(value_col="result.adequacy.kurtosis")

        # Get data bounds
        vmin = data["result.adequacy.kurtosis"].min()
        vmax = data["result.adequacy.kurtosis"].max()

        # Calculate relative zero position on the colourmap (0.0 to 1.0)
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
            data,
            kdims=[
                ("treatment_variable_inx", "Treatment variable"),
                ("outcome_variable_inx", "Outcome variable"),
            ],
            vdims=[
                ("result.adequacy.kurtosis", "Kurtosis"),
                "estimator.treatment_variable",
                "estimator.outcome_variable",
                "result.adequacy.kurtosis",
            ],
        ).opts(
            cmap=asymmetric_cmap,
            clim=(vmin, vmax),
            clipping_colors={"NaN": "grey"},  # Grey out invalid tests
            colorbar=True,
            xrotation=45,
            data_aspect=1,
            xticks=self.xticks,
            yticks=self.yticks,
            xlim=(-0.5, len(self.xticks) - 0.5),
            ylim=(-0.5, len(self.yticks) - 0.5),
            hooks=[add_colorbar_annotations],
            hover_tooltips=[
                ("Treatment variable", "@estimator.treatment_variable"),
                ("Outcome variable", "@estimator.outcome_variable"),
                ("Adequacy", "@{result.adequacy.kurtosis}"),
            ],
            tools=[
                "hover",
                "fullscreen",
            ],
            xlabel="Treatment variable",
            ylabel="Outcome variable",
            clabel="Causal test adequacy",
            title="Data Adequacy",
            **kwargs,
        )

    def dag_adequacy_heatmap(self, **kwargs) -> hv.HeatMap:
        """
        Visualise dag adequacy as an adjacency matrix heatmap of the percentage of passing test cases.
        """
        data = self.sort_df_by_median_split(value_col="result.adequacy.passing")

        # Turn passing test cases into a percentage
        data["result.adequacy.passing"] = (
            data["result.adequacy.passing"] / data["result.adequacy.bootstrap_size"]
        ) * 100

        return hv.HeatMap(
            data,
            kdims=[
                ("treatment_variable_inx", "Treatment variable"),
                ("outcome_variable_inx", "Outcome variable"),
            ],
            vdims=[
                ("result.adequacy.passing", "Passing (%)"),
                "estimator.treatment_variable",
                "estimator.outcome_variable",
                "result.adequacy.passing",
            ],
        ).opts(
            cmap="RdYlGn",
            clim=(0, 100),
            clipping_colors={"NaN": "grey"},  # Grey out invalid tests
            colorbar=True,
            xrotation=45,
            data_aspect=1,
            xticks=self.xticks,
            yticks=self.yticks,
            xlim=(-0.5, len(self.xticks) - 0.5),
            ylim=(-0.5, len(self.yticks) - 0.5),
            hover_tooltips=[
                ("Treatment variable", "@estimator.treatment_variable"),
                ("Outcome variable", "@estimator.outcome_variable"),
                ("Passing", "@result.adequacy.passing%"),
            ],
            tools=[
                "hover",
                "fullscreen",
            ],
            xlabel="Treatment variable",
            ylabel="Outcome variable",
            clabel="Percentage passing test cases",
            title="DAG Adequacy",
            **kwargs,
        )

    def outcome_adjacency(self, **kwargs) -> hv.HeatMap:
        """
        Visualise causal test results as an adjacency matrix.
        """
        data = self.sort_df_by_median_split(value_col="result.outcome.value")

        return hv.HeatMap(
            data,
            kdims=[
                ("treatment_variable_inx", "Treatment variable"),
                ("outcome_variable_inx", "Outcome variable"),
            ],
            vdims=[
                ("result.outcome", "Outcome"),
                "estimator.treatment_variable",
                "estimator.outcome_variable",
                "result.outcome",
            ],
        ).opts(
            cmap={k.name: v for k, v in self.colour_map.items()},
            clipping_colors={"NaN": "grey"},
            hover_tooltips=[
                ("Treatment variable", "@estimator.treatment_variable"),
                ("Outcome variable", "@estimator.outcome_variable"),
                ("Test outcome", "@result.outcome"),
            ],
            tools=[
                "hover",
                "fullscreen",
            ],
            xlabel="Treatment variable",
            ylabel="Outcome variable",
            hooks=[self.add_discrete_legend],
            xticks=self.xticks,
            yticks=self.yticks,
            data_aspect=1,
            xlim=(-0.5, len(self.xticks) - 0.5),
            ylim=(-0.5, len(self.yticks) - 0.5),
            xrotation=45,
            **kwargs,
        )
