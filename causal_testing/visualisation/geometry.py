"""
This module implements various helper functions for plotting geometry.
"""

import re

import holoviews as hv
import numpy as np
import pandas as pd
from bokeh.models import Arrow, Ellipse, NormalHead
from holoviews.plotting.bokeh.graphs import GraphPlot
from scipy.interpolate import make_splprep  # pylint: disable=E0611


def _get_split_category_order(df: pd.DataFrame, col: str, value_col: str) -> list:
    """
    Partitions categories into two groups relative to overall_median:
    - Group median < overall_median: sorted by category min (ascending).
    - Group median >= overall_median: sorted by category max (ascending).
    """
    stats = df.groupby(col)[value_col].agg(["median", "min", "max"]).reset_index()

    # Sort lower half by min value, upper half by max value
    lower_order = stats[stats["median"] < stats["median"].median()].sort_values(by="min", ascending=True)[col].tolist()
    upper_order = stats[stats["median"] >= stats["median"].median()].sort_values(by="max", ascending=True)[col].tolist()

    return lower_order + upper_order


def sort_df_by_median_split(
    df: pd.DataFrame,
    value_col: str,
    treatment_col: str = "estimator.treatment_variable",
    outcome_col: str = "estimator.outcome_variable",
    vdims: list[str] = None,
) -> pd.DataFrame:
    """
    Sorts treatment and outcome variables relative to the overall median kurtosis.
    """

    vdims = [] if vdims is None else vdims

    # Fill missing (treatment, outcome) combinations with empty rows
    # We need this to ensure that it's possible to obtain the correct ordering in the heatmap
    df = (
        df.set_index([treatment_col, outcome_col])
        .reindex(
            pd.MultiIndex.from_product(
                [
                    df[treatment_col].dropna().unique(),
                    df[outcome_col].dropna().unique(),
                ],
                names=[treatment_col, outcome_col],
            )
        )
        .reset_index()
    )

    # Apply ordered categoricals so HoloViews maps the axes to these index positions
    df_sorted = df[[treatment_col, outcome_col, value_col] + vdims].copy()
    df_sorted[treatment_col] = pd.Categorical(
        df[treatment_col], categories=_get_split_category_order(df, treatment_col, value_col), ordered=True
    )
    df_sorted[outcome_col] = pd.Categorical(
        df[outcome_col], categories=_get_split_category_order(df, outcome_col, value_col), ordered=True
    )

    df_sorted = df_sorted.sort_values(by=[treatment_col, outcome_col]).dropna()

    # Need to convert the values back to strings, otherwise holoviz thinks they're not unique
    df_sorted[treatment_col] = df_sorted[treatment_col].astype(str)
    df_sorted[outcome_col] = df_sorted[outcome_col].astype(str)
    return df_sorted


def parse_dot_spline(pos_str: str) -> list[tuple[float, float]]:
    """
    Parse Graphviz 'pos' string into control points.
    See https://graphviz.org/docs/attr-types/splineType for syntax details.
    NOTE: This will ignore segments separated by ";", but this shouldn't be a problem in our limited context.

    :param pos_str: The graphviz position string representing the list of control points.
    """
    end_point = None
    points = []

    for point_type, x, y in re.findall(r"(?:(s|e),)?(\d+(?:.\d+)?),(\d+(?:\.\d+)?)", pos_str):
        x, y = float(x), float(y)
        if point_type == "s":
            points = [(x, y)] + points
        elif point_type == "e":
            end_point = x, y
        else:
            points.append((x, y))

    if end_point:
        points.append(end_point)

    # Remove consecutive duplicate points
    points = np.array(points)
    mask = np.ones(len(points), dtype=bool)
    mask[1:] = np.any(np.diff(points, axis=0) != 0, axis=1)
    return points[mask]


def edge_spline(
    dot_pos: str,
    target_node_centre: tuple[float, float],
    target_node_width: float,
    target_node_height: float = 16,
    num_points: int = 100,
    shorten: float = 0.05,
):
    """
    Generate an edge spline from the given control points, trimmed at source & target ellipse boundaries.

    :param dot_pos: The DOT `pos` attribute of the edge, representing the control points of the spline.
    :param target_node_centre: The coordinates of the centre of the target node.
    :param target_node_width: The width of the target_node.
    :param target_node_height: The height of the target_node (defaults to 16).
    :param num_points: The number of spline points to generage (defaults to 100).
    :param shorten: The percentage of the line to shorten by to allow for the arrow head (defaults to 5%).
    """
    b_spline, _ = make_splprep(parse_dot_spline(dot_pos).T, k=3, s=0)

    # Evaluate the BSpline object at uniform parametric points
    smooth_points = b_spline(np.linspace(0, 1, num_points)).T

    # Calculate angle and boundary radius for end node
    dx_e = target_node_centre[0] - smooth_points[-round(num_points * shorten)][0]
    dy_e = target_node_centre[1] - smooth_points[-round(num_points * shorten)][1]
    angle_end = np.arctan2(dy_e, dx_e)
    r_end = (target_node_width * target_node_height) / np.sqrt(
        (target_node_width * np.sin(angle_end)) ** 2 + (target_node_height * np.cos(angle_end)) ** 2
    )

    dists_end = np.hypot(smooth_points[:, 0] - target_node_centre[0], smooth_points[:, 1] - target_node_centre[1])
    end_idx = len(smooth_points) - np.searchsorted(dists_end[::-1], r_end)

    trimmed_path = smooth_points[:end_idx]
    return trimmed_path


def node_width(label: str, text_font_size: int = 9, padding: int = 24) -> float:
    """
    Calculate the width that a node should be to accomadate the label.

    :param label: The node label.
    :param text_font_size: The font size in pt.
    :param padding: Node inner padding in pt.
    """
    return len(label) * text_font_size + padding


def style_graph_hook(plot: GraphPlot, element: hv.Graph):
    """
    Hook to properly style nodes to be an ellipse of the correct size.

    :param plot: The current plot figure.
    :param element: The Graph element.
    """
    fig = plot.handles["plot"]
    graph_renderer = plot.handles["glyph_renderer"]

    # Supply widths and heights to the node source
    node_source = graph_renderer.node_renderer.data_source
    node_source.data["width"] = element.nodes.data["node_id"].apply(node_width)
    node_source.data["height"] = [32] * len(element.nodes.data)

    # Define primary Ellipse glyph
    graph_renderer.node_renderer.glyph = Ellipse(
        width="width",
        height="height",
        fill_color="white",
        line_color="gray",
    )

    # Define hover / inspection Ellipse glyph (prevents reverting to green circles)
    graph_renderer.node_renderer.hover_glyph = Ellipse(
        width="width",
        height="height",
        fill_color="skyblue",
        line_color="gray",
    )

    # Add Arrowheads with matching edge colors
    for _, row in element.data.iterrows():
        color = row["color"]
        arrow = Arrow(
            end=NormalHead(fill_color=color, line_color=color, size=8),
            x_start=row["arrow_starts_x"],
            y_start=row["arrow_starts_y"],
            x_end=row["arrow_ends_x"],
            y_end=row["arrow_ends_y"],
            line_alpha=0,
        )
        fig.add_layout(arrow)
