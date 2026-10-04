"""
This module implements various helper functions for plotting geometry.
"""

import re

import holoviews as hv
import numpy as np
from bokeh.models import Arrow, Ellipse, NormalHead, Title
from holoviews.plotting.bokeh.graphs import GraphPlot
from scipy.interpolate import make_splprep  # pylint: disable=E0611


def parse_dot_spline(pos_str: str) -> list[tuple[float, float]]:
    """
    Parse Graphviz 'pos' string into control points.
    See https://graphviz.org/docs/attr-types/splineType for syntax details.
    NOTE: This will ignore segments separated by ";", but this shouldn't be a problem in our limited context.

    :param pos_str: The graphviz position string representing the list of control points.

    :returns: list of the parsed control points.
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


def add_colorbar_annotations(plot: GraphPlot, element: hv.Graph):  # pylint: disable=unused-argument
    """
    Hook to add "Suspiciously (un)stable" annotations for the data adequacy plot.

    :param plot: The current plot figure.
    :param element: The Graph element (not used).
    """
    fig = plot.state

    # Add space for the titles
    colorbar = plot.handles["colorbar"]
    colorbar.styles = {"margin-right": "30px"}

    bottom_label = Title(
        text="Suspiciously unstable", standoff=-10, text_font_size="10pt", text_align="right", vertical_align="top"
    )
    top_label = Title(
        text="Suspiciously stable", standoff=-60, text_font_size="10pt", text_align="right", vertical_align="bottom"
    )
    fig.add_layout(bottom_label, "right")
    fig.add_layout(top_label, "right")


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
    node_source.data["height"] = [32] * len(element.nodes.data)  # 32 px high looks about right

    # Define primary Ellipse glyph
    graph_renderer.node_renderer.glyph = Ellipse(
        width="width",
        height="height",
        fill_color="white",
        line_color="gray",
    )

    # Define hover / inspection Ellipse glyph (prevents reverting to green circles)
    graph_renderer.node_renderer.hover_glyph = graph_renderer.node_renderer.glyph.clone()
    graph_renderer.node_renderer.hover_glyph.fill_color = "skyblue"

    # Stops edges disappearing when you over over them
    graph_renderer.edge_renderer.hover_glyph = graph_renderer.edge_renderer.glyph.clone()

    # Add Arrowheads with matching edge colors
    for _, row in element.data.iterrows():
        color = row.get("color", "black")
        arrow = Arrow(
            end=NormalHead(fill_color=color, line_color=color, size=8),
            x_start=row["arrow_starts_x"],
            y_start=row["arrow_starts_y"],
            x_end=row["arrow_ends_x"],
            y_end=row["arrow_ends_y"],
            line_alpha=0,
        )
        fig.add_layout(arrow)
