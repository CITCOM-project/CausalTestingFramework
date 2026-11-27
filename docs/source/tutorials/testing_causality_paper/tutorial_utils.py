"""
Shared helpers for the "Testing Causality in Scientific Modelling Software" tutorials.
"""

import networkx as nx
from IPython.display import Image, display


def render_dag(dag: nx.DiGraph, rankdir: str = "LR"):
    """
    Display a causal DAG as an image in a Jupyter notebook.

    :param dag: The causal DAG to render.
    :param rankdir: The direction in which to lay out the graph ("LR" for left-to-right, "TB" for top-to-bottom).
    """
    graph = nx.nx_pydot.to_pydot(dag)
    graph.set("rankdir", rankdir)
    display(Image(graph.create_png(prog="dot")))
