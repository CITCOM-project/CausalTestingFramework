from abc import ABC
import networkx as nx


class CausalDAG(nx.DiGraph):
    """ A causal DAG is a directed acyclic graph in which nodes represent random variables and edges represent causality
        between a pair of random variables. We implement a CausalDAG as a networkx DiGraph with an additional check that
        ensures it is acyclic. A CausalDAG must be specified as a dot file. """

    def __init__(self, dot_path=None, **attr):
        super().__init__(**attr)
        if dot_path:
            self.graph = nx.DiGraph(nx.drawing.nx_agraph.read_dot(dot_path))
        else:
            self.graph = nx.DiGraph()

        # Raise an exception if the DAG contains a cycle
        if not self.is_acyclic():
            raise nx.HasACycle("Invalid Causal DAG: contains a cycle.")

    def add_edge(self, u_of_edge, v_of_edge, **attr):
        """
        Add an edge to the causal DAG. Overrides the default networkx method to prevent users from adding a cycle.
        :param u_of_edge: From node
        :param v_of_edge: To node
        :param attr: Attributes
        """
        self.graph.add_edge(u_of_edge, v_of_edge, **attr)
        if not self.is_acyclic():
            raise nx.HasACycle("Invalid Causal DAG: contains a cycle.")

    def is_acyclic(self):
        """
        Checks if the graph is acyclic.
        :return: True if acyclic, False otherwise.
        """
        return not list(nx.simple_cycles(self.graph))

    def __str__(self):
        return f'Nodes: {self.graph.nodes}\nEdges: {self.graph.edges}'


class Scenario(ABC):
    pass


class CausalSpecification(ABC):

    def __init__(self, scenario: Scenario, causal_dag: CausalDAG):
        self.scenario = scenario
        self.causal_dag = causal_dag
