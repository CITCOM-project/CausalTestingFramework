from abc import ABC, abstractmethod
import networkx as nx


class CausalSpecification(ABC):

    def __init__(self, scenario, causal_dag):
        self.scenario = scenario
        self.causal_dag = causal_dag


class CausalDAG(ABC):

    def __init__(self, dot_path=None):
        self.vertices = []
        self.edges = []
        if dot_path:
            self.load_dot(dot_path)

    def load_dot(self, dot_path):
        dag = nx.DiGraph(nx.drawing.nx_agraph.read_dot(dot_path))
        self.edges = dag.edges
        self.vertices = dag.nodes

    def d_separates(self, variables):
        pass

    def __str__(self):
        return(f'Vertices: {self.vertices}\nEdges: {self.edges}')

