"""A library of dataclasses for properties implied by a causal DAG."""
from dataclasses import dataclass
from typing import Iterable

from causal_specification import Node


@dataclass(order=True, frozen=True)
class ConditionalIndependenceRelation:
    treatment: Node
    outcome: Node
    adjustment_set: Iterable[Node]

    def __str__(self):
        return f"{self.treatment} _||_ {self.outcome} | {self.adjustment_set}"


@dataclass(order=True, frozen=True)
class CausalRelation:
    treatment: Node
    outcome: Node
    adjustment_set: Iterable[Node] = None

    def __str__(self):
        formatted_string = f"{self.treatment} --> {self.outcome}"
        if self.adjustment_set:
            formatted_string += f" | {self.adjustment_set}"
        return formatted_string
