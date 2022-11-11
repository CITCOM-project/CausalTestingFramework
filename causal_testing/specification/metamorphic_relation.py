from dataclasses import dataclass
from abc import abstractmethod
from typing import Iterable
from itertools import combinations
import numpy as np
import pandas as pd

from causal_testing.specification.causal_specification import CausalDAG, Node

@dataclass(order=True)
class MetamorphicRelation:
    """Class representing a metamorphic relation."""
    treatment_var: Node
    output_var: Node
    adjustment_vars: Iterable[Node]
    dag: CausalDAG
    tests: Iterable = None

    def generate_follow_up(self,
                           n_tests: int,
                           min_val: float,
                           max_val: float,
                           seed: int = 0):
        """Generate numerical follow-up input configurations."""
        np.random.seed(seed)

        # Get set of variables to change, excluding the treatment itself
        variables_to_change = set([node for node in self.dag.graph.nodes if
                                   self.dag.graph.in_degree(node) == 0])
        if self.adjustment_vars:
            variables_to_change |= set(self.adjustment_vars)
        if self.treatment_var in variables_to_change:
            variables_to_change.remove(self.treatment_var)

        # Assign random numerical values to the variables to change
        test_inputs = pd.DataFrame(
            np.random.randint(min_val, max_val,
                              size=(n_tests, len(variables_to_change))
                              ),
            columns=sorted(variables_to_change)
        )

        # Enumerate the possible source, follow-up pairs for the treatment
        candidate_source_follow_up_pairs = np.array(
            list(combinations(range(int(min_val), int(max_val+1)), 2))
        )

        # Sample without replacement from the possible source, follow-up pairs
        sampled_source_follow_up_indices = np.random.choice(
            candidate_source_follow_up_pairs.shape[0], n_tests, replace=False
        )

        follow_up_input = f"{self.treatment_var}\'"
        source_follow_up_test_inputs = pd.DataFrame(
            candidate_source_follow_up_pairs[sampled_source_follow_up_indices],
            columns=sorted([self.treatment_var] + [follow_up_input])
        )
        source_test_inputs = source_follow_up_test_inputs[[self.treatment_var]]
        follow_up_test_inputs = source_follow_up_test_inputs[[follow_up_input]]
        follow_up_test_inputs.rename({follow_up_input: self.treatment_var})

        # TODO: Add a metamorphic test dataclass that stores these attributes
        self.tests = list(
            zip(
                source_test_inputs.to_dict(orient="records"),
                follow_up_test_inputs.to_dict(orient="records"),
                test_inputs.to_dict(orient="records") if not test_inputs.empty
                else [{}] * len(source_test_inputs),
                [self.output_var] * len(source_test_inputs),
                [str(self)] * len(source_test_inputs)
            )
        )

    @abstractmethod
    def test_oracle(self):
        """A test oracle i.e. a method that checks correctness of a test."""
        ...

    @abstractmethod
    def execute_test(self):
        """Execute a test for this metamorphic relation."""
        ...


@dataclass(order=True)
class ShouldCause(MetamorphicRelation):
    """Class representing a should cause metamorphic relation."""

    def test_oracle(self):
        pass

    def execute_test(self):
        pass

    def __str__(self):
        formatted_str = f"{self.treatment_var} --> {self.output_var}"
        if self.adjustment_vars:
            formatted_str += f" | {self.adjustment_vars}"
        return formatted_str
