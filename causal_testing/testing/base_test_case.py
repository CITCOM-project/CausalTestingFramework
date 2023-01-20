from dataclasses import dataclass
from causal_testing.specification.variable import Variable
from causal_testing.testing.effect import Effect


@dataclass(frozen=True)
class BaseTestCase:
    """
    A base causal test case represents the relationship of an edge on a causal DAG.
    """

    treatment_variable: Variable
    outcome_variable: Variable
    effect: str = Effect.total.value
