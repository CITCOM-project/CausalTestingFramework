"""This module contains the BaseTestCase dataclass, which stores the information required for identification"""

from dataclasses import dataclass
from causal_testing.specification.variable import Variable
from causal_testing.testing.effect import Effect


@dataclass(frozen=True)
class BaseTestCase:
    """
    A base causal test case represents the relationship of an edge on a causal DAG.
    :param treatment_variable: A causal variable representing the treatment/control variable
    :param outcome_variable: A causal variable representing the outcome/output variable
    :param effect: A string representing the effect, current support effects are 'direct' and 'total'
    """

    treatment_variable: Variable
    outcome_variable: Variable
    effect: str = Effect.TOTAL.value

    def __post_init__(self):
        if self.treatment_variable == self.outcome_variable:
            raise ValueError(f"Treatment variable {self.treatment_variable} cannot also be the outcome variable.")
