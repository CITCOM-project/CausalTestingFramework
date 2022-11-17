from dataclasses import dataclass

@dataclass
class BaseCausalTest:
    """
    A base causal test case represents the relationship of an edge on a causal DAG.
    """
    treatment_variable: str
    outcome_variable: str
    effect: str = 'total'

