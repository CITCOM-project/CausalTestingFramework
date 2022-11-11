from dataclasses import dataclass
from abc import abstractmethod


@dataclass(order=True, frozen=True)
class MetamorphicRelation:
    """Class representing a metamorphic relation."""

    @abstractmethod
    def generate_follow_up(self, source_input_configuration):
        """Generate a follow-up input configuration from a given source input
           configuration."""
        ...

    @abstractmethod
    def test_oracle(self):
        """A test oracle i.e. a method that checks correctness of a test."""
        ...

    @abstractmethod
    def execute_test(self):
        """Execute a test for this metamorphic relation."""
        ...


@dataclass(order=True, frozen=True)
class ShouldCause(MetamorphicRelation):
    """Class representing a should cause metamorphic relation."""

    def generate_follow_up(self, source_input_configuration):
        pass

    def test_oracle(self):
        pass

    def execute_test(self):
        pass

