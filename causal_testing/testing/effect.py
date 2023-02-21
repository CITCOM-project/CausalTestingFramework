"""This module contains the Enum Class Effect, allowing effect types TOTAL and DIRECT"""

from enum import Enum


class Effect(Enum):
    """An enumeration of allowable effect types."""

    TOTAL = "total"
    DIRECT = "direct"
