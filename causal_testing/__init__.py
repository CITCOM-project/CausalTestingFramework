"""
This is the CausalTestingFramework Module
It contains 5 subpackages:
estimation
specification
surrogate
testing
utils
"""

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

try:
    from ._version import version as __version__
except ImportError:
    __version__ = "unknown"
