"""
libpsam - Python bindings

Fast, lightweight sequence prediction using learned token associations.
"""

from .core import PSAM, PSAMError, ModelStats, ExplainTerm, ExplainResult, is_library_available

__version__ = "0.1.0"
__all__ = [
    "PSAM",
    "PSAMError",
    "ModelStats",
    "ExplainTerm",
    "ExplainResult",
    "is_library_available",
]
