"""
libpsam - Python bindings

Fast, lightweight sequence prediction using learned token associations.
"""

from .core import (
    PSAM,
    LayeredComposite,
    PSAMError,
    ModelStats,
    ExplainTerm,
    ExplainResult,
    CompositeLayerInfo,
    LogitTransform,
    SamplerConfig,
    load_composite,
    save_composite_manifest,
    is_library_available,
)

__version__ = "0.1.0"
__all__ = [
    "PSAM",
    "LayeredComposite",
    "PSAMError",
    "ModelStats",
    "ExplainTerm",
    "ExplainResult",
    "CompositeLayerInfo",
    "LogitTransform",
    "SamplerConfig",
    "load_composite",
    "save_composite_manifest",
    "is_library_available",
]
