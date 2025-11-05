"""
libpsam - Python bindings

Fast, lightweight sequence prediction using learned token associations.
"""

from .core import (
    PSAM,
    PSAMGenerator,
    LayeredComposite,
    PSAMError,
    ModelStats,
    ExplainTerm,
    ExplainResult,
    CompositeLayerInfo,
    LogitTransform,
    SamplerConfig,
    ResidualConfig,
    SalienceConfig,
    load_composite,
    save_composite_manifest,
    is_library_available,
)

__version__ = "0.1.0"
__all__ = [
    "PSAM",
    "PSAMGenerator",
    "LayeredComposite",
    "PSAMError",
    "ModelStats",
    "ExplainTerm",
    "ExplainResult",
    "CompositeLayerInfo",
    "LogitTransform",
    "SamplerConfig",
    "ResidualConfig",
    "SalienceConfig",
    "load_composite",
    "save_composite_manifest",
    "is_library_available",
]
