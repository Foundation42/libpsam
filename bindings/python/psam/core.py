"""
Core PSAM bindings using ctypes
"""

import ctypes
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple
from dataclasses import dataclass

import numpy as np

# Error codes
class PSAMErrorCode:
    OK = 0
    NULL_PARAM = -1
    INVALID_CONFIG = -2
    OUT_OF_MEMORY = -3
    IO = -4
    INVALID_MODEL = -5
    NOT_TRAINED = -6
    LAYER_NOT_FOUND = -7


class PSAMError(Exception):
    """PSAM library error"""
    pass


@dataclass
class ModelStats:
    """Model statistics"""
    vocab_size: int
    row_count: int
    edge_count: int
    total_tokens: int
    memory_bytes: int


class LogitTransform:
    """Logit transform modes for temperature sampling"""
    RAW = 0
    ZSCORE = 1
    CALIBRATED = 2
    LEGACY = 3


@dataclass
class SamplerConfig:
    """Sampler configuration for temperature control"""
    transform: int = LogitTransform.ZSCORE
    temperature: float = 1.0
    top_k: int = 0
    top_p: float = 0.95
    seed: int = None


@dataclass
class ResidualConfig:
    """Residual activation configuration for deferred associations"""
    max_lookahead: int = 3
    residual_decay: float = 0.8
    residual_blend: float = 0.4
    enable: bool = True


@dataclass
class SalienceConfig:
    """Salience tracking configuration for pop-out anchors"""
    max_anchors: int = 16
    ewma_freq_halflife: float = 128.0
    ewma_contrib_halflife: float = 64.0
    eta: float = 1.0
    kappa: float = 0.25
    beta: float = 0.3
    pop_decay_distance: float = 256.0
    min_salience: float = 0.1
    enable: bool = True


# Struct definitions matching C API
PSAM_LAYER_ID_MAX = 64

class PSAMSampler(ctypes.Structure):
    _fields_ = [
        ("transform", ctypes.c_uint32),
        ("temperature", ctypes.c_float),
        ("top_k", ctypes.c_int32),
        ("top_p", ctypes.c_float),
        ("seed", ctypes.c_uint64),
    ]


class PSAMResidualConfig(ctypes.Structure):
    _fields_ = [
        ("max_lookahead", ctypes.c_int32),
        ("residual_decay", ctypes.c_float),
        ("residual_blend", ctypes.c_float),
        ("enable", ctypes.c_bool),
    ]


class PSAMSalienceConfig(ctypes.Structure):
    _fields_ = [
        ("max_anchors", ctypes.c_int32),
        ("ewma_freq_halflife", ctypes.c_float),
        ("ewma_contrib_halflife", ctypes.c_float),
        ("eta", ctypes.c_float),
        ("kappa", ctypes.c_float),
        ("beta", ctypes.c_float),
        ("pop_decay_distance", ctypes.c_float),
        ("min_salience", ctypes.c_float),
        ("enable", ctypes.c_bool),
    ]


class PSAMPrediction(ctypes.Structure):
    _fields_ = [
        ("token_id", ctypes.c_uint32),
        ("score", ctypes.c_float),
        ("raw_strength", ctypes.c_float),
        ("support_count", ctypes.c_uint16),
        ("_reserved", ctypes.c_uint16),
        ("calibrated_prob", ctypes.c_float),
    ]


class PSAMExplainTerm(ctypes.Structure):
    _fields_ = [
        ("source_token", ctypes.c_uint32),
        ("rel_offset", ctypes.c_int16),
        ("weight_ppmi", ctypes.c_float),
        ("idf", ctypes.c_float),
        ("decay", ctypes.c_float),
        ("contribution", ctypes.c_float),
    ]


class PSAMExplainResult(ctypes.Structure):
    _fields_ = [
        ("candidate", ctypes.c_uint32),
        ("total_score", ctypes.c_float),
        ("bias_score", ctypes.c_float),
        ("term_count", ctypes.c_int32),
    ]


class PSAMCompositeLayerInfo(ctypes.Structure):
    _fields_ = [
        ("id", ctypes.c_char * PSAM_LAYER_ID_MAX),
        ("weight", ctypes.c_float),
        ("bias", ctypes.c_float),
    ]


class PSAMCompositeLayerFile(ctypes.Structure):
    _fields_ = [
        ("id", ctypes.c_char_p),
        ("weight", ctypes.c_float),
        ("path", ctypes.c_char_p),
    ]


@dataclass
class ExplainTerm:
    """Explanation term showing why a token was predicted"""
    source: int
    offset: int
    weight: float
    idf: float
    decay: float
    contribution: float


@dataclass
class ExplainResult:
    """Full explanation response for a candidate token"""
    candidate: int
    total: float
    bias: float
    term_count: int
    terms: List[ExplainTerm]


@dataclass
class CompositeLayerInfo:
    """Metadata for a layer inside a composite"""
    layer_id: str
    weight: float
    bias: float = 0.0


class PSAMStatsStruct(ctypes.Structure):
    _fields_ = [
        ("vocab_size", ctypes.c_uint32),
        ("row_count", ctypes.c_uint32),
        ("edge_count", ctypes.c_uint64),
        ("total_tokens", ctypes.c_uint64),
        ("memory_bytes", ctypes.c_size_t),
    ]


# Library loading
_lib = None


def _load_library():
    """Load libpsam shared library"""
    global _lib

    if _lib is not None:
        return _lib

    # Try LIBPSAM_PATH environment variable
    lib_path = os.environ.get("LIBPSAM_PATH")

    if lib_path and os.path.exists(lib_path):
        try:
            _lib = ctypes.CDLL(lib_path)
            _configure_library(_lib)
            return _lib
        except OSError as e:
            raise PSAMError(f"Failed to load library from {lib_path}: {e}")

    # Try common locations
    search_paths = []

    # Package directory
    pkg_dir = Path(__file__).parent
    search_paths.extend([
        pkg_dir / "libpsam.so",
        pkg_dir / "libpsam.dylib",
        pkg_dir / "libpsam.dll",
    ])

    # Build directory (for development)
    repo_root = pkg_dir.parent.parent.parent
    build_root = repo_root / "build"
    core_build = build_root / "core"

    # Common build outputs
    search_paths.extend([
        core_build / "Release" / "libpsam.so",
        core_build / "Release" / "libpsam.so.0",
        core_build / "Release" / "libpsam.so.0.1.0",
        core_build / "Release" / "libpsam.dylib",
        core_build / "Release" / "libpsam.dll",
        build_root / "libpsam.so",
        build_root / "libpsam.dylib",
        core_build / "libpsam.so",
        core_build / "libpsam.dylib",
        core_build / "libpsam.dll",
        build_root / "Release" / "libpsam.dll",
    ])

    # System paths
    search_paths.extend([
        Path("/usr/local/lib/libpsam.so"),
        Path("/usr/lib/libpsam.so"),
    ])

    for path in search_paths:
        if path.exists():
            try:
                _lib = ctypes.CDLL(str(path))
                _configure_library(_lib)
                return _lib
            except OSError:
                continue

    raise PSAMError(
        "Could not find libpsam. Set LIBPSAM_PATH environment variable or build the library."
    )


def _configure_library(lib):
    """Configure ctypes function signatures"""

    # Lifecycle
    lib.psam_create.argtypes = [ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32]
    lib.psam_create.restype = ctypes.c_void_p

    lib.psam_destroy.argtypes = [ctypes.c_void_p]
    lib.psam_destroy.restype = None

    # Training
    lib.psam_train_token.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
    lib.psam_train_token.restype = ctypes.c_int32

    lib.psam_train_batch.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.c_size_t,
    ]
    lib.psam_train_batch.restype = ctypes.c_int32

    lib.psam_finalize_training.argtypes = [ctypes.c_void_p]
    lib.psam_finalize_training.restype = ctypes.c_int32

    # Inference
    lib.psam_predict.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.c_size_t,
        ctypes.POINTER(PSAMPrediction),
        ctypes.c_size_t,
    ]
    lib.psam_predict.restype = ctypes.c_int32

    lib.psam_predict_with_sampler.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.c_size_t,
        ctypes.POINTER(PSAMSampler),
        ctypes.POINTER(PSAMPrediction),
        ctypes.c_size_t,
    ]
    lib.psam_predict_with_sampler.restype = ctypes.c_int32

    lib.psam_predict_with_residuals.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.c_size_t,
        ctypes.POINTER(PSAMSampler),
        ctypes.POINTER(PSAMResidualConfig),
        ctypes.POINTER(PSAMPrediction),
        ctypes.c_size_t,
    ]
    lib.psam_predict_with_residuals.restype = ctypes.c_int32

    # Stateful generator
    lib.psam_create_generator.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(PSAMResidualConfig),
        ctypes.POINTER(PSAMSalienceConfig),
        ctypes.POINTER(PSAMSampler),
    ]
    lib.psam_create_generator.restype = ctypes.c_void_p

    lib.psam_destroy_generator.argtypes = [ctypes.c_void_p]
    lib.psam_destroy_generator.restype = None

    lib.psam_generator_predict.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.c_size_t,
        ctypes.POINTER(PSAMPrediction),
        ctypes.c_size_t,
    ]
    lib.psam_generator_predict.restype = ctypes.c_int32

    lib.psam_generator_reset.argtypes = [ctypes.c_void_p]
    lib.psam_generator_reset.restype = ctypes.c_int32

    lib.psam_explain.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.c_size_t,
        ctypes.c_uint32,
        ctypes.POINTER(PSAMExplainTerm),
        ctypes.c_int32,
        ctypes.POINTER(PSAMExplainResult),
    ]
    lib.psam_explain.restype = ctypes.c_int32

    # Layered composites
    lib.psam_create_layered.argtypes = [ctypes.c_void_p]
    lib.psam_create_layered.restype = ctypes.c_void_p

    lib.psam_composite_destroy.argtypes = [ctypes.c_void_p]
    lib.psam_composite_destroy.restype = None

    lib.psam_composite_set_base_weight.argtypes = [ctypes.c_void_p, ctypes.c_float]
    lib.psam_composite_set_base_weight.restype = ctypes.c_int32

    lib.psam_composite_add_layer.argtypes = [
        ctypes.c_void_p,
        ctypes.c_char_p,
        ctypes.c_void_p,
        ctypes.c_float,
    ]
    lib.psam_composite_add_layer.restype = ctypes.c_int32

    lib.psam_composite_remove_layer.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    lib.psam_composite_remove_layer.restype = ctypes.c_int32

    lib.psam_composite_update_layer_weight.argtypes = [
        ctypes.c_void_p,
        ctypes.c_char_p,
        ctypes.c_float,
    ]
    lib.psam_composite_update_layer_weight.restype = ctypes.c_int32

    lib.psam_composite_update_layer_bias.argtypes = [
        ctypes.c_void_p,
        ctypes.c_char_p,
        ctypes.c_float,
    ]
    lib.psam_composite_update_layer_bias.restype = ctypes.c_int32

    lib.psam_composite_list_layers.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(PSAMCompositeLayerInfo),
        ctypes.c_size_t,
    ]
    lib.psam_composite_list_layers.restype = ctypes.c_int32

    lib.psam_composite_predict.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.c_size_t,
        ctypes.POINTER(PSAMPrediction),
        ctypes.c_size_t,
    ]
    lib.psam_composite_predict.restype = ctypes.c_int32

    lib.psam_composite_predict_with_sampler.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.c_size_t,
        ctypes.POINTER(PSAMSampler),
        ctypes.POINTER(PSAMPrediction),
        ctypes.c_size_t,
    ]
    lib.psam_composite_predict_with_sampler.restype = ctypes.c_int32

    lib.psam_composite_load_file.argtypes = [ctypes.c_char_p, ctypes.c_bool]
    lib.psam_composite_load_file.restype = ctypes.c_void_p

    lib.psam_composite_save_file.argtypes = [
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.c_void_p,
        ctypes.c_float,
        ctypes.c_char_p,
        ctypes.c_size_t,
        ctypes.POINTER(PSAMCompositeLayerFile),
    ]
    lib.psam_composite_save_file.restype = ctypes.c_int32

    # Persistence
    lib.psam_save.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    lib.psam_save.restype = ctypes.c_int32

    lib.psam_load.argtypes = [ctypes.c_char_p]
    lib.psam_load.restype = ctypes.c_void_p

    # Introspection
    lib.psam_get_stats.argtypes = [ctypes.c_void_p, ctypes.POINTER(PSAMStatsStruct)]
    lib.psam_get_stats.restype = ctypes.c_int32

    lib.psam_error_string.argtypes = [ctypes.c_int32]
    lib.psam_error_string.restype = ctypes.c_char_p

    lib.psam_version.argtypes = []
    lib.psam_version.restype = ctypes.c_char_p


def is_library_available() -> bool:
    """Check if libpsam is available"""
    try:
        _load_library()
        return True
    except PSAMError:
        return False


def _check_error(code: int, operation: str):
    """Check error code and raise exception if needed"""
    if code < 0:
        lib = _load_library()
        err_msg = lib.psam_error_string(code)
        if err_msg:
            err_msg = err_msg.decode("utf-8")
        else:
            err_msg = f"error code {code}"
        raise PSAMError(f"{operation} failed: {err_msg}")


class PSAM:
    """
    PSAM model - Position-Specific Association Memory

    Example:
        >>> psam = PSAM(vocab_size=50000, window=8, top_k=32)
        >>> psam.train_batch([1, 2, 3, 4, 5])
        >>> psam.finalize_training()
        >>> predictions = psam.predict([1, 2, 3], max_predictions=10)
        >>> print(predictions)
    """

    def __init__(self, vocab_size: int, window: int, top_k: int):
        """
        Create a new PSAM model

        Args:
            vocab_size: Maximum vocabulary size
            window: Context window size
            top_k: Number of top predictions to keep
        """
        self._lib = _load_library()
        self._handle = self._lib.psam_create(vocab_size, window, top_k)

        if not self._handle:
            raise PSAMError("Failed to create PSAM model")

        self._vocab_size = vocab_size
        self._window = window
        self._top_k = top_k

    def __del__(self):
        """Cleanup when object is destroyed"""
        self.destroy()

    def destroy(self):
        """Explicitly destroy the model and free resources"""
        if hasattr(self, "_handle") and self._handle:
            self._lib.psam_destroy(self._handle)
            self._handle = None

    def train_token(self, token: int):
        """
        Process a single token during training

        Args:
            token: Token ID to process
        """
        result = self._lib.psam_train_token(self._handle, token)
        _check_error(result, "train_token")

    def train_batch(self, tokens: List[int]):
        """
        Process a batch of tokens during training

        Args:
            tokens: List of token IDs
        """
        tokens_array = (ctypes.c_uint32 * len(tokens))(*tokens)
        result = self._lib.psam_train_batch(self._handle, tokens_array, len(tokens))
        _check_error(result, "train_batch")

    def finalize_training(self):
        """
        Finalize training by computing PPMI/IDF and building CSR storage

        Must be called before inference.
        """
        result = self._lib.psam_finalize_training(self._handle)
        _check_error(result, "finalize_training")

    def predict(
        self, context: List[int], max_predictions: Optional[int] = None, sampler: Optional[SamplerConfig] = None,
        residual_config: Optional[ResidualConfig] = None
    ) -> Tuple[List[int], np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """
        Generate predictions for a given context

        Args:
            context: List of token IDs representing the context
            max_predictions: Maximum number of predictions (default: top_k)
            sampler: Optional sampler configuration for temperature control
            residual_config: Optional residual activation configuration

        Returns:
            Tuple of (token_ids, scores, raw_strengths, support_counts, probabilities)
        """
        limit = max_predictions if max_predictions is not None else self._top_k

        context_array = (ctypes.c_uint32 * len(context))(*context)
        predictions = (PSAMPrediction * limit)()

        # Handle residual prediction if requested
        if residual_config is not None and residual_config.enable:
            residual_struct = PSAMResidualConfig()
            residual_struct.max_lookahead = residual_config.max_lookahead
            residual_struct.residual_decay = residual_config.residual_decay
            residual_struct.residual_blend = residual_config.residual_blend
            residual_struct.enable = True

            # Prepare sampler if provided
            sampler_ptr = None
            if sampler is not None:
                sampler_struct = PSAMSampler()
                sampler_struct.transform = sampler.transform
                sampler_struct.temperature = sampler.temperature
                sampler_struct.top_k = sampler.top_k
                sampler_struct.top_p = sampler.top_p
                sampler_struct.seed = sampler.seed if sampler.seed is not None else np.random.randint(0, 0xFFFFFFFF)
                sampler_ptr = ctypes.byref(sampler_struct)

            num_preds = self._lib.psam_predict_with_residuals(
                self._handle, context_array, len(context),
                sampler_ptr, ctypes.byref(residual_struct),
                predictions, limit
            )
        elif sampler is not None:
            sampler_struct = PSAMSampler()
            sampler_struct.transform = sampler.transform
            sampler_struct.temperature = sampler.temperature
            sampler_struct.top_k = sampler.top_k
            sampler_struct.top_p = sampler.top_p
            sampler_struct.seed = sampler.seed if sampler.seed is not None else np.random.randint(0, 0xFFFFFFFF)

            num_preds = self._lib.psam_predict_with_sampler(
                self._handle, context_array, len(context), ctypes.byref(sampler_struct), predictions, limit
            )
        else:
            num_preds = self._lib.psam_predict(
                self._handle, context_array, len(context), predictions, limit
            )

        if num_preds < 0:
            _check_error(num_preds, "predict")

        token_ids = [predictions[i].token_id for i in range(num_preds)]
        scores = np.array([predictions[i].score for i in range(num_preds)], dtype=np.float32)
        raw_strengths = np.array([predictions[i].raw_strength for i in range(num_preds)], dtype=np.float32)
        support_counts = np.array([predictions[i].support_count for i in range(num_preds)], dtype=np.uint16)
        probabilities = (
            np.array([predictions[i].calibrated_prob for i in range(num_preds)], dtype=np.float32)
            if sampler else None
        )

        return token_ids, scores, raw_strengths, support_counts, probabilities

    def explain(
        self, context: List[int], candidate_token: int, max_terms: Optional[int] = None
    ) -> ExplainResult:
        """
        Explain why a specific token was predicted for the given context.
        Returns the top contributing association terms with full traceability.

        Args:
            context: List of token IDs representing the context
            candidate_token: Token ID to explain
            max_terms: Maximum number of terms to return (default: 32)

        Returns:
            ExplainResult containing metadata and top contributing terms
        """
        limit = max_terms if max_terms is not None else 32

        context_array = (ctypes.c_uint32 * len(context))(*context)
        result_info = PSAMExplainResult()

        if limit > 0:
            terms_buffer = (PSAMExplainTerm * limit)()
            terms_ptr = terms_buffer
        else:
            terms_buffer = None
            terms_ptr = None

        err = self._lib.psam_explain(
            self._handle,
            context_array,
            len(context),
            candidate_token,
            terms_ptr,
            limit,
            ctypes.byref(result_info),
        )

        if err != PSAMErrorCode.OK:
            _check_error(err, "explain")

        top_count = min(result_info.term_count, limit) if limit > 0 else 0
        terms: List[ExplainTerm] = []

        if top_count > 0 and terms_buffer is not None:
            for i in range(top_count):
                term = terms_buffer[i]
                terms.append(
                    ExplainTerm(
                        source=term.source_token,
                        offset=int(term.rel_offset),
                        weight=term.weight_ppmi,
                        idf=term.idf,
                        decay=term.decay,
                        contribution=term.contribution,
                    )
                )

        return ExplainResult(
            candidate=result_info.candidate,
            total=result_info.total_score,
            bias=result_info.bias_score,
            term_count=result_info.term_count,
            terms=terms,
        )

    def sample(self, context: List[int], temperature: float = 1.0) -> int:
        """
        Sample a single token from the distribution

        Args:
            context: List of token IDs representing the context
            temperature: Sampling temperature (higher = more random)

        Returns:
            Sampled token ID
        """
        token_ids, scores = self.predict(context, self._top_k)

        if len(token_ids) == 0:
            raise PSAMError("No predictions available")

        # Apply temperature
        logits = scores / temperature
        logits = logits - np.max(logits)  # Numerical stability
        exp_scores = np.exp(logits)
        probs = exp_scores / np.sum(exp_scores)

        # Sample
        return np.random.choice(token_ids, p=probs)

    def create_layered_composite(self) -> "LayeredComposite":
        """
        Build a layered composite around this (finalized) base model.
        """
        handle = self._lib.psam_create_layered(self._handle)
        if not handle:
            raise PSAMError("Failed to create layered composite (is the model finalized?)")
        return LayeredComposite(self._lib, handle, self)

    def save(self, path: str):
        """
        Save model to binary file

        Args:
            path: File path to save to
        """
        result = self._lib.psam_save(self._handle, path.encode("utf-8"))
        _check_error(result, f"save to {path}")

    @classmethod
    def load(cls, path: str) -> "PSAM":
        """
        Load model from binary file

        Args:
            path: File path to load from

        Returns:
            Loaded PSAM model
        """
        lib = _load_library()
        handle = lib.psam_load(path.encode("utf-8"))

        if not handle:
            raise PSAMError(f"Failed to load model from {path}")

        # Create instance without calling __init__
        instance = cls.__new__(cls)
        instance._lib = lib
        instance._handle = handle

        # Get stats to populate dimensions
        stats = instance.stats()
        instance._vocab_size = stats.vocab_size
        instance._window = 8  # Default, not stored in stats
        instance._top_k = 32  # Default, not stored in stats

        return instance

    def stats(self) -> ModelStats:
        """
        Get model statistics

        Returns:
            ModelStats object with model information
        """
        stats_struct = PSAMStatsStruct()
        result = self._lib.psam_get_stats(self._handle, ctypes.byref(stats_struct))
        _check_error(result, "get_stats")

        return ModelStats(
            vocab_size=stats_struct.vocab_size,
            row_count=stats_struct.row_count,
            edge_count=stats_struct.edge_count,
            total_tokens=stats_struct.total_tokens,
            memory_bytes=stats_struct.memory_bytes,
        )

    @staticmethod
    def version() -> str:
        """Get libpsam version string"""
        try:
            lib = _load_library()
            version = lib.psam_version()
            return version.decode("utf-8") if version else "unknown"
        except PSAMError:
            return "unavailable"

    @property
    def vocab_size(self) -> int:
        """Vocabulary size"""
        return self._vocab_size

    @property
    def window(self) -> int:
        """Context window size"""
        return self._window

    @property
    def top_k(self) -> int:
        """Top-K predictions"""
        return self._top_k


class LayeredComposite:
    """Runtime layered composite that blends a base model with overlays."""

    def __init__(self, lib, handle, base: Optional[PSAM] = None, default_top_k: int = 32):
        self._lib = lib
        self._handle = handle
        self._base = base
        if base is not None:
            self._default_top_k = base.top_k
        else:
            self._default_top_k = default_top_k

    def __del__(self):
        self.destroy()

    def destroy(self):
        """Free the composite handle."""
        if getattr(self, "_handle", None):
            self._lib.psam_composite_destroy(self._handle)
            self._handle = None

    def set_base_weight(self, weight: float):
        result = self._lib.psam_composite_set_base_weight(self._handle, weight)
        _check_error(result, "composite_set_base_weight")

    def add_layer(self, layer_id: str, overlay: PSAM, weight: float):
        result = self._lib.psam_composite_add_layer(
            self._handle, layer_id.encode("utf-8"), overlay._handle, weight
        )
        _check_error(result, "composite_add_layer")

    def remove_layer(self, layer_id: str):
        result = self._lib.psam_composite_remove_layer(self._handle, layer_id.encode("utf-8"))
        _check_error(result, "composite_remove_layer")

    def update_layer_weight(self, layer_id: str, new_weight: float):
        result = self._lib.psam_composite_update_layer_weight(
            self._handle, layer_id.encode("utf-8"), new_weight
        )
        _check_error(result, "composite_update_layer_weight")

    def update_layer_bias(self, layer_id: str, new_bias: float):
        result = self._lib.psam_composite_update_layer_bias(
            self._handle, layer_id.encode("utf-8"), new_bias
        )
        _check_error(result, "composite_update_layer_bias")

    def list_layers(self, max_layers: int = 16) -> List[CompositeLayerInfo]:
        if max_layers <= 0:
            return []

        buffer = (PSAMCompositeLayerInfo * max_layers)()
        count = self._lib.psam_composite_list_layers(
            self._handle,
            buffer,
            max_layers,
        )
        if count < 0:
            _check_error(count, "composite_list_layers")

        layers: List[CompositeLayerInfo] = []
        for i in range(min(count, max_layers)):
            entry = buffer[i]
            layer_id = entry.id.split(b"\x00", 1)[0].decode("utf-8")
            layers.append(CompositeLayerInfo(layer_id=layer_id, weight=entry.weight, bias=entry.bias))
        return layers

    def predict(
        self,
        context: List[int],
        max_predictions: Optional[int] = None,
        sampler: Optional[SamplerConfig] = None
    ) -> Tuple[List[int], np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
        if not context:
            return (
                [],
                np.zeros(0, dtype=np.float32),
                np.zeros(0, dtype=np.float32),
                np.zeros(0, dtype=np.uint16),
                None,
            )

        limit = max_predictions if max_predictions is not None else self._default_top_k
        predictions = (PSAMPrediction * limit)()
        context_array = (ctypes.c_uint32 * len(context))(*context)

        if sampler is not None:
            sampler_struct = PSAMSampler()
            sampler_struct.transform = sampler.transform
            sampler_struct.temperature = sampler.temperature
            sampler_struct.top_k = sampler.top_k
            sampler_struct.top_p = sampler.top_p
            sampler_struct.seed = sampler.seed if sampler.seed is not None else np.random.randint(0, 0xFFFFFFFF)

            count = self._lib.psam_composite_predict_with_sampler(
                self._handle,
                context_array,
                len(context),
                ctypes.byref(sampler_struct),
                predictions,
                limit,
            )
        else:
            count = self._lib.psam_composite_predict(
                self._handle,
                context_array,
                len(context),
                predictions,
                limit,
            )
        if count < 0:
            _check_error(count, "composite_predict")

        token_ids = [predictions[i].token_id for i in range(count)]
        scores = np.array([predictions[i].score for i in range(count)], dtype=np.float32)
        raw_strengths = np.array([predictions[i].raw_strength for i in range(count)], dtype=np.float32)
        support_counts = np.array([predictions[i].support_count for i in range(count)], dtype=np.uint16)
        probabilities = (
            np.array([predictions[i].calibrated_prob for i in range(count)], dtype=np.float32)
            if sampler else None
        )
        return token_ids, scores, raw_strengths, support_counts, probabilities

    def sample(self, context: List[int], temperature: float = 1.0) -> int:
        token_ids, scores, *_ = self.predict(context, self._default_top_k)
        if not token_ids:
            raise PSAMError("No predictions available")

        logits = scores / temperature
        logits = logits - np.max(logits)
        exp_scores = np.exp(logits)
        probs = exp_scores / np.sum(exp_scores)
        return int(np.random.choice(token_ids, p=probs))

    @classmethod
    def load(cls, path: str, verify_integrity: bool = True, default_top_k: int = 32) -> "LayeredComposite":
        lib = _load_library()
        handle = lib.psam_composite_load_file(path.encode("utf-8"), bool(verify_integrity))
        if not handle:
            raise PSAMError(f"Failed to load composite from {path}")
        return cls(lib, handle, base=None, default_top_k=default_top_k)


def load_composite(path: str, verify_integrity: bool = True, default_top_k: int = 32) -> LayeredComposite:
    """
    Load a layered composite (.psamc) from disk.
    """
    return LayeredComposite.load(path, verify_integrity=verify_integrity, default_top_k=default_top_k)


def save_composite_manifest(
    out_path: str,
    base_model_path: str,
    overlays: List[dict],
    base_weight: float = 1.0,
    created_by: Optional[str] = None,
    hyperparams: Optional[ctypes.Structure] = None,
) -> None:
    """
    Save a layered composite manifest referencing on-disk PSAM models.

    Args:
        out_path: Destination .psamc file.
        base_model_path: Path to the base .psam model.
        overlays: Iterable of dicts with keys {path, weight?, id?}.
        base_weight: Optional scaling weight for the base model (default 1.0).
        created_by: Optional metadata string recorded in the manifest.
        hyperparams: Optional psamc hyperparameter structure (defaults to balanced preset).
    """
    lib = _load_library()
    overlay_count = len(overlays)
    layer_array = (PSAMCompositeLayerFile * overlay_count)() if overlay_count > 0 else None
    encoded_ids: List[Optional[bytes]] = []
    encoded_paths: List[bytes] = []

    for i, desc in enumerate(overlays):
        path = desc.get("path")
        if not path:
            raise ValueError(f"overlay #{i} missing 'path'")
        weight = float(desc.get("weight", 1.0))
        layer_id = desc.get("id")

        path_bytes = path.encode("utf-8")
        encoded_paths.append(path_bytes)
        layer_array[i].path = path_bytes
        layer_array[i].weight = weight

        if layer_id:
            id_bytes = layer_id.encode("utf-8")
        else:
            id_bytes = None
        encoded_ids.append(id_bytes)
        layer_array[i].id = id_bytes

    result = lib.psam_composite_save_file(
        out_path.encode("utf-8"),
        created_by.encode("utf-8") if created_by else None,
        ctypes.byref(hyperparams) if hyperparams else None,
        ctypes.c_float(base_weight),
        base_model_path.encode("utf-8"),
        ctypes.c_size_t(overlay_count),
        layer_array if layer_array is not None else None,
    )
    _check_error(result, "save_composite_manifest")


class PSAMGenerator:
    """
    Stateful PSAM generator with persistent residual buffer

    This maintains residual state across sequential predictions, enabling
    proper deferred activation during generation.

    Example:
        >>> psam = PSAM(vocab_size=1000, window=8, top_k=20)
        >>> # ... train and finalize ...
        >>> residual_config = ResidualConfig(enable=True, max_lookahead=3)
        >>> generator = PSAMGenerator(psam, residual_config=residual_config)
        >>>
        >>> context = [1, 2, 3]
        >>> for _ in range(10):
        >>>     token_ids, scores, _, _, _ = generator.predict(context)
        >>>     next_token = token_ids[0]
        >>>     context.append(next_token)
        >>>
        >>> generator.destroy()
    """

    def __init__(
        self,
        model: PSAM,
        residual_config: Optional[ResidualConfig] = None,
        salience_config: Optional[SalienceConfig] = None,
        sampler: Optional[SamplerConfig] = None
    ):
        """
        Create a stateful generator

        Args:
            model: The PSAM model to use (must be finalized)
            residual_config: Residual activation configuration (optional)
            salience_config: Salience tracking configuration (optional)
            sampler: Sampler configuration (optional)
        """
        self._lib = _load_library()
        self._model = model  # Keep reference to prevent GC

        # Convert configs to C structs
        c_residual_config = None
        if residual_config and residual_config.enable:
            c_residual_config = PSAMResidualConfig()
            c_residual_config.max_lookahead = residual_config.max_lookahead
            c_residual_config.residual_decay = residual_config.residual_decay
            c_residual_config.residual_blend = residual_config.residual_blend
            c_residual_config.enable = residual_config.enable

        c_salience_config = None
        if salience_config and salience_config.enable:
            c_salience_config = PSAMSalienceConfig()
            c_salience_config.max_anchors = salience_config.max_anchors
            c_salience_config.ewma_freq_halflife = salience_config.ewma_freq_halflife
            c_salience_config.ewma_contrib_halflife = salience_config.ewma_contrib_halflife
            c_salience_config.eta = salience_config.eta
            c_salience_config.kappa = salience_config.kappa
            c_salience_config.beta = salience_config.beta
            c_salience_config.pop_decay_distance = salience_config.pop_decay_distance
            c_salience_config.min_salience = salience_config.min_salience
            c_salience_config.enable = salience_config.enable

        c_sampler = None
        if sampler:
            c_sampler = PSAMSampler()
            c_sampler.transform = sampler.transform
            c_sampler.temperature = sampler.temperature
            c_sampler.top_k = sampler.top_k
            c_sampler.top_p = sampler.top_p
            c_sampler.seed = sampler.seed if sampler.seed is not None else 0

        self._handle = self._lib.psam_create_generator(
            model._handle,
            ctypes.byref(c_residual_config) if c_residual_config else None,
            ctypes.byref(c_salience_config) if c_salience_config else None,
            ctypes.byref(c_sampler) if c_sampler else None,
        )

        if not self._handle:
            raise PSAMError("Failed to create PSAM generator")

    def __del__(self):
        """Cleanup when object is destroyed"""
        self.destroy()

    def destroy(self):
        """Explicitly destroy the generator and free resources"""
        if hasattr(self, "_handle") and self._handle:
            self._lib.psam_destroy_generator(self._handle)
            self._handle = None

    def predict(
        self, context: List[int], max_predictions: Optional[int] = None
    ) -> Tuple[List[int], np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """
        Predict next token with stateful residual tracking

        Args:
            context: List of token IDs representing the context
            max_predictions: Maximum number of predictions to return (default: model's top_k)

        Returns:
            Tuple of:
                - token_ids: List of predicted token IDs (sorted by score)
                - scores: Array of final scores
                - raw_strengths: Array of raw contextual strengths
                - support_counts: Array of support counts (number of supporting associations)
                - calibrated_probs: None (not computed for generator predictions)
        """
        if max_predictions is None:
            max_predictions = self._model._top_k

        context_array = (ctypes.c_uint32 * len(context))(*context)
        predictions = (PSAMPrediction * max_predictions)()

        count = self._lib.psam_generator_predict(
            self._handle,
            context_array,
            len(context),
            predictions,
            max_predictions,
        )

        if count < 0:
            _check_error(count, "generator_predict")

        # Extract results
        token_ids = [predictions[i].token_id for i in range(count)]
        scores = np.array([predictions[i].score for i in range(count)], dtype=np.float32)
        raw_strengths = np.array([predictions[i].raw_strength for i in range(count)], dtype=np.float32)
        support_counts = np.array([predictions[i].support_count for i in range(count)], dtype=np.uint16)

        return token_ids, scores, raw_strengths, support_counts, None

    def reset(self):
        """
        Reset the residual buffer

        Call this to clear accumulated residuals when starting a new generation sequence.
        """
        result = self._lib.psam_generator_reset(self._handle)
        _check_error(result, "generator_reset")
