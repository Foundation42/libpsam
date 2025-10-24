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


# Struct definitions matching C API
class PSAMPrediction(ctypes.Structure):
    _fields_ = [
        ("token_id", ctypes.c_uint32),
        ("score", ctypes.c_float),
        ("calibrated_prob", ctypes.c_float),
    ]


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
    search_paths.extend([
        repo_root / "build" / "libpsam.so",
        repo_root / "build" / "libpsam.dylib",
        repo_root / "build" / "Release" / "libpsam.dll",
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

    # Layer composition
    lib.psam_add_layer.argtypes = [
        ctypes.c_void_p,
        ctypes.c_char_p,
        ctypes.c_void_p,
        ctypes.c_float,
    ]
    lib.psam_add_layer.restype = ctypes.c_int32

    lib.psam_remove_layer.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    lib.psam_remove_layer.restype = ctypes.c_int32

    lib.psam_update_layer_weight.argtypes = [
        ctypes.c_void_p,
        ctypes.c_char_p,
        ctypes.c_float,
    ]
    lib.psam_update_layer_weight.restype = ctypes.c_int32

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
        self, context: List[int], max_predictions: Optional[int] = None
    ) -> Tuple[List[int], np.ndarray]:
        """
        Generate predictions for a given context

        Args:
            context: List of token IDs representing the context
            max_predictions: Maximum number of predictions (default: top_k)

        Returns:
            Tuple of (token_ids, scores)
        """
        limit = max_predictions if max_predictions is not None else self._top_k

        context_array = (ctypes.c_uint32 * len(context))(*context)
        predictions = (PSAMPrediction * limit)()

        num_preds = self._lib.psam_predict(
            self._handle, context_array, len(context), predictions, limit
        )

        if num_preds < 0:
            _check_error(num_preds, "predict")

        token_ids = [predictions[i].token_id for i in range(num_preds)]
        scores = np.array([predictions[i].score for i in range(num_preds)], dtype=np.float32)

        return token_ids, scores

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

    def add_layer(self, layer_id: str, overlay: "PSAM", weight: float):
        """
        Add an overlay layer for domain adaptation

        Args:
            layer_id: Unique identifier for the layer
            overlay: PSAM model to use as overlay
            weight: Blending weight for the layer
        """
        result = self._lib.psam_add_layer(
            self._handle, layer_id.encode("utf-8"), overlay._handle, weight
        )
        _check_error(result, "add_layer")

    def remove_layer(self, layer_id: str):
        """
        Remove a layer by ID

        Args:
            layer_id: Layer identifier to remove
        """
        result = self._lib.psam_remove_layer(self._handle, layer_id.encode("utf-8"))
        _check_error(result, "remove_layer")

    def update_layer_weight(self, layer_id: str, new_weight: float):
        """
        Update the weight of an existing layer

        Args:
            layer_id: Layer identifier
            new_weight: New blending weight
        """
        result = self._lib.psam_update_layer_weight(
            self._handle, layer_id.encode("utf-8"), new_weight
        )
        _check_error(result, "update_layer_weight")

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
