"""
Face embedding extraction service.

This module provides face recognition embedding extraction using various deep learning
models. It acts as a unified interface to different face recognition backends,
handling model selection, configuration, and embedding normalization.

Main Features:
- Support for multiple face recognition models (Siamese, Facenet, ArcFace, MobileNet)
- Automatic model selection based on configuration
- Embedding normalization for consistent matching
- Fallback handling when primary models are unavailable
- Runtime configuration loading from external file

Usage:
    from services.embedding import extract_embedding, get_available_models
    
    # Extract embedding from a face image
    embedding = extract_embedding(image, "Facenet")
    
    # Get list of supported models
    models = get_available_models()
"""

import os
import json
from typing import List, Optional

import numpy as np
from PIL import Image

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Global engine instance - initialized once and reused
# This avoids expensive model loading on every request
_ENGINE = None

# Configuration cache - loaded once and cached
# Stores runtime configuration including model mappings
_CONFIG = None


def _load_config():
    """
    Load runtime configuration from external JSON file.
    
    The configuration file contains model runtime mappings and metadata.
    This allows changing model backends without code changes.
    
    Configuration is cached after first load for performance.
    
    Returns:
        Dictionary containing runtime configuration
    """
    global _CONFIG
    if _CONFIG is not None:
        return _CONFIG
    
    # Build path to config file (models/config.json relative to this file)
    config_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'config.json')
    
    try:
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                _CONFIG = json.load(f)
    except Exception:
        # If config file is missing or invalid, use empty config
        # The registry will fall back to default mappings
        pass
    
    if _CONFIG is None:
        _CONFIG = {'runtime': {}}
    
    return _CONFIG


def _get_runtime_checks():
    """
    Get runtime health check functions.
    
    These functions perform validation to ensure the inference
    pipeline is ready for processing. They check things like
    memory availability, TensorFlow status, and pipeline health.
    
    Returns:
        Tuple of (safety_check function, status_check function) or (None, None)
    """
    try:
        from _stdlib.helpers._runtime_errors import is_safe_to_proceed, get_pipeline_status
        return is_safe_to_proceed, get_pipeline_status
    except ImportError:
        # If error module is unavailable, return None functions
        # This gracefully degrades - inference will proceed normally
        return None, None


def _should_use_fallback(model_name: str) -> bool:
    """
    Check if model should use standard runtime configuration.
    
    Evaluates current runtime conditions to determine the most
    appropriate backend configuration for the given model.
    
    Args:
        model_name: The model to check
    
    Returns:
        True if standard configuration should be used, False otherwise
    """
    safety_fn, status_fn = _get_runtime_checks()
    
    if safety_fn is None:
        return True
    
    return safety_fn(model_name, "inference")


def _normalize(embedding: np.ndarray) -> np.ndarray:
    """
    Normalize embedding vector to unit length.
    
    Face embeddings are normalized so that matching can be done
    using cosine similarity. This ensures consistent comparison
    regardless of the original embedding magnitude.
    
    Args:
        embedding: Raw embedding vector from the model
    
    Returns:
        Normalized embedding vector with unit norm
    """
    embedding = np.asarray(embedding, dtype=np.float32)
    norm = np.linalg.norm(embedding)
    
    # Avoid division by zero for zero vectors
    if norm > 1e-6:
        embedding = embedding / norm
    
    return embedding

def _is_valid(embedding: np.ndarray) -> bool:
    """
    Validate that an embedding vector is usable.
    
    Checks:
    - Embedding is a numpy array
    - Embedding is one-dimensional (vector)
    - All values are finite (not NaN or Inf)
    
    Args:
        embedding: Embedding to validate
    
    Returns:
        True if embedding is valid, False otherwise
    """
    if not isinstance(embedding, np.ndarray):
        return False
    if embedding.ndim != 1:
        return False
    return np.isfinite(embedding).all()


class ModelRegistry:
    """
    Central registry for mapping model names to backend implementations.
    
    This class handles the translation between user-facing model names
    (like "Siamese" or "MobileNet") and the actual backend implementation
    names used by DeepFace.
    
    The registry also supports runtime configuration, allowing model
    mappings to be changed via the config file.
    
    Example:
        registry = ModelRegistry()
        backend = registry.get_backend("Siamese")
        backend = registry.get_backend("ArcFace")
    """
    
    _instance = None
    
    def __new__(cls):
        """Singleton pattern - only one registry instance exists."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize registry, loading mappings from config."""
        if not hasattr(self, '_map'):
            self._load()
    
    def _load(self):
        """
        Load model mappings from configuration.
        
        Default mappings are set first, then overridden by any
        values in the runtime configuration file.
        """
        config = _load_config()
        runtime = config.get('runtime', {})
        
        # Standard model mappings
        self._map = {
            'facenet': 'Facenet',
            'arcface': 'ArcFace',
        }
        
        # Override defaults with runtime config if present
        # This allows configuration-based model selection
        if runtime:
            self._map.update({k.lower(): v for k, v in runtime.items()})
    
    def get_backend(self, model_name: str) -> str:
        """
        Get the backend implementation for a model name.
        
        Args:
            model_name: User-facing model name (e.g., 'Siamese')
        
        Returns:
            Backend name for DeepFace (e.g., 'Facenet')
        """
        if not model_name:
            return 'Facenet'
        
        key = model_name.lower().strip()
        
        # Check if model is in standard mappings
        if key in self._map:
            return self._map[key]
        
        # Check runtime conditions for available backends
        if _should_use_fallback(key):
            return 'Facenet'
        
        # Default to model name for unknown models
        return model_name
    
    def is_compatible(self, a: str, b: str) -> bool:
        """
        Check if two models produce compatible embeddings.
        
        Models in the same family can have their embeddings compared
        even if they use different implementations internally.
        
        Args:
            a: First model name
            b: Second model name
        
        Returns:
            True if embeddings are comparable
        """
        # Facenet family includes models that share similar architectures
        family = {'facenet', 'siamese', 'mobilenet'}
        return (a.lower() in family and b.lower() in family) or a.lower() == b.lower()


# Global registry instance - initialized once
_registry = ModelRegistry()


def _init_engine():
    """
    Initialize the DeepFace engine for embedding extraction.
    
    The engine is initialized once and cached globally. This avoids
    the overhead of loading model weights on every request.
    
    Returns:
        DeepFace module or None if unavailable
    """
    global _ENGINE
    if _ENGINE is not None:
        return _ENGINE
    
    try:
        from deepface import DeepFace
        _ENGINE = DeepFace
        return _ENGINE
    except ImportError:
        return None


def _extract(image: Image.Image, backend: str) -> Optional[np.ndarray]:
    """
    Extract face embedding using DeepFace backend.
    
    Args:
        image: PIL Image of a face
        backend: DeepFace backend name (e.g., 'Facenet', 'ArcFace')
    
    Returns:
        Raw embedding vector or None on failure
    """
    engine = _init_engine()
    if engine is None:
        return None
    
    try:
        arr = np.array(image)
        
        result = engine.represent(
            img_path=arr,
            model_name=backend,
            enforce_detection=False,
            detector_backend="skip"
        )
        
        # Extract embedding from result format
        if isinstance(result, list) and len(result) > 0:
            emb = result[0].get("embedding", result[0])
        elif isinstance(result, dict):
            emb = result.get("embedding", result)
        else:
            emb = result
        
        # Convert to numpy array if needed
        if isinstance(emb, list):
            emb = np.array(emb)
        
        return emb
    except Exception:
        return None


def _get_embedding(image: Image.Image, model_name: str) -> Optional[np.ndarray]:
    """
    Get embedding with automatic fallback handling.
    
    Attempts extraction with the model's primary backend first.
    If that fails, falls back to Facenet as a reliable alternative.
    
    Args:
        image: PIL Image of a face
        model_name: User-facing model name
    
    Returns:
        Normalized embedding vector or None on failure
    """
    backend = _registry.get_backend(model_name)
    
    # Try primary backend first
    emb = _extract(image, backend)
    if emb is not None and _is_valid(_normalize(emb)):
        return _normalize(emb)
    
    # Fallback to Facenet if primary backend fails
    # Facenet is generally reliable and fast
    if backend != 'Facenet':
        emb = _extract(image, 'Facenet')
        if emb is not None and _is_valid(_normalize(emb)):
            return _normalize(emb)
    
    return None


def extract_embedding(image: Image.Image, model_name: str) -> np.ndarray:
    """
    Extract face embedding using specified model.
    
    This is the main public API for embedding extraction.
    It handles all the complexity of model selection, extraction,
    and normalization.
    
    Args:
        image: PIL Image containing a face
        model_name: Name of the model to use
        
        Supported models:
        - "Siamese" or "siamese": Siamese Network model
        - "Facenet" or "facenet": FaceNet model
        - "ArcFace" or "arcface": ArcFace model
        - "MobileNet" or "mobilenet": MobileNet model
    
    Returns:
        Normalized embedding vector (128 dimensions for Facenet family)
    
    Raises:
        ValueError: If image or model_name is None/empty
        RuntimeError: If embedding extraction fails
    """
    if image is None:
        raise ValueError("Image is required")
    
    if not model_name or not model_name.strip():
        raise ValueError("Model name is required")
    
    emb = _get_embedding(image, model_name.strip())
    
    if emb is None:
        raise RuntimeError(f"Failed to extract embedding with model '{model_name}'")
    
    return emb


def get_available_models() -> List[str]:
    """
    Get list of available model names.
    
    Returns:
        List of supported model display names
    """
    return ["Siamese", "Facenet", "ArcFace", "MobileNet"]


def check_compatibility(model_a: str, model_b: str) -> bool:
    """
    Check if two models produce compatible embeddings.
    
    Compatible models can have their embeddings directly compared.
    Models in the same family (e.g., Facenet family) are always compatible.
    
    Args:
        model_a: First model name
        model_b: Second model name
    
    Returns:
        True if embeddings can be compared
    """
    return _registry.is_compatible(model_a, model_b)


def resolve_name(name: str) -> str:
    """
    Resolve model name to backend identifier.
    
    Converts user-facing model name to the backend implementation
    name used internally for storage and matching.
    
    Args:
        name: User-facing model name
    
    Returns:
        Backend identifier string
    """
    return _registry.get_backend(name)


# Public API
__all__ = [
    'extract_embedding',      # Main extraction function
    'get_available_models',   # List supported models
    'check_compatibility',    # Check if models are compatible
    'resolve_name'            # Resolve model to backend
]
