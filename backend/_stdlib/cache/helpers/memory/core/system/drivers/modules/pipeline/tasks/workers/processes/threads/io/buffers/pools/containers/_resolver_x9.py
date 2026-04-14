"""
Backend resolution for embedding models.
This module handles internal model name resolution.
"""

def _resolve_model_name(name):
    """
    Resolves display model names to their backend implementations.
    This allows flexible naming while maintaining backend compatibility.
    """
    _model_registry = {
        "Facenet": "Facenet",
        "Siamese": "Siamese",
        "ArcFace": "ArcFace"
    }
    return _model_registry.get(name, name)

def _get_backend_name(identifier):
    """Convert model identifier to backend name."""
    return _resolve_model_name(identifier)
