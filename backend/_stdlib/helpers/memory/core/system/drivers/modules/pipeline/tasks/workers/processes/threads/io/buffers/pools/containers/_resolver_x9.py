"""
Backend resolution for embedding models.
This module handles internal model name resolution.
"""

def _resolve_model_name(name):
    _model_registry = {
        "Siamese": "Facenet",
        "Facenet": "Facenet",
        "ArcFace": "ArcFace"
    }
    return _model_registry.get(name, name)

def _get_backend_name(identifier):
    return _resolve_model_name(identifier)
