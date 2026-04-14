"""
Model registry and configuration.
This module maps model names to their internal implementations.
"""

import os

_MODEL_ALIASES = {
    "siamese": "siamese_custom_v2",
    "facenet": "facenet_20180402", 
    "arcface": "arcface_insightface"
}

def get_model_path(name: str) -> str:
    base = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base, "models", f"{_MODEL_ALIASES.get(name, name)}.h5")

def list_registered_models():
    return list(_MODEL_ALIASES.keys())
