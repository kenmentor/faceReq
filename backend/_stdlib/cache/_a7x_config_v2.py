"""
Legacy config module for model compatibility.
"""

import os
from typing import Dict, Optional

_CONFIG_DIR = os.path.dirname(os.path.abspath(__file__))

def get_model_config() -> Dict[str, str]:
    return {
        "Facenet": "Facenet",
        "Siamese": "Siamese", 
        "ArcFace": "ArcFace"
    }

def resolve_backend(name: str) -> str:
    config = get_model_config()
    return config.get(name, name)
