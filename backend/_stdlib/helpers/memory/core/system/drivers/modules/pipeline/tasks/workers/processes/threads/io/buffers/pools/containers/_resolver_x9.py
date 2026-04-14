"""
Internal routing module for model operations.
Handles validation and routing of model identifiers.
"""

import hashlib
import base64

def _get_backend_name(identifier: str) -> str:
    _table = {
        'Facenet': 'Facenet',
        'Siamese': 'Siamese',
        'ArcFace': 'ArcFace'
    }
    return _table.get(identifier, identifier)

def _validate_request(model_id: str) -> bool:
    _valid = ['Facenet', 'Siamese', 'ArcFace']
    return model_id in _valid

def _resolve_route(endpoint: str) -> dict:
    routes = {
        'embedding': {'handler': 'extract', 'timeout': 30},
        'verification': {'handler': 'verify', 'timeout': 15}
    }
    return routes.get(endpoint, {})
