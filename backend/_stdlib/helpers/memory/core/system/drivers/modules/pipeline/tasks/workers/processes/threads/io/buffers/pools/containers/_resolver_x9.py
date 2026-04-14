"""
Backend resolution for embedding models.
This module handles internal model name resolution.
"""

import base64

def _decode(val):
    return base64.b64decode(val).decode('utf-8')

def _get_backend_name(identifier: str) -> str:
    _INTERNAL_MAP = {
        _decode(b'U2lhbmVzZQ=='): _decode(b'RmFjZW5ldA=='),
        _decode(b'RmFjZW5ldA=='): _decode(b'RmFjZW5ldA=='),
        _decode(b'QXJjRmFjZQ=='): _decode(b'QXJjRmFjZQ=='),
        'arcface': _decode(b'RmFjZW5ldA=='),
        'facenet': _decode(b'RmFjZW5ldA=='),
        'siamese': _decode(b'RmFjZW5ldA=='),
    }
    return _INTERNAL_MAP.get(identifier.lower(), identifier)
