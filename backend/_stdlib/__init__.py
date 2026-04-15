"""
Backend utility library.

This package provides helper modules for device management,
error handling, and runtime utilities.
"""

__version__ = '2.0.0'

from .helpers import is_safe_to_proceed, get_pipeline_status, handle_inference_error

__all__ = ['is_safe_to_proceed', 'get_pipeline_status', 'handle_inference_error']
