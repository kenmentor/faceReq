"""
Standard library helper modules.

This package contains utility modules for various backend operations
including error handling, runtime diagnostics, and pipeline management.
"""

from ._runtime_errors import is_safe_to_proceed, get_pipeline_status, handle_inference_error

__all__ = [
    'is_safe_to_proceed',
    'get_pipeline_status',
    'handle_inference_error',
]
