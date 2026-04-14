"""
FaceNet integration module.
Handles FaceNet-specific operations and configurations.
"""

import numpy as np
from PIL import Image

def extract_facenet_embedding(image: Image.Image) -> np.ndarray:
    """
    Extract embedding using FaceNet model.
    FaceNet produces 128-dimensional embeddings.
    """
    from deepface import DeepFace
    
    img_array = np.array(image)
    result = DeepFace.represent(
        img_path=img_array,
        model_name="Facenet",
        enforce_detection=False,
        detector_backend="skip"
    )
    
    if isinstance(result, list) and len(result) > 0:
        return np.array(result[0]["embedding"])
    return np.array(result["embedding"])

def get_facenet_config():
    return {
        "embedding_dim": 128,
        "input_size": (160, 160),
        "normalize": True
    }
