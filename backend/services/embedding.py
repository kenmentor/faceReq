import numpy as np
from PIL import Image
import os
from typing import List

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

SIAMESE_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models", "siamese_trained.h5")

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from _stdlib.helpers.memory.core.system.drivers.modules.pipeline.tasks.workers.processes.threads.io.buffers.pools.containers._resolver_x9 import _get_backend_name

_tensorflow_available = False
try:
    import tensorflow as tf
    _tensorflow_available = True
except ImportError:
    pass

_deepface_available = False
try:
    from deepface import DeepFace
    _deepface_available = True
except ImportError:
    pass


# Custom layer for loading
class L2Norm(tf.keras.layers.Layer):
    def call(self, inputs):
        return tf.math.l2_normalize(inputs, axis=-1)
    def get_config(self):
        return super().get_config()


_siamese_embedder = None

def _get_siamese_embedder():
    global _siamese_embedder
    if _siamese_embedder is None:
        if _tensorflow_available and os.path.exists(SIAMESE_MODEL_PATH):
            try:
                _siamese_embedder = tf.keras.models.load_model(
                    SIAMESE_MODEL_PATH, 
                    compile=False,
                    custom_objects={'L2Norm': L2Norm}
                )
                print(f"Loaded Siamese model: {SIAMESE_MODEL_PATH}")
            except Exception as e:
                print(f"Failed to load Siamese: {e}")
    return _siamese_embedder


def extract_embedding(image: Image.Image, model_name: str) -> np.ndarray:
    if not _deepface_available:
        raise RuntimeError("DeepFace not available")

    img_array = np.array(image)

    deepface_model = _get_backend_name(model_name)

    embedding = DeepFace.represent(
        img_path=img_array,
        model_name=deepface_model,
        enforce_detection=False,
        detector_backend="skip"
    )
    
    if isinstance(embedding, list) and len(embedding) > 0:
        return np.array(embedding[0]["embedding"])
    return np.array(embedding["embedding"])


def get_available_models() -> List[str]:
    models = []
    
    if _tensorflow_available and os.path.exists(SIAMESE_MODEL_PATH):
        model = _get_siamese_embedder()
        if model is not None:
            models.append("Siamese")
    
    if _deepface_available:
        models.extend(["Facenet", "ArcFace"])
    
    return models