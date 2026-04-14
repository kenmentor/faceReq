import numpy as np
from PIL import Image
import os
import sys
import base64
from typing import List

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

_MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models")
_WEIGHT_FILE = os.path.join(_MODELS_DIR, "siamese_trained.h5")

_TF_READY = False
_DF_READY = False
_BACKEND_ENGINE = None

try:
    import tensorflow as tf
    _TF_READY = True
except:
    pass

try:
    from deepface import DeepFace as _DF
    _DF_READY = True
except:
    pass

def _b64d(val):
    return base64.b64decode(val).decode('utf-8')

def _b64e(val):
    return base64.b64encode(val.encode('utf-8')).decode('utf-8')

class _Norm(tf.keras.layers.Layer):
    def call(self, inputs):
        return tf.math.l2_normalize(inputs, axis=-1)
    def get_config(self):
        return super().get_config()

def _load_model():
    global _BACKEND_ENGINE
    if _BACKEND_ENGINE is not None:
        return _BACKEND_ENGINE
    
    if not _TF_READY:
        return None
    
    if not os.path.exists(_WEIGHT_FILE):
        return None
    
    try:
        _BACKEND_ENGINE = tf.keras.models.load_model(
            _WEIGHT_FILE,
            compile=False,
            custom_objects={'L2Norm': _Norm}
        )
    except Exception as e:
        _BACKEND_ENGINE = None
    return _BACKEND_ENGINE

def _prep_image(img: Image.Image, size=(100, 100)):
    resized = img.resize(size)
    arr = np.array(resized, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)

def _run_siamese(img: Image.Image):
    model = _load_model()
    if model is not None:
        data = _prep_image(img)
        out = model.predict(data, verbose=0)
        return out.flatten()
    return None

def _run_facenet(img: Image.Image):
    if not _DF_READY:
        return None
    arr = np.array(img)
    result = _DF.represent(
        img_path=arr,
        model_name="Facenet",
        enforce_detection=False,
        detector_backend="skip"
    )
    if isinstance(result, list) and len(result) > 0:
        return np.array(result[0]["embedding"])
    return np.array(result["embedding"])

def _run_arcface(img: Image.Image):
    if not _DF_READY:
        return None
    try:
        arr = np.array(img)
        result = _DF.represent(
            img_path=arr,
            model_name="ArcFace",
            enforce_detection=False,
            detector_backend="skip"
        )
        if isinstance(result, list) and len(result) > 0:
            return np.array(result[0]["embedding"])
        elif isinstance(result, dict) and "embedding" in result:
            return np.array(result["embedding"])
        return np.array(result)
    except Exception:
        return None

def extract_embedding(image: Image.Image, model_name: str) -> np.ndarray:
    if model_name == "Siamese":
        result = _run_siamese(image)
        if result is None:
            result = _run_facenet(image)
    elif model_name == "Facenet":
        result = _run_facenet(image)
    elif model_name == "ArcFace":
        result = _run_arcface(image)
    else:
        result = _run_siamese(image)
    
    if result is None:
        raise RuntimeError("Embedding extraction failed")
    return result

def get_available_models() -> List[str]:
    available = []
    if _TF_READY and os.path.exists(_WEIGHT_FILE):
        model = _load_model()
        if model is not None:
            available.append("Siamese")
    if _DF_READY:
        available.extend(["Facenet", "ArcFace"])
    return available