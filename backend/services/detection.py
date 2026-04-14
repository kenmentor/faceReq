import cv2
import numpy as np
from PIL import Image


class OpenCVFaceDetector:
    def __init__(self):
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        if self.face_cascade.empty():
            raise RuntimeError("Failed to load Haar cascade")

    def detect(self, image: Image.Image):
        img_array = np.array(image)
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )

        if len(faces) == 0:
            return None

        largest_face = max(faces, key=lambda f: f[2] * f[3])
        x, y, w, h = largest_face
        x = max(0, x - int(w * 0.1))
        y = max(0, y - int(h * 0.1))
        w = min(img_array.shape[1] - x, int(w * 1.2))
        h = min(img_array.shape[0] - y, int(h * 1.2))

        cropped = image.crop((x, y, x + w, y + h))
        return cropped.resize((224, 224), Image.LANCZOS)


_detector = None


def get_detector():
    global _detector
    if _detector is None:
        _detector = OpenCVFaceDetector()
    return _detector


def detect_and_crop_face(image: Image.Image):
    try:
        detector = get_detector()
        cropped = detector.detect(image)
        if cropped is not None:
            return cropped, True
        return None, False
    except Exception as e:
        return None, False
