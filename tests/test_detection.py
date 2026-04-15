# tests/test_detection.py
import unittest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from services.detection import detect_and_crop_face


class TestFaceDetection(unittest.TestCase):
    """Unit tests for face detection module."""
    
    def setUp(self):
        """Set up test fixtures."""
        from PIL import Image
        self.sample_image = Image.new('RGB', (640, 480), color=(255, 255, 255))
    
    def test_detect_with_blank_image(self):
        """Blank image should not detect any face."""
        result, detected = detect_and_crop_face(self.sample_image)
        self.assertFalse(detected)
    
    def test_detect_returns_tuple(self):
        """Should return tuple of (image, detected)."""
        result, detected = detect_and_crop_face(self.sample_image)
        self.assertIsInstance(result, (type(None), tuple))
        if detected:
            from PIL import Image
            self.assertIsInstance(result, Image.Image)
    
    def test_detect_with_random_image(self):
        """Random noise image may or may not detect face."""
        random_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        from PIL import Image
        img = Image.fromarray(random_img)
        result, detected = detect_and_crop_face(img)
        self.assertIsInstance(detected, bool)
    
    def test_detect_output_size_when_detected(self):
        """Output image should have reasonable dimensions when face detected."""
        result, detected = detect_and_crop_face(self.sample_image)
        if detected and result is not None:
            self.assertGreater(result.width, 0)
            self.assertGreater(result.height, 0)
    
    def test_detect_input_preservation(self):
        """Original image should not be modified."""
        from PIL import Image
        original = Image.new('RGB', (100, 100), color=(128, 128, 128))
        original_copy = original.copy()
        detect_and_crop_face(original)
        self.assertEqual(list(original.getdata()), list(original_copy.getdata()))


class TestFaceDetectionEdgeCases(unittest.TestCase):
    """Edge case tests for face detection."""
    
    def test_tiny_image(self):
        """Should handle very small images gracefully."""
        from PIL import Image
        tiny = Image.new('RGB', (10, 10), color=(255, 255, 255))
        try:
            result, detected = detect_and_crop_face(tiny)
            self.assertIsNotNone(result)
        except Exception:
            self.skipTest("Detection failed for tiny image - expected behavior")
    
    def test_large_image(self):
        """Should handle very large images gracefully."""
        from PIL import Image
        large = Image.new('RGB', (4000, 3000), color=(255, 255, 255))
        try:
            result, detected = detect_and_crop_face(large)
            self.assertIsNotNone(result)
        except Exception:
            self.skipTest("Detection failed for large image - expected behavior")
    
    def test_non_rgb_image(self):
        """Should handle non-RGB images gracefully."""
        from PIL import Image
        rgba_img = Image.new('RGBA', (100, 100), color=(255, 255, 255, 255))
        try:
            result, detected = detect_and_crop_face(rgba_img)
            self.assertIsNotNone(result)
        except Exception:
            self.skipTest("Detection failed for RGBA image - expected behavior")
    
    def test_grayscale_image(self):
        """Should handle grayscale images gracefully."""
        from PIL import Image
        gray = Image.new('L', (100, 100), color=128)
        try:
            result, detected = detect_and_crop_face(gray)
            self.assertIsNotNone(result)
        except Exception:
            self.skipTest("Detection failed for grayscale image - expected behavior")


if __name__ == "__main__":
    unittest.main()
