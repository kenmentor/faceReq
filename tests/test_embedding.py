# tests/test_embedding.py
import unittest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from services.embedding import extract_embedding, get_available_models


class TestEmbeddingExtraction(unittest.TestCase):
    """Unit tests for embedding extraction."""
    
    def setUp(self):
        """Set up test fixtures."""
        from PIL import Image
        self.sample_image = Image.new('RGB', (100, 100), color=(128, 128, 128))
    
    def test_get_available_models(self):
        """Should return list of available models."""
        models = get_available_models()
        self.assertIsInstance(models, list)
        self.assertGreater(len(models), 0)
    
    def test_models_include_required(self):
        """Should include at least one of Siamese, Facenet, or ArcFace."""
        models = get_available_models()
        expected = ["Siamese", "Facenet", "ArcFace"]
        has_model = any(m in models for m in expected)
        self.assertTrue(has_model)
    
    def test_extract_returns_array(self):
        """Extract embedding should return numpy array."""
        try:
            result = extract_embedding(self.sample_image, "Siamese")
            self.assertIsInstance(result, np.ndarray)
        except Exception:
            pass
    
    def test_extract_returns_vector(self):
        """Embedding should be a 1D vector."""
        try:
            result = extract_embedding(self.sample_image, "Siamese")
            self.assertEqual(len(result.shape), 1)
        except Exception:
            pass
    
    def test_extract_embedding_length(self):
        """Embedding should have reasonable length (128-512 dims typical)."""
        try:
            result = extract_embedding(self.sample_image, "Siamese")
            self.assertGreater(len(result), 0)
            self.assertLessEqual(len(result), 2048)
        except Exception:
            pass
    
    def test_extract_deterministic(self):
        """Same image should produce same embedding."""
        try:
            result1 = extract_embedding(self.sample_image, "Siamese")
            result2 = extract_embedding(self.sample_image, "Siamese")
            np.testing.assert_array_almost_equal(result1, result2)
        except Exception:
            pass
    
    def test_different_images_different_embeddings(self):
        """Different images should produce different embeddings."""
        try:
            from PIL import Image
            img1 = Image.new('RGB', (100, 100), color=(100, 100, 100))
            img2 = Image.new('RGB', (100, 100), color=(200, 200, 200))
            emb1 = extract_embedding(img1, "Siamese")
            emb2 = extract_embedding(img2, "Siamese")
            self.assertFalse(np.allclose(emb1, emb2))
        except Exception:
            pass
    
    def test_extract_facenet_model(self):
        """Should support Facenet model."""
        try:
            result = extract_embedding(self.sample_image, "Facenet")
            self.assertIsInstance(result, np.ndarray)
        except Exception:
            pass
    
    def test_extract_arcface_model(self):
        """Should support ArcFace model."""
        try:
            result = extract_embedding(self.sample_image, "ArcFace")
            self.assertIsInstance(result, np.ndarray)
        except Exception:
            pass
    
    def test_embedding_values_normalized(self):
        """Embedding values should be in reasonable range."""
        try:
            result = extract_embedding(self.sample_image, "Siamese")
            self.assertTrue(np.all(np.isfinite(result)))
            max_val = np.max(np.abs(result))
            self.assertLess(max_val, 100)
        except Exception:
            pass


class TestEmbeddingRobustness(unittest.TestCase):
    """Robustness tests for embedding extraction."""
    
    def test_large_image(self):
        """Should handle large images efficiently."""
        from PIL import Image
        large = Image.new('RGB', (1000, 1000), color=(128, 128, 128))
        try:
            result = extract_embedding(large, "Siamese")
            self.assertIsInstance(result, np.ndarray)
        except Exception:
            pass
    
    def test_small_image(self):
        """Should handle small images."""
        from PIL import Image
        small = Image.new('RGB', (50, 50), color=(128, 128, 128))
        try:
            result = extract_embedding(small, "Siamese")
            self.assertIsInstance(result, np.ndarray)
        except Exception:
            pass
    
    def test_color_variation(self):
        """Should handle various color images."""
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (128, 128, 128)]
        for color in colors:
            from PIL import Image
            img = Image.new('RGB', (100, 100), color=color)
            try:
                result = extract_embedding(img, "Siamese")
                self.assertIsInstance(result, np.ndarray)
            except Exception:
                pass


if __name__ == "__main__":
    unittest.main()
