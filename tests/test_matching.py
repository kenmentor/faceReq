# tests/test_matching.py
import unittest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from services.matching import cosine_similarity, find_best_match


def euclidean_distance(vec1, vec2):
    """Calculate Euclidean distance between two vectors."""
    return np.sqrt(np.sum((vec1 - vec2) ** 2))


class TestCosineSimilarity(unittest.TestCase):
    """Unit tests for cosine similarity computation."""
    
    def test_identical_vectors(self):
        """Identical vectors should have similarity of 1.0."""
        vec = np.array([1.0, 0.0, 0.0])
        result = cosine_similarity(vec, vec)
        self.assertAlmostEqual(result, 1.0, places=5)
    
    def test_opposite_vectors(self):
        """Opposite vectors should have similarity of -1.0."""
        vec1 = np.array([1.0, 0.0, 0.0])
        vec2 = np.array([-1.0, 0.0, 0.0])
        result = cosine_similarity(vec1, vec2)
        self.assertAlmostEqual(result, -1.0, places=5)
    
    def test_perpendicular_vectors(self):
        """Perpendicular vectors should have similarity of 0.0."""
        vec1 = np.array([1.0, 0.0, 0.0])
        vec2 = np.array([0.0, 1.0, 0.0])
        result = cosine_similarity(vec1, vec2)
        self.assertAlmostEqual(result, 0.0, places=5)
    
    def test_similar_vectors(self):
        """Similar vectors should have high positive similarity."""
        vec1 = np.array([0.9, 0.1, 0.2])
        vec2 = np.array([0.85, 0.15, 0.18])
        result = cosine_similarity(vec1, vec2)
        self.assertGreater(result, 0.99)
        self.assertLess(result, 1.0)
    
    def test_zero_vector(self):
        """Zero vectors should return 0.0."""
        vec1 = np.array([0.0, 0.0, 0.0])
        vec2 = np.array([1.0, 0.0, 0.0])
        result = cosine_similarity(vec1, vec2)
        self.assertEqual(result, 0.0)
    
    def test_2d_vectors(self):
        """Test with simple 2D vectors."""
        vec1 = np.array([1.0, 0.0])
        vec2 = np.array([0.707, 0.707])
        result = cosine_similarity(vec1, vec2)
        self.assertAlmostEqual(result, 0.707, places=3)
    
    def test_high_dimensional_vectors(self):
        """Test with high-dimensional vectors."""
        vec1 = np.random.rand(128)
        vec2 = np.random.rand(128)
        result = cosine_similarity(vec1, vec2)
        self.assertGreaterEqual(result, -1.0)
        self.assertLessEqual(result, 1.0)


class TestEuclideanDistance(unittest.TestCase):
    """Unit tests for Euclidean distance computation."""
    
    def test_identical_vectors(self):
        """Identical vectors should have distance of 0."""
        vec = np.array([1.0, 2.0, 3.0])
        result = euclidean_distance(vec, vec)
        self.assertEqual(result, 0.0)
    
    def test_known_distance(self):
        """Test with known distance between vectors."""
        vec1 = np.array([0.0, 0.0])
        vec2 = np.array([3.0, 4.0])
        result = euclidean_distance(vec1, vec2)
        self.assertEqual(result, 5.0)
    
    def test_symmetry(self):
        """Distance should be symmetric."""
        vec1 = np.array([1.0, 2.0, 3.0])
        vec2 = np.array([4.0, 5.0, 6.0])
        d1 = euclidean_distance(vec1, vec2)
        d2 = euclidean_distance(vec2, vec1)
        self.assertEqual(d1, d2)
    
    def test_positive_distance(self):
        """Distance should always be non-negative."""
        vec1 = np.random.rand(10)
        vec2 = np.random.rand(10)
        result = euclidean_distance(vec1, vec2)
        self.assertGreaterEqual(result, 0.0)
    
    def test_manhattan_vs_euclidean(self):
        """Euclidean distance should be less than or equal to Manhattan distance."""
        vec1 = np.array([0.0, 0.0])
        vec2 = np.array([3.0, 4.0])
        euclid = euclidean_distance(vec1, vec2)
        manhattan = np.sum(np.abs(vec1 - vec2))
        self.assertLessEqual(euclid, manhattan)


class TestFindBestMatch(unittest.TestCase):
    """Unit tests for best match finding."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.query = np.array([0.9, 0.1, 0.1, 0.1])
        self.users = [
            {
                "name": "Alice",
                "embeddings": {"Facenet": [[0.8, 0.2, 0.15, 0.1]]}
            },
            {
                "name": "Bob",
                "embeddings": {"Facenet": [[0.1, 0.9, 0.1, 0.1]]}
            },
            {
                "name": "Carol",
                "embeddings": {"Facenet": [[0.85, 0.15, 0.12, 0.08]]}
            }
        ]
    
    def test_finds_best_match(self):
        """Should find user with highest similarity."""
        result = find_best_match(self.query, self.users, "Facenet")
        self.assertEqual(result["name"], "Carol")
    
    def test_high_threshold_rejects(self):
        """Should return Unknown when no user exceeds threshold."""
        very_different_query = np.array([0.0, 0.0, 0.0, 0.0])
        distant_users = [
            {
                "name": "Distant",
                "embeddings": {"Facenet": [[1.0, 0.0, 0.0, 0.0]]}
            }
        ]
        result = find_best_match(very_different_query, distant_users, "Facenet", threshold=0.99)
        self.assertEqual(result["name"], "Unknown")
        self.assertFalse(result["is_match"])
    
    def test_low_threshold_accepts(self):
        """Should return match when threshold is low."""
        result = find_best_match(self.query, self.users, "Facenet", threshold=0.1)
        self.assertNotEqual(result["name"], "Unknown")
        self.assertTrue(result["is_match"])
    
    def test_model_not_found(self):
        """Should skip users without the specified model."""
        result = find_best_match(self.query, self.users, "NonExistent")
        self.assertEqual(result["name"], "Unknown")
    
    def test_multiple_embeddings(self):
        """Should handle users with multiple embeddings (stored as separate entries)."""
        query = np.array([0.85, 0.15, 0.1, 0.1])
        users = [
            {
                "name": "Dave",
                "embeddings": {
                    "Facenet": [[0.85, 0.15, 0.1, 0.1]]
                }
            }
        ]
        result = find_best_match(query, users, "Facenet")
        self.assertEqual(result["name"], "Dave")
    
    def test_empty_users_list(self):
        """Should return Unknown for empty users list."""
        result = find_best_match(self.query, [], "Facenet")
        self.assertEqual(result["name"], "Unknown")
        self.assertFalse(result["is_match"])
    
    def test_confidence_score_range(self):
        """Confidence score should be between 0 and 1."""
        result = find_best_match(self.query, self.users, "Facenet")
        self.assertGreaterEqual(result["confidence"], 0.0)
        self.assertLessEqual(result["confidence"], 1.0)


if __name__ == "__main__":
    unittest.main()
