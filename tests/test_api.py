# tests/test_api.py
import unittest
import io
import numpy as np
import sys
import os
import time

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.join(os.path.dirname(TESTS_DIR), 'backend')
sys.path.insert(0, BACKEND_DIR)
os.chdir(BACKEND_DIR)

from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


class TestRootEndpoints(unittest.TestCase):
    """Test root and health endpoints."""
    
    def test_root_returns_status(self):
        """Root endpoint should return status."""
        response = client.get("/")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue("status" in data or "message" in data)


class TestModelEndpoints(unittest.TestCase):
    """Test model-related endpoints."""
    
    def test_models_endpoint_exists(self):
        """Should return list of available models."""
        response = client.get("/models")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIsInstance(data, list)
        self.assertGreaterEqual(len(data), 1)
    
    def test_models_have_required_fields(self):
        """Each model should have name, display_name, and available fields."""
        response = client.get("/models")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        for model in data:
            self.assertIn("name", model)
            self.assertIn("display_name", model)
            self.assertIn("available", model)


class TestEnrollmentEndpoint(unittest.TestCase):
    """Test user enrollment endpoint."""
    
    def test_enroll_requires_minimum_images(self):
        """Enrollment should require at least 3 images."""
        response = client.post("/enroll", data={"name": "Test User"})
        self.assertEqual(response.status_code, 400)
    
    def test_enroll_missing_name(self):
        """Enrollment should require a name."""
        files = [
            ("files", ("img1.jpg", io.BytesIO(b"fake"), "image/jpeg"))
            for _ in range(3)
        ]
        response = client.post("/enroll", data={}, files=files)
        self.assertEqual(response.status_code, 422)


class TestVerificationEndpoint(unittest.TestCase):
    """Test face verification endpoint."""
    
    def test_verify_requires_file(self):
        """Verification should require an image file."""
        response = client.post("/verify", data={"model": "Siamese"})
        self.assertEqual(response.status_code, 422)
    
    def test_verify_requires_model(self):
        """Verification should require a model parameter."""
        response = client.post("/verify", files={"file": ("test.jpg", io.BytesIO(b"fake"), "image/jpeg")})
        self.assertEqual(response.status_code, 422)
    
    def test_verify_invalid_model(self):
        """Verification should validate model name."""
        files = {"file": ("test.jpg", io.BytesIO(b"fake"), "image/jpeg")}
        response = client.post("/verify", data={"model": "InvalidModel"}, files=files)
        self.assertIn(response.status_code, [400, 422, 500])


class TestHistoryEndpoint(unittest.TestCase):
    """Test verification history endpoint."""
    
    def test_get_history(self):
        """Should return list of verification history."""
        response = client.get("/history")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIsInstance(data, list)
    
    def test_history_filter_by_model(self):
        """Should filter history by model name."""
        response = client.get("/history?model=Siamese")
        self.assertEqual(response.status_code, 200)
    
    def test_history_filter_by_name(self):
        """Should filter history by user name."""
        response = client.get("/history?name=John")
        self.assertEqual(response.status_code, 200)


class TestUserManagement(unittest.TestCase):
    """Test user management endpoints."""
    
    def test_get_users(self):
        """Should return list of users."""
        response = client.get("/users")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIsInstance(data, list)
    
    def test_delete_nonexistent_user(self):
        """Should return 404 for nonexistent user."""
        response = client.delete("/user/nonexistent-id-12345")
        self.assertEqual(response.status_code, 404)


class TestSettingsEndpoint(unittest.TestCase):
    """Test settings endpoint."""
    
    def test_get_settings(self):
        """Should return current settings (if endpoint exists)."""
        response = client.get("/settings")
        if response.status_code == 404:
            self.skipTest("Settings endpoint not implemented")
        self.assertEqual(response.status_code, 200)


class TestPerformance(unittest.TestCase):
    """Performance-related tests."""
    
    def test_response_time_under_load(self):
        """Verify endpoint should respond within reasonable time."""
        files = {"file": ("test.jpg", io.BytesIO(np.random.bytes(1000)), "image/jpeg")}
        start = time.time()
        response = client.post(
            "/verify",
            data={"model": "Siamese", "threshold": 0.7},
            files=files
        )
        elapsed = time.time() - start
        self.assertLess(elapsed, 5.0)


if __name__ == "__main__":
    unittest.main()
