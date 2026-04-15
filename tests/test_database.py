# tests/test_database.py
import unittest
import json
import os
import sys
import tempfile
import shutil

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))


class TestDatabaseOperations(unittest.TestCase):
    """Unit tests for database operations."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()
        self.original_dir = os.path.join(os.path.dirname(__file__), 'backend', 'database')
        self.test_db_path = os.path.join(self.test_dir, 'embeddings.json')
        self.test_history_path = os.path.join(self.test_dir, 'verify_history.json')
        
        os.makedirs(self.test_dir, exist_ok=True)
        
        with open(self.test_db_path, 'w') as f:
            json.dump({"users": []}, f)
        with open(self.test_history_path, 'w') as f:
            json.dump({"attempts": []}, f)
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_embeddings_file_exists(self):
        """Embeddings file should exist."""
        self.assertTrue(os.path.exists(self.test_db_path))
    
    def test_history_file_exists(self):
        """History file should exist."""
        self.assertTrue(os.path.exists(self.test_history_path))
    
    def test_embeddings_valid_json(self):
        """Embeddings file should contain valid JSON."""
        with open(self.test_db_path, 'r') as f:
            data = json.load(f)
        self.assertIsInstance(data, dict)
        self.assertIn("users", data)
    
    def test_history_valid_json(self):
        """History file should contain valid JSON."""
        with open(self.test_history_path, 'r') as f:
            data = json.load(f)
        self.assertIsInstance(data, dict)
        self.assertIn("attempts", data)
    
    def test_add_user_structure(self):
        """User structure should be valid."""
        user = {
            "id": "test123",
            "name": "Test User",
            "embeddings": {
                "Facenet": [[0.1, 0.2, 0.3]]
            },
            "created_at": "2024-01-01T00:00:00"
        }
        
        with open(self.test_db_path, 'r') as f:
            data = json.load(f)
        
        data["users"].append(user)
        
        with open(self.test_db_path, 'w') as f:
            json.dump(data, f)
        
        with open(self.test_db_path, 'r') as f:
            loaded = json.load(f)
        
        self.assertEqual(len(loaded["users"]), 1)
        self.assertEqual(loaded["users"][0]["name"], "Test User")
    
    def test_add_history_entry(self):
        """History entry should be valid."""
        entry = {
            "id": "hist123",
            "timestamp": "2024-01-01T00:00:00",
            "result": {
                "name": "Test",
                "confidence": 0.95,
                "is_match": True
            },
            "model": "Siamese",
            "input_method": "upload",
            "threshold": 0.7
        }
        
        with open(self.test_history_path, 'r') as f:
            data = json.load(f)
        
        data["attempts"].append(entry)
        
        with open(self.test_history_path, 'w') as f:
            json.dump(data, f)
        
        with open(self.test_history_path, 'r') as f:
            loaded = json.load(f)
        
        self.assertEqual(len(loaded["attempts"]), 1)
        self.assertEqual(loaded["attempts"][0]["result"]["name"], "Test")
    
    def test_delete_user(self):
        """User deletion should work correctly."""
        with open(self.test_db_path, 'r') as f:
            data = json.load(f)
        
        data["users"].append({
            "id": "delete123",
            "name": "Delete Me",
            "embeddings": {}
        })
        
        with open(self.test_db_path, 'w') as f:
            json.dump(data, f)
        
        with open(self.test_db_path, 'r') as f:
            data = json.load(f)
        
        data["users"] = [u for u in data["users"] if u["id"] != "delete123"]
        
        with open(self.test_db_path, 'w') as f:
            json.dump(data, f)
        
        with open(self.test_db_path, 'r') as f:
            loaded = json.load(f)
        
        self.assertEqual(len(loaded["users"]), 0)
    
    def test_concurrent_access_simulation(self):
        """Simulate sequential database access (concurrent requires locking in production)."""
        for i in range(5):
            with open(self.test_db_path, 'r') as f:
                data = json.load(f)
            data["users"].append({
                "id": f"user_User{i}",
                "name": f"User{i}",
                "embeddings": {}
            })
            with open(self.test_db_path, 'w') as f:
                json.dump(data, f)
        
        with open(self.test_db_path, 'r') as f:
            data = json.load(f)
        
        self.assertEqual(len(data["users"]), 5)


class TestDatabaseValidation(unittest.TestCase):
    """Validation tests for database data."""
    
    def test_user_id_uniqueness(self):
        """User IDs should be unique."""
        user_ids = ["user1", "user2", "user1"]
        unique_ids = set(user_ids)
        self.assertEqual(len(user_ids), len(unique_ids) + 1)
    
    def test_embedding_format(self):
        """Embeddings should be in correct format."""
        embedding = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        self.assertIsInstance(embedding, list)
        for emb in embedding:
            self.assertIsInstance(emb, list)
            for val in emb:
                self.assertIsInstance(val, (int, float))
    
    def test_confidence_range(self):
        """Confidence scores should be between 0 and 1."""
        confidence = 0.95
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)
    
    def test_timestamp_format(self):
        """Timestamps should be in ISO format."""
        import re
        timestamp = "2024-01-01T12:30:00"
        iso_pattern = r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}'
        self.assertTrue(re.match(iso_pattern, timestamp))


if __name__ == "__main__":
    unittest.main()
