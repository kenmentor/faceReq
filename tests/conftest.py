# tests/conftest.py
import pytest
import sys
import os
import io
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))


@pytest.fixture
def sample_face_image():
    """Generate a sample face-like image for testing."""
    img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    return Image.fromarray(img)


@pytest.fixture
def sample_face_bytes():
    """Generate face image as bytes."""
    img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    img_bytes = io.BytesIO()
    Image.fromarray(img).save(img_bytes, format='JPEG')
    img_bytes.seek(0)
    return img_bytes


@pytest.fixture
def sample_face_file(sample_face_bytes):
    """Create a file-like object for upload."""
    return ('test.jpg', sample_face_bytes, 'image/jpeg')


@pytest.fixture
def blank_image():
    """Create a blank white image."""
    return Image.new('RGB', (640, 480), color=(255, 255, 255))


@pytest.fixture
def random_embedding():
    """Generate a random embedding vector."""
    return np.random.rand(128).tolist()


@pytest.fixture
def sample_users():
    """Generate sample users for testing."""
    return [
        {
            "id": "user1",
            "name": "Alice",
            "embeddings": {
                "Facenet": [[0.9, 0.1, 0.1] + [0.0] * 125],
                "Siamese": [[0.85, 0.15, 0.1] + [0.0] * 125]
            }
        },
        {
            "id": "user2",
            "name": "Bob",
            "embeddings": {
                "Facenet": [[0.1, 0.9, 0.1] + [0.0] * 125],
                "Siamese": [[0.15, 0.85, 0.1] + [0.0] * 125]
            }
        },
        {
            "id": "user3",
            "name": "Carol",
            "embeddings": {
                "Facenet": [[0.8, 0.2, 0.15] + [0.0] * 125],
                "ArcFace": [[0.75, 0.2, 0.2] + [0.0] * 125]
            }
        }
    ]


@pytest.fixture
def temp_database():
    """Create a temporary database for testing."""
    import tempfile
    import json
    import shutil
    
    temp_dir = tempfile.mkdtemp()
    db_path = os.path.join(temp_dir, 'test_embeddings.json')
    history_path = os.path.join(temp_dir, 'test_history.json')
    
    with open(db_path, 'w') as f:
        json.dump({"users": []}, f)
    with open(history_path, 'w') as f:
        json.dump({"attempts": []}, f)
    
    yield {"dir": temp_dir, "db": db_path, "history": history_path}
    
    shutil.rmtree(temp_dir, ignore_errors=True)


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )
