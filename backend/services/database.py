import json
import os
import shutil
import uuid
from datetime import datetime
from typing import List, Dict, Optional
import filelock

DATABASE_DIR = "database"
EMBEDDINGS_FILE = os.path.join(DATABASE_DIR, "embeddings.json")
HISTORY_FILE = os.path.join(DATABASE_DIR, "verify_history.json")
EMBEDDINGS_BACKUP = os.path.join(DATABASE_DIR, "embeddings_backup.json")
HISTORY_BACKUP = os.path.join(DATABASE_DIR, "history_backup.json")

_lock_file = os.path.join(DATABASE_DIR, "database.lock")


def init_database():
    os.makedirs(DATABASE_DIR, exist_ok=True)

    if not os.path.exists(EMBEDDINGS_FILE):
        with open(EMBEDDINGS_FILE, "w") as f:
            json.dump({"users": []}, f)

    if not os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "w") as f:
            json.dump({"attempts": []}, f)


def _read_json_safe(filepath: str) -> dict:
    lock = filelock.FileLock(_lock_file)
    with lock:
        try:
            with open(filepath, "r") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}


def _write_json_safe(filepath: str, data: dict, backup_path: Optional[str] = None):
    lock = filelock.FileLock(_lock_file)
    with lock:
        if backup_path and os.path.exists(filepath):
            shutil.copy(filepath, backup_path)

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)


def load_embeddings() -> dict:
    return _read_json_safe(EMBEDDINGS_FILE)


def save_embeddings(data: dict):
    _write_json_safe(EMBEDDINGS_FILE, data, EMBEDDINGS_BACKUP)


def load_history() -> dict:
    return _read_json_safe(HISTORY_FILE)


def save_history(data: dict):
    _write_json_safe(HISTORY_FILE, data, HISTORY_BACKUP)


def add_user(name: str, embeddings: Dict[str, List]) -> str:
    data = load_embeddings()
    user_id = str(uuid.uuid4())

    average_embeddings = {}
    for model_name, emb_list in embeddings.items():
        if isinstance(emb_list, list) and len(emb_list) > 0:
            arr = np.array(emb_list)
            if len(arr.shape) == 2:
                average = np.mean(arr, axis=0)
            else:
                average = arr
            average_embeddings[model_name] = average.tolist()
        else:
            average_embeddings[model_name] = emb_list

    user = {
        "id": user_id,
        "name": name,
        "enrolled_at": datetime.now().isoformat(),
        "embeddings": average_embeddings
    }

    data["users"].append(user)
    save_embeddings(data)

    return user_id


def get_user(user_id: str) -> Optional[Dict]:
    data = load_embeddings()
    for user in data.get("users", []):
        if user["id"] == user_id:
            return user
    return None


def delete_user(user_id: str) -> bool:
    data = load_embeddings()
    users = data.get("users", [])
    original_len = len(users)
    data["users"] = [u for u in users if u["id"] != user_id]

    if len(data["users"]) < original_len:
        save_embeddings(data)
        return True
    return False


def add_history_entry(result: Dict, model: str, input_method: str, threshold: float):
    data = load_history()

    entry = {
        "id": str(uuid.uuid4()),
        "timestamp": datetime.now().isoformat(),
        "result": result,
        "model": model,
        "input_method": input_method,
        "threshold": threshold
    }

    data["attempts"].append(entry)
    save_history(data)


import numpy as np
