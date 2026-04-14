import numpy as np
from typing import List, Dict, Tuple, Optional


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    vec1 = vec1.flatten()
    vec2 = vec2.flatten()

    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return float(dot_product / (norm1 * norm2))


def find_best_match(
    query_embedding: np.ndarray,
    users: List[Dict],
    model_name: str,
    threshold: float = 0.7
) -> Dict:
    best_match = {
        "name": "Unknown",
        "confidence": 0.0,
        "is_match": False
    }

    fallback_models = ["Siamese", "Facenet", "ArcFace"]
    if model_name in fallback_models:
        fallback_models = [m for m in fallback_models if m != model_name]
        fallback_models.insert(0, model_name)

    for user in users:
        embeddings = user.get("embeddings", {})
        stored_embedding = None
        
        for m in fallback_models:
            stored_embedding = embeddings.get(m)
            if stored_embedding is not None:
                break

        if stored_embedding is None:
            continue

        stored_vec = np.array(stored_embedding)
        if isinstance(stored_vec, list):
            if len(stored_vec) > 0 and isinstance(stored_vec[0], list):
                similarities = []
                for emb in stored_vec:
                    sim = cosine_similarity(query_embedding, np.array(emb))
                    similarities.append(sim)
                confidence = max(similarities) if similarities else 0.0
            else:
                confidence = cosine_similarity(query_embedding, stored_vec)
        else:
            confidence = cosine_similarity(query_embedding, stored_vec)

        if confidence > best_match["confidence"]:
            best_match = {
                "name": user["name"],
                "confidence": confidence,
                "is_match": confidence >= threshold
            }

    return best_match
