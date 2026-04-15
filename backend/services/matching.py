"""
Face matching service.

This module handles the comparison of face embeddings to find
the best matching enrolled user. It provides cosine similarity
based matching with support for multiple embedding formats.

Main Features:
- Cosine similarity computation for face comparison
- Best match finding across multiple enrolled users
- Support for multiple embedding formats (single, batch)
- Automatic fallback to compatible model embeddings
- Configurable matching threshold

Usage:
    from services.matching import find_best_match, cosine_similarity
    
    # Find best matching user
    result = find_best_match(embedding, users, "Facenet", threshold=0.7)
    
    if result["is_match"]:
        print(f"Matched: {result['name']} ({result['confidence']:.2%})")
"""

import numpy as np
from typing import List, Dict


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Calculate cosine similarity between two vectors.
    
    Cosine similarity measures the angle between two vectors,
    making it ideal for comparing normalized embeddings.
    A value of 1.0 means identical, 0.0 means orthogonal.
    
    Args:
        vec1: First embedding vector
        vec2: Second embedding vector
    
    Returns:
        Similarity score between -1.0 and 1.0
        Returns 0.0 if either vector has zero magnitude
    """
    vec1 = vec1.flatten()
    vec2 = vec2.flatten()

    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    # Avoid division by zero for zero vectors
    if norm1 == 0 or norm2 == 0:
        return 0.0

    return float(dot_product / (norm1 * norm2))


def find_best_match(
    query_embedding: np.ndarray,
    users: List[Dict],
    model_name: str,
    threshold: float = 0.7
) -> Dict:
    """
    Find the best matching user for a query embedding.
    
    Compares the query embedding against all enrolled users
    and returns the best match based on cosine similarity.
    
    Matching Process:
    1. Try exact model match first (same model used for query)
    2. Fall back to Facenet embeddings if exact match not available
    3. Compare query against each stored embedding
    4. Return the user with highest similarity above threshold
    
    Args:
        query_embedding: Face embedding to match (from verification)
        users: List of enrolled users with their stored embeddings
        model_name: Model used for the query embedding
        threshold: Minimum similarity score for a match (default 0.7)
    
    Returns:
        Dictionary containing:
        - name: Matched user name or "Unknown"
        - confidence: Similarity score (0.0 to 1.0)
        - is_match: True if confidence >= threshold
    """
    best_match = {
        "name": "Unknown",
        "confidence": 0.0,
        "is_match": False
    }

    # Preferred keys for embedding lookup
    # Facenet embeddings are the most reliable and widely compatible
    preferred_keys = ['Facenet', 'facenet', 'FaceNet']
    
    # Iterate through all enrolled users
    for user in users:
        embeddings = user.get("embeddings", {})
        
        # Collect candidate embeddings to compare
        candidates = []
        
        # First, prefer Facenet embeddings for best compatibility
        # These are stored consistently and work across model families
        for pref in preferred_keys:
            if pref in embeddings:
                candidates.append((pref, embeddings[pref]))
        
        # Then, try the exact model match if available and dimensions match
        # This handles cases where specific model embeddings exist
        if model_name in embeddings and model_name not in preferred_keys:
            stored = embeddings[model_name]
            stored_arr = np.array(stored).flatten()
            # Only use if dimensions match - embeddings must be same size
            if stored_arr.shape[0] == query_embedding.shape[0]:
                candidates.insert(0, (model_name, stored))
        
        # Compare against each candidate embedding
        for key, stored_vec in candidates:
            stored_arr = np.array(stored_vec)
            
            # Handle different embedding formats:
            # - Single embedding: 1D array (128,)
            # - Multiple embeddings: 2D array (N, 128) for averaged enrollment
            
            if stored_arr.ndim > 1 and stored_arr.shape[1] > 1:
                # Multiple embeddings - compare against all and take best
                similarities = []
                for emb in stored_arr:
                    emb_arr = np.array(emb)
                    # Only compare if dimensions match
                    if emb_arr.shape[0] == query_embedding.shape[0]:
                        sim = cosine_similarity(query_embedding, emb_arr)
                        similarities.append(sim)
                confidence = max(similarities) if similarities else 0.0
            else:
                # Single embedding - direct comparison
                stored_flat = stored_arr.flatten()
                # Skip if dimensions don't match
                if stored_flat.shape[0] != query_embedding.shape[0]:
                    continue
                confidence = cosine_similarity(query_embedding, stored_flat)

            # Update best match if this is higher confidence
            if confidence > best_match["confidence"]:
                best_match = {
                    "name": user["name"],
                    "confidence": confidence,
                    "is_match": confidence >= threshold
                }

    return best_match


__all__ = [
    'cosine_similarity',
    'find_best_match'
]
