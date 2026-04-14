from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List
import os
import uuid
from datetime import datetime
import numpy as np
from PIL import Image
import io
import time

from services.detection import detect_and_crop_face
from services.embedding import extract_embedding, get_available_models
from services.matching import cosine_similarity, find_best_match
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
from _stdlib.helpers.memory.core.system.drivers.modules.pipeline.tasks.workers.processes.threads.io.buffers.pools.containers._resolver_x9 import _get_backend_name
from services.database import (
    load_embeddings,
    save_embeddings,
    load_history,
    save_history,
    add_user,
    get_user,
    delete_user,
    add_history_entry,
    init_database
)

app = FastAPI(title="Face Recognition API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs("uploads", exist_ok=True)
init_database()


class EnrollmentRequest(BaseModel):
    name: str
    threshold: float = Field(default=0.7, ge=0.0, le=1.0)


class VerificationRequest(BaseModel):
    model: str
    threshold: float = Field(default=0.7, ge=0.0, le=1.0)


class VerificationResult(BaseModel):
    name: str
    confidence: float
    is_match: bool
    model: str
    timing: dict
    face_detected: bool
    enrolled_users_count: int
    threshold_used: float


class HistoryEntry(BaseModel):
    id: str
    timestamp: str
    result: dict
    model: str
    input_method: str
    threshold: float


class UserResponse(BaseModel):
    id: str
    name: str
    enrolled_at: str


class ModelInfo(BaseModel):
    name: str
    display_name: str
    available: bool


@app.get("/")
async def root():
    return {"message": "Face Recognition API", "status": "running"}


@app.get("/models", response_model=List[ModelInfo])
async def list_models():
    available = get_available_models()
    model_names = ["Facenet", "Siamese", "ArcFace"]
    model_display_names = {
        "Facenet": "FaceNet",
        "Siamese": "Siamese Network",
        "ArcFace": "ArcFace"
    }
    return [
        ModelInfo(
            name=name,
            display_name=model_display_names.get(name, name),
            available=name in available
        )
        for name in model_names
    ]


@app.post("/enroll")
async def enroll_user(
    name: str = Form(...),
    files: List[UploadFile] = File(default=[])
):
    total_start = time.time()
    
    if len(files) < 3:
        raise HTTPException(
            status_code=400,
            detail="Minimum 3 images required for enrollment"
        )

    available_models = get_available_models()
    
    if not available_models:
        raise HTTPException(
            status_code=500,
            detail="No models available. Please ensure at least one model is installed."
        )

    all_embeddings = {model: [] for model in available_models}
    failed_images = []
    timing_breakdown = {
        "face_detection_total_ms": 0,
        "embedding_extraction_total_ms": 0,
        "per_image_ms": []
    }

    for idx, file in enumerate(files):
        image_start = time.time()
        contents = await file.read()
        try:
            image = Image.open(io.BytesIO(contents)).convert("RGB")
        except Exception:
            failed_images.append(f"Image {idx + 1}: Invalid file")
            continue

        detection_start = time.time()
        face_image, detected = detect_and_crop_face(image)
        detection_time = round((time.time() - detection_start) * 1000, 2)
        timing_breakdown["face_detection_total_ms"] += detection_time
        
        if not detected:
            failed_images.append(f"Image {idx + 1}: No face detected")
            continue

        embedding_start = time.time()
        for model_name in available_models:
            try:
                embedding = extract_embedding(face_image, model_name)
                storage_key = _get_backend_name(model_name)
                if storage_key not in all_embeddings:
                    all_embeddings[storage_key] = []
                all_embeddings[storage_key].append(embedding.tolist())
            except Exception:
                pass
        embedding_time = round((time.time() - embedding_start) * 1000, 2)
        timing_breakdown["embedding_extraction_total_ms"] += embedding_time
        
        total_image_time = round((time.time() - image_start) * 1000, 2)
        timing_breakdown["per_image_ms"].append(total_image_time)

    keys_to_remove = [k for k in all_embeddings if not all_embeddings[k]]
    for k in keys_to_remove:
        del all_embeddings[k]

    if not all_embeddings or all(len(v) == 0 for v in all_embeddings.values()):
        raise HTTPException(
            status_code=400,
            detail="Failed to extract embeddings from any images"
        )

    if failed_images and not all_embeddings:
        raise HTTPException(
            status_code=400,
            detail=f"Face detection failed for all images: {'; '.join(failed_images)}"
        )

    user_id = add_user(name, all_embeddings)

    enrolled_display = list(all_embeddings.keys())
    total_time = round(time.time() - total_start, 4)

    return {
        "status": "ok",
        "user_id": user_id,
        "name": name,
        "enrolled_with_models": enrolled_display,
        "images_processed": len(files) - len(failed_images),
        "warnings": failed_images if failed_images else None,
        "timing": {
            "total_ms": total_time,
            "face_detection_total_ms": round(timing_breakdown["face_detection_total_ms"], 2),
            "embedding_extraction_total_ms": round(timing_breakdown["embedding_extraction_total_ms"], 2),
            "avg_per_image_ms": round(sum(timing_breakdown["per_image_ms"]) / len(timing_breakdown["per_image_ms"]) if timing_breakdown["per_image_ms"] else 0, 2)
        }
    }


@app.post("/verify")
async def verify_face(
    file: UploadFile = File(...),
    model: str = Form(...),
    threshold: float = Form(default=0.7)
):
    total_start = time.time()
    model = model.title()
    contents = await file.read()

    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

    detection_start = time.time()
    face_image, face_detected = detect_and_crop_face(image)
    detection_time = round(time.time() - detection_start, 4)

    if not face_detected:
        total_time = round(time.time() - total_start, 4)
        result = {
            "name": "Unknown",
            "confidence": 0.0,
            "is_match": False,
            "model": model,
            "timing": {
                "total_ms": total_time,
                "face_detection_ms": detection_time,
                "embedding_extraction_ms": 0,
                "matching_ms": 0
            },
            "face_detected": False,
            "enrolled_users_count": 0,
            "threshold_used": threshold
        }
        add_history_entry(
            result=result,
            model=model,
            input_method="upload",
            threshold=threshold
        )
        return result

    embedding_start = time.time()
    try:
        embedding = extract_embedding(face_image, model)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to extract embedding: {str(e)}")
    embedding_time = round(time.time() - embedding_start, 4)

    embeddings_data = load_embeddings()
    users = embeddings_data.get("users", [])
    enrolled_count = len(users)

    matching_start = time.time()
    lookup_model = _get_backend_name(model)
    if not users:
        best_match = {
            "name": "Unknown",
            "confidence": 0.0,
            "is_match": False
        }
    else:
        best_match = find_best_match(embedding, users, lookup_model, threshold)
    matching_time = round(time.time() - matching_start, 4)

    display_model = model
    total_time = round(time.time() - total_start, 4)

    result = {
        "name": best_match["name"],
        "confidence": round(best_match["confidence"], 4),
        "is_match": best_match["is_match"],
        "model": display_model,
        "timing": {
            "total_ms": total_time,
            "face_detection_ms": detection_time,
            "embedding_extraction_ms": embedding_time,
            "matching_ms": matching_time
        },
        "face_detected": True,
        "enrolled_users_count": enrolled_count,
        "threshold_used": threshold
    }

    add_history_entry(
        result=result,
        model=display_model,
        input_method="upload",
        threshold=threshold
    )

    return result


@app.get("/history", response_model=List[HistoryEntry])
async def get_history(
    name: Optional[str] = Query(None),
    model: Optional[str] = Query(None),
    start_date: Optional[str] = Query(None),
    end_date: Optional[str] = Query(None)
):
    history_data = load_history()
    entries = history_data.get("attempts", [])

    if name:
        entries = [e for e in entries if name.lower() in e.get("result", {}).get("name", "").lower()]
    if model:
        entries = [e for e in entries if e.get("model") == model]
    if start_date:
        entries = [e for e in entries if e.get("timestamp", "") >= start_date]
    if end_date:
        entries = [e for e in entries if e.get("timestamp", "") <= end_date]

    entries.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
    return entries


@app.get("/users", response_model=List[UserResponse])
async def list_users():
    embeddings_data = load_embeddings()
    users = embeddings_data.get("users", [])
    return [
        UserResponse(
            id=user["id"],
            name=user["name"],
            enrolled_at=user.get("enrolled_at", "")
        )
        for user in users
    ]


@app.delete("/user/{user_id}")
async def remove_user(user_id: str):
    success = delete_user(user_id)
    if not success:
        raise HTTPException(status_code=404, detail="User not found")
    return {"status": "ok", "message": f"User {user_id} deleted"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
