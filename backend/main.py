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
    model_names = ["Siamese", "Facenet", "ArcFace"]
    model_display_names = {
        "Siamese": "Siamese Network",
        "Facenet": "Facenet (DeepFace)",
        "ArcFace": "ArcFace (DeepFace)"
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
    print(f"[DEBUG] enroll_user called with name: {name}, files: {len(files)}")
    
    if len(files) < 3:
        raise HTTPException(
            status_code=400,
            detail="Minimum 3 images required for enrollment"
        )

    available_models = get_available_models()
    print(f"[DEBUG] Available models: {available_models}")
    
    if not available_models:
        raise HTTPException(
            status_code=500,
            detail="No models available. Please ensure at least one model is installed."
        )

    all_embeddings = {model: [] for model in available_models}
    failed_images = []

    for idx, file in enumerate(files):
        print(f"[DEBUG] Processing file {idx + 1}")
        contents = await file.read()
        try:
            image = Image.open(io.BytesIO(contents)).convert("RGB")
            print(f"[DEBUG] Image loaded: {image.size}")
        except Exception as e:
            print(f"[DEBUG] Image open error: {e}")
            failed_images.append(f"Image {idx + 1}: Invalid file")
            continue

        face_image, detected = detect_and_crop_face(image)
        print(f"[DEBUG] Face detection result: {detected}")
        if not detected:
            failed_images.append(f"Image {idx + 1}: No face detected")
            continue

        for model_name in available_models:
            try:
                print(f"[DEBUG] Extracting embedding with {model_name}")
                embedding = extract_embedding(face_image, model_name)
                print(f"[DEBUG] Embedding shape: {embedding.shape}")
                
                storage_model = _get_backend_name(model_name)
                all_embeddings[storage_model].append(embedding.tolist())
            except Exception as e:
                print(f"[DEBUG] Embedding error for {model_name}: {e}")

    for model_name in available_models:
        if not all_embeddings[model_name]:
            del all_embeddings[model_name]

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

    enrolled_display = ["Siamese" if m == "Facenet" else m for m in all_embeddings.keys()]

    return {
        "status": "ok",
        "user_id": user_id,
        "name": name,
        "enrolled_with_models": enrolled_display,
        "images_processed": len(files) - len(failed_images),
        "warnings": failed_images if failed_images else None
    }


@app.post("/verify")
async def verify_face(
    file: UploadFile = File(...),
    model: str = Form(...),
    threshold: float = Form(default=0.7)
):

    model = _get_backend_name(model.title())
    contents = await file.read()

    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

    face_image, face_detected = detect_and_crop_face(image)
    if not face_detected:
        raise HTTPException(status_code=400, detail="No face detected in image")

    try:
        embedding = extract_embedding(face_image, model)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to extract embedding: {str(e)}")

    embeddings_data = load_embeddings()
    users = embeddings_data.get("users", [])

    if not users:
        result = {
            "name": "Unknown",
            "confidence": 0.0,
            "is_match": False,
            "model": model
        }
        add_history_entry(
            result=result,
            model=model,
            input_method="upload",
            threshold=threshold
        )
        return result

    best_match = find_best_match(embedding, users, model, threshold)

    display_model = "Siamese" if model == "Facenet" else model
    result = {
        "name": best_match["name"],
        "confidence": round(best_match["confidence"], 4),
        "is_match": best_match["is_match"],
        "model": display_model
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
