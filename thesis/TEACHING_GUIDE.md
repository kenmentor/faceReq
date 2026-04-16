# Face Recognition System - Comprehensive Teaching Guide

*This guide provides a complete explanation of the FaceID face recognition system, from fundamental concepts to implementation details.*

---

## Table of Contents

1. [Introduction to Face Recognition](#1-introduction-to-face-recognition)
2. [How Face Recognition Works](#2-how-face-recognition-works)
3. [Deep Learning for Face Recognition](#3-deep-learning-for-face-recognition)
4. [Neural Network Models Explained](#4-neural-network-models-explained)
5. [The Siamese Network Architecture](#5-the-siamese-network-architecture)
6. [Embedding Vectors and Similarity](#6-embedding-vectors-and-similarity)
7. [System Architecture](#7-system-architecture)
8. [Backend Implementation](#8-backend-implementation)
9. [Model Selection and Configuration](#9-model-selection-and-configuration)
10. [API Design](#10-api-design)
11. [Database Structure](#11-database-structure)
12. [Frontend Integration](#12-frontend-integration)
13. [Testing and Evaluation](#13-testing-and-evaluation)

---

## 1. Introduction to Face Recognition

### 1.1 What is Face Recognition?

Face recognition is a biometric technology that identifies or verifies individuals by analyzing unique patterns in their facial features. Unlike traditional authentication methods like passwords or PINs, face recognition provides:

- **Contactless authentication** - No physical contact required
- **Natural interaction** - Works like human vision
- **Non-transferable** - Facial features cannot be forgotten or stolen
- **Speed** - Recognition happens in milliseconds

### 1.2 Face Recognition vs Face Detection

It's important to distinguish between two related but different concepts:

| Concept | Description | Purpose |
|---------|-------------|---------|
| **Face Detection** | Locating faces in an image | Find WHERE faces are |
| **Face Recognition** | Identifying WHO the face belongs to | Find WHO it is |

The system first detects faces, then recognizes them.

### 1.3 Verification vs Identification

Face recognition operates in two modes:

**Verification (1:1)**
- "Is this person who they claim to be?"
- Compare one face against one stored template
- Example: Phone unlock, passport verification
- Our system uses this mode

**Identification (1:N)**
- "Who is this person?"
- Compare one face against all stored templates
- Example: Criminal identification, watchlist screening

---

## 2. How Face Recognition Works

### 2.1 The Pipeline

A face recognition system follows a step-by-step pipeline:

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Image     │───▶│   Face      │───▶│  Embedding  │───▶│  Matching   │
│   Input     │    │   Detection  │    │  Extraction  │    │             │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
                          │                    │                    │
                    Find faces in       Convert face to      Compare against
                    the image           numerical vector      stored templates
```

### 2.2 Step-by-Step Process

**Step 1: Image Acquisition**
- Capture face image via camera or file upload
- Support various formats: JPEG, PNG, BMP
- Store as pixel data (height × width × color channels)

**Step 2: Face Detection**
- Locate faces within the image
- Return bounding box coordinates
- Handle multiple faces (use largest or prompt user)

**Step 3: Face Preprocessing**
- Crop to detected face region
- Resize to standard dimensions (e.g., 160×160 or 224×224)
- Normalize pixel values (0-255 → -1 to 1)
- Align face to standard pose

**Step 4: Embedding Extraction**
- Pass preprocessed face through neural network
- Network outputs a numerical vector (embedding)
- Vector captures facial features mathematically

**Step 5: Template Matching**
- Compare query embedding against stored templates
- Use similarity metric (cosine similarity)
- Apply threshold to decide match/no-match

---

## 3. Deep Learning for Face Recognition

### 3.1 Why Deep Learning?

Traditional computer vision methods used hand-crafted features:
- Edge detection
- Texture analysis
- Geometric measurements

These struggled with variations in:
- Lighting conditions
- Face pose and angle
- Facial expressions
- Aging and accessories

**Deep learning** learns features automatically from millions of images, achieving human-level or better accuracy.

### 3.2 Convolutional Neural Networks (CNNs)

CNNs are the foundation of modern image recognition. They process images through layers:

```
Input Image (224×224×3)
        │
        ▼
┌─────────────────┐
│ Conv Layer 1    │  Extracts edges and textures
│ (32 filters)    │
└─────────────────┘
        │
        ▼
┌─────────────────┐
│ Conv Layer 2    │  Combines into patterns
│ (64 filters)    │
└─────────────────┘
        │
        ▼
┌─────────────────┐
│ Conv Layer 3    │  Complex feature combinations
│ (128 filters)   │
└─────────────────┘
        │
        ▼
┌─────────────────┐
│ Pooling Layer   │  Reduces size, keeps important features
└─────────────────┘
        │
        ▼
┌─────────────────┐
│ Fully Connected  │  Makes final prediction
│ Output Layer     │
└─────────────────┘
```

### 3.3 How CNNs Learn

1. **Forward Pass**: Image → CNN → Prediction
2. **Compare**: Prediction vs Actual (calculate loss)
3. **Backward Pass**: Adjust weights to reduce loss
4. **Repeat**: Millions of times with millions of images

After training, the network's weights encode facial features.

---

## 4. Neural Network Models Explained

### 4.1 What Makes Models Different?

Different face recognition models vary in:

| Aspect | Description |
|--------|-------------|
| **Architecture** | How layers are organized |
| **Training Loss** | How the model learns to distinguish faces |
| **Embedding Size** | How many numbers represent each face |
| **Speed** | How fast it processes faces |
| **Accuracy** | How well it distinguishes people |

### 4.2 FaceNet Model

**Developed by Google (2015)**

Key innovation: **Triplet Loss**

```
┌─────────┐     ┌─────────┐     ┌─────────┐
│  Face A │     │  Face A'│     │  Face B │
│ (Anchor)│     │(Positive│     │(Negative│
│  Same    │     │ Same)   │     │ Different)
│  Person │     │ Person  │     │ Person  │
└────┬────┘     └────┬────┘     └────┬────┘
     │                │                │
     └────────────────┴────────────────┘
              Triplet Loss:
              Minimize distance(A, A')
              Maximize distance(A, B)
```

**Specifications:**
- Embedding size: **128 dimensions**
- Input size: 160×160 pixels
- Very fast inference
- Excellent balance of speed and accuracy

### 4.3 ArcFace Model

**Additive Angular Margin Loss (2018)**

Key innovation: **Angular margin in embedding space**

Instead of just making same faces close, ArcFace pushes different faces apart by a margin:

```
                    Decision Boundary
                         │
    Person A ───────────┼────────── Person B
                    margin gap
                         │
                    (Angular distance)
```

**Specifications:**
- Embedding size: **512 dimensions**
- Input size: 112×112 pixels
- Higher accuracy than FaceNet
- Slower due to larger embeddings

### 4.4 MobileNet Model

**Designed for mobile/edge devices**

Key innovation: **Depthwise separable convolutions**

Instead of one large convolution, uses two smaller ones:
- Depthwise: Filter each channel separately
- Pointwise: Combine channels

**Specifications:**
- Embedding size: **1280 dimensions**
- Very fast inference
- Slightly lower accuracy
- Efficient memory usage

### 4.5 Model Comparison

| Model | Embedding Size | Speed | Accuracy | Best For |
|-------|---------------|-------|----------|----------|
| FaceNet | 128 | Very Fast | High | Real-time applications |
| ArcFace | 512 | Medium | Very High | Maximum accuracy |
| MobileNet | 1280 | Fastest | Good | Mobile/embedded devices |

---

## 5. The Siamese Network Architecture

### 5.1 The Problem with Traditional Classification

Traditional CNNs for face recognition would:
- Have one output per person
- Require thousands of images per person
- Need retraining for new people

This doesn't scale!

### 5.2 Siamese Network Solution

A Siamese network learns to compare faces rather than classify them:

```
┌─────────────────────────────────────────────────────────┐
│                                                         │
│   ┌──────────┐         ┌──────────┐                    │
│   │ Network  │         │ Network  │     Siamese       │
│   │ (Clone)  │         │ (Clone)  │     Network       │
│   │    ①     │         │    ②     │                    │
│   └───┬──────┘         └────┬──────┘                    │
│       │                      │                           │
│       ▼                      ▼                           │
│   ┌────────┐            ┌────────┐                      │
│   │Embed A │            │Embed B │                      │
│   └────┬───┘            └────┬───┘                      │
│        │                      │                           │
│        └──────────┬──────────┘                           │
│                   ▼                                       │
│            ┌────────────┐                                │
│            │  Compare   │                                │
│            │    (Dist)  │                                │
│            └────────────┘                                │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### 5.3 How It Works

1. **Two identical networks** share weights
2. **Both networks** process their respective faces
3. **Outputs** are embeddings (numerical vectors)
4. **Distance** between embeddings is computed
5. **Small distance** = same person
6. **Large distance** = different people

### 5.4 Contrastive Loss

The network is trained to minimize:

```
Loss = (1 - Y) * D² + Y * max(0, margin - D)²

Where:
- Y = 1 if same person, 0 if different
- D = distance between embeddings
- margin = minimum separation between classes
```

**Key insight**: The network learns a "distance metric" that measures face similarity.

---

## 6. Embedding Vectors and Similarity

### 6.1 What is an Embedding?

An embedding is a list of numbers that represents a face:

```
Face Photo  ────▶  [0.234, -0.567, 0.891, -0.123, ...]  ────▶  128/512/1280 numbers
                    │
                    └── Embedding Vector (Feature Vector)
```

Each number captures some aspect of the face:
- Nose shape
- Eye position
- Jawline
- Distance between features
- And hundreds more...

### 6.2 Properties of Embeddings

**Normalized**: Embeddings have length 1 (unit vectors)
```
||embedding|| = √(x₁² + x₂² + ... + xₙ²) = 1
```

**Compact**: A face is reduced to just 128-512 numbers
- Original image: 160×160×3 = 76,800 values
- Embedding: 128 values
- 600× compression!

**Discriminative**: Similar faces have similar embeddings
- Same person: Distance ≈ 0.1 - 0.3
- Different people: Distance ≈ 0.6 - 1.0

### 6.3 Cosine Similarity

The most common way to compare embeddings:

```
Cosine Similarity = (A · B) / (||A|| × ||B||)

For normalized vectors:
Cosine Similarity = A · B = x₁y₁ + x₂y₂ + ... + xₙyₙ

Result: -1 to 1
- 1.0 = Identical
- 0.0 = Orthogonal (unrelated)
- -1.0 = Opposite
```

### 6.4 Threshold Selection

A threshold determines match vs. no-match:

```
Confidence ≥ Threshold  →  MATCH
Confidence < Threshold  →  NO MATCH
```

Typical thresholds:
- **0.4** - Permissive (more matches, more false positives)
- **0.5** - Balanced
- **0.6** - Strict (fewer matches, fewer false positives)
- **0.7** - Very strict

The choice depends on your security requirements.

---

## 7. System Architecture

### 7.1 Three-Tier Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        PRESENTATION TIER                             │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                    React Web Application                      │   │
│  │   ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐     │   │
│  │   │ Enroll  │  │ Verify  │  │ History │  │Settings │     │   │
│  │   │  Page   │  │  Page   │  │  Page   │  │  Page   │     │   │
│  │   └─────────┘  └─────────┘  └─────────┘  └─────────┘     │   │
│  └─────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  │ HTTP/REST
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        PROCESSING TIER                                │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                      FastAPI Backend                           │   │
│  │   ┌─────────────┐  ┌─────────────┐  ┌─────────────┐          │   │
│  │   │  API Layer  │  │  Services   │  │   Database  │          │   │
│  │   │ (Endpoints) │──│(Embedding, │──│  (JSON)     │          │   │
│  │   │             │  │  Matching)  │  │             │          │   │
│  │   └─────────────┘  └─────────────┘  └─────────────┘          │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                  │                                   │
│                                  ▼                                   │
│                    ┌─────────────────────────┐                     │
│                    │    TensorFlow/DeepFace   │                     │
│                    │    (Neural Network)      │                     │
│                    └─────────────────────────┘                     │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                          DATA TIER                                   │
│  ┌──────────────────────┐    ┌──────────────────────┐              │
│  │    embeddings.json    │    │ verify_history.json  │              │
│  │   (User Templates)    │    │   (Audit Trail)     │              │
│  └──────────────────────┘    └──────────────────────┘              │
│                                                                     │
│  ┌──────────────────────┐                                           │
│  │   models/config.json  │                                           │
│  │   (Configuration)     │                                           │
│  └──────────────────────┘                                           │
└─────────────────────────────────────────────────────────────────────┘
```

### 7.2 Component Interactions

```
User Browser                      FastAPI Server
     │                                  │
     │  POST /enroll                    │
     │  ─────────────────▶              │
     │                   ┌───────────────┴───────────────┐
     │                   │                                │
     │                   ▼                                ▼
     │          ┌──────────────┐              ┌──────────────┐
     │          │   Image     │              │    Image     │
     │          │ Preprocess  │              │ Preprocess   │
     │          └──────┬──────┘              └──────┬──────┘
     │                 │                             │
     │                 ▼                             ▼
     │          ┌──────────────┐              ┌──────────────┐
     │          │   Neural    │              │    Neural    │
     │          │   Network   │              │   Network    │
     │          └──────┬──────┘              └──────┬──────┘
     │                 │                             │
     │                 └──────────┬──────────────────┘
     │                            │
     │                            ▼
     │                   ┌──────────────┐
     │                   │   Database   │
     │                   │   (Save)     │
     │                   └──────────────┘
     │                            │
     │◀───────────────────────────│
     │        { "status": "ok" } │
```

---

## 8. Backend Implementation

### 8.1 Technology Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| Web Framework | FastAPI | REST API handling |
| Image Processing | Pillow, OpenCV | Image manipulation |
| Neural Networks | TensorFlow, DeepFace | Face detection & embedding |
| Data Storage | JSON Files | Persistent storage |
| Server | Uvicorn | ASGI server |

### 8.2 Key Modules

**embedding.py** - Face embedding extraction
```python
# Main function for extracting embeddings
def extract_embedding(image, model_name):
    """
    Extract face embedding using specified model.
    
    Args:
        image: PIL Image containing a face
        model_name: Name of model ('Siamese', 'Facenet', 'ArcFace', 'MobileNet')
    
    Returns:
        Normalized embedding vector
    """
    # 1. Initialize DeepFace engine
    # 2. Resolve model name to backend
    # 3. Preprocess image
    # 4. Extract embedding
    # 5. Normalize and return
```

**matching.py** - Face comparison
```python
# Compare query against enrolled users
def find_best_match(query_embedding, users, model_name, threshold=0.7):
    """
    Find best matching user for query embedding.
    
    Args:
        query_embedding: Face embedding to match
        users: List of enrolled users
        model_name: Model used for query
        threshold: Minimum similarity for match
    
    Returns:
        Dict with name, confidence, is_match
    """
    # 1. For each user, get stored embedding
    # 2. Calculate cosine similarity
    # 3. Track best match
    # 4. Return result with confidence
```

**detection.py** - Face detection
```python
# Locate faces in images
def detect_and_crop_face(image):
    """
    Detect face in image and crop to standard size.
    
    Returns:
        Tuple of (cropped_face_image, detected_bool)
    """
    # 1. Load face detector
    # 2. Detect face location
    # 3. Crop and resize
    # 4. Return with detection status
```

### 8.3 API Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/enroll` | POST | Register new user with face images |
| `/verify` | POST | Verify face against enrolled users |
| `/users` | GET | List all enrolled users |
| `/user/{id}` | DELETE | Remove user from database |
| `/history` | GET | Query verification history |
| `/models` | GET | List available models |

### 8.4 Enrollment Flow

```python
# Pseudocode for enrollment
async def enroll_user(name, files):
    embeddings = {}
    
    # Process each uploaded image
    for file in files:
        # Read image
        image = Image.open(file)
        
        # Detect face
        face, detected = detect_and_crop_face(image)
        if not detected:
            continue
        
        # Extract embedding for each model
        for model in ['Siamese', 'Facenet', 'ArcFace', 'MobileNet']:
            embedding = extract_embedding(face, model)
            embeddings[model].append(embedding.tolist())
    
    # Save to database
    user_id = add_user(name, embeddings)
    
    return {"status": "ok", "user_id": user_id}
```

### 8.5 Verification Flow

```python
# Pseudocode for verification
async def verify_face(file, model, threshold):
    # Read and process image
    image = Image.open(file)
    face, detected = detect_and_crop_face(image)
    
    if not detected:
        return {"name": "Unknown", "confidence": 0}
    
    # Extract embedding
    embedding = extract_embedding(face, model)
    
    # Load enrolled users
    users = load_embeddings()
    
    # Find best match
    result = find_best_match(embedding, users, model, threshold)
    
    return result
```

---

## 9. Model Selection and Configuration

### 9.1 Configuration File

The system uses a configuration file to manage model mappings:

```json
{
    "version": "2.0",
    "runtime": {
        "facenet": "Facenet",
        "arcface": "ArcFace"
    },
    "metadata": {
        "description": "Model runtime configuration"
    }
}
```

### 9.2 Model Registry

The registry maps user-facing model names to backend implementations:

```python
class ModelRegistry:
    """Central registry for model name mapping."""
    
    _map = {
        'facenet': 'Facenet',
        'arcface': 'ArcFace',
    }
    
    def get_backend(self, model_name):
        """Get backend for model."""
        return self._map.get(model_name.lower(), model_name)
```

### 9.3 Adding New Models

To add a new model:

1. Add mapping to configuration file
2. Ensure DeepFace supports the model
3. Test embedding extraction
4. Verify dimension compatibility

---

## 10. API Design

### 10.1 Enrollment Endpoint

**POST /enroll**

**Request:**
```
Content-Type: multipart/form-data

name: "John Doe"
files: [image1.jpg, image2.jpg, image3.jpg]
```

**Response:**
```json
{
    "status": "ok",
    "user_id": "550e8400-e29b-41d4-a716-446655440000",
    "name": "John Doe",
    "enrolled_with_models": ["Facenet", "ArcFace"],
    "images_processed": 3,
    "timing": {
        "total_ms": 1523.45,
        "face_detection_total_ms": 245.3,
        "embedding_extraction_total_ms": 1278.15
    }
}
```

### 10.2 Verification Endpoint

**POST /verify**

**Request:**
```
Content-Type: multipart/form-data

file: image.jpg
model: Facenet
threshold: 0.7
```

**Response:**
```json
{
    "name": "John Doe",
    "confidence": 0.85,
    "is_match": true,
    "model": "Facenet",
    "timing": {
        "total_ms": 0.423,
        "face_detection_ms": 0.072,
        "embedding_extraction_ms": 0.328,
        "matching_ms": 0.023
    },
    "face_detected": true,
    "enrolled_users_count": 5,
    "threshold_used": 0.7
}
```

### 10.3 Error Responses

```json
{
    "detail": "Minimum 3 images required for enrollment"
}
```

| Status Code | Meaning |
|------------|---------|
| 200 | Success |
| 400 | Bad request (validation error) |
| 404 | User not found |
| 500 | Server error |

---

## 11. Database Structure

### 11.1 embeddings.json

Stores enrolled users and their face embeddings:

```json
{
    "version": "2.0",
    "users": [
        {
            "id": "uuid-string",
            "name": "John Doe",
            "enrolled_at": "2024-01-15T10:30:00Z",
            "embeddings": {
                "Facenet": [0.123, -0.456, ...],
                "ArcFace": [0.789, 0.012, ...]
            }
        }
    ]
}
```

### 11.2 verify_history.json

Stores verification attempts for auditing:

```json
{
    "attempts": [
        {
            "id": "uuid-string",
            "timestamp": "2024-01-15T11:00:00Z",
            "result": {
                "name": "John Doe",
                "confidence": 0.85,
                "is_match": true,
                "model": "Facenet"
            },
            "model": "Facenet",
            "input_method": "upload",
            "threshold": 0.7
        }
    ]
}
```

### 11.3 Database Operations

| Operation | Function | Description |
|----------|----------|-------------|
| Add User | `add_user()` | Insert new user |
| Get User | `get_user()` | Retrieve user by ID |
| List Users | `load_embeddings()` | Get all enrolled users |
| Delete User | `delete_user()` | Remove user |
| Add History | `add_history_entry()` | Log verification |
| Get History | `load_history()` | Query history logs |

---

## 12. Frontend Integration

### 12.1 Technology Stack

| Component | Technology |
|-----------|------------|
| Framework | React 18 |
| Styling | Tailwind CSS |
| State | React Hooks |
| HTTP | Fetch API |
| Routing | Next.js Pages Router |

### 12.2 Page Structure

```
/frontend/src/app/
├── page.tsx              # Home page
├── enroll/
│   └── page.tsx          # User enrollment
├── verify/
│   └── page.tsx          # Face verification
├── history/
│   └── page.tsx          # Verification history
└── settings/
    └── page.tsx          # Model & threshold settings
```

### 12.3 API Integration

```typescript
// frontend/lib/api.ts

const API_BASE = 'http://localhost:8000';

export async function enrollUser(name: string, files: File[]) {
    const formData = new FormData();
    formData.append('name', name);
    files.forEach(file => formData.append('files', file));
    
    const response = await fetch(`${API_BASE}/enroll`, {
        method: 'POST',
        body: formData
    });
    
    return response.json();
}

export async function verifyFace(file: File, model: string, threshold: number) {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('model', model);
    formData.append('threshold', threshold.toString());
    
    const response = await fetch(`${API_BASE}/verify`, {
        method: 'POST',
        body: formData
    });
    
    return response.json();
}
```

### 12.4 Verification Page Flow

```typescript
function VerifyPage() {
    const [result, setResult] = useState(null);
    const [selectedModel, setSelectedModel] = useState('Facenet');
    
    async function handleVerify(file: File) {
        const response = await verifyFace(file, selectedModel, 0.7);
        setResult(response);
    }
    
    return (
        <div>
            <ModelSelector 
                models={['Facenet', 'Siamese', 'ArcFace', 'MobileNet']}
                selected={selectedModel}
                onChange={setSelectedModel}
            />
            <ImageUpload onUpload={handleVerify} />
            <ResultDisplay result={result} />
        </div>
    );
}
```

---

## 13. Testing and Evaluation

### 13.1 Types of Testing

| Type | Purpose | Tools |
|------|---------|-------|
| Unit Tests | Test individual functions | pytest |
| Integration Tests | Test component interaction | pytest |
| API Tests | Test HTTP endpoints | FastAPI TestClient |
| System Tests | End-to-end testing | Manual + automation |

### 13.2 Key Test Cases

**Face Detection Tests**
- Single face in image
- Multiple faces in image
- No face in image
- Partial face visible

**Embedding Tests**
- Extract with different models
- Verify embedding dimensions
- Check normalization
- Test with different image sizes

**Matching Tests**
- Same person matches
- Different people don't match
- Threshold boundaries
- Empty database handling

### 13.3 Performance Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| Detection Time | Face detection latency | < 100ms |
| Embedding Time | Extraction per image | < 500ms |
| Matching Time | Comparison latency | < 50ms |
| Memory Usage | RAM consumption | < 2GB |

### 13.4 Expected Results

| Metric | Facenet | ArcFace | MobileNet |
|--------|---------|---------|-----------|
| Embedding Dim | 128 | 512 | 1280 |
| Extract Speed | Fast | Medium | Very Fast |
| Match Accuracy | ~95% | ~98% | ~92% |
| Memory Usage | Low | Medium | Very Low |

---

## Appendix A: Glossary

| Term | Definition |
|------|------------|
| **Embedding** | Numerical vector representing a face |
| **CNN** | Convolutional Neural Network |
| **Facenet** | Google's face recognition model |
| **ArcFace** | Face model using angular margin loss |
| **MobileNet** | Lightweight model for mobile devices |
| **Siamese Network** | Neural network that compares pairs |
| **Cosine Similarity** | Measure of angle between vectors |
| **Threshold** | Decision boundary for matching |
| **Verification** | 1:1 face comparison |
| **DeepFace** | Library for face analysis |

## Appendix B: Quick Reference

### Common Commands

```bash
# Start backend server
cd backend
uvicorn main:app --reload --port 8000

# Run tests
cd tests
python run_tests.py

# Start frontend
cd frontend
npm run dev
```

### API Quick Reference

```bash
# Enroll user
curl -X POST http://localhost:8000/enroll \
  -F "name=John" \
  -F "files=@photo1.jpg" \
  -F "files=@photo2.jpg" \
  -F "files=@photo3.jpg"

# Verify face
curl -X POST http://localhost:8000/verify \
  -F "file=@test.jpg" \
  -F "model=Facenet" \
  -F "threshold=0.7"

# List users
curl http://localhost:8000/users

# Get history
curl http://localhost:8000/history
```

---

*End of Teaching Guide*
