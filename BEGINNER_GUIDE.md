# Face Recognition System - Beginner Guide

> **IMPORTANT: To view this file properly, install a Markdown extension in your editor:**
> - **VS Code**: Install "Markdown All in One" or "Markdown Preview Enhanced"
> - **Notepad++**: Install "Markdown Viewer"
> - **VS**: Install "Markdown Editor"
> - Or open in any web browser to read it

---

## What Is This Project?

This is a **Face Recognition System** that can identify people from photos. It uses AI (Siamese Neural Networks) to compare faces and determine if two images show the same person.

**Example use cases:**
- Doorbell that recognizes who's at the door
- Attendance system that logs when people arrive
- Photo app that groups photos by person

---

## Quick Start (5 minutes)

### 1. Open Command Prompt

Press `Win + R`, type `cmd`, press Enter.

### 2. Activate the Environment

```bash
conda activate codeml
```

### 3. Start the Backend Server

```bash
cd C:\Users\MENTOR\Desktop\last\backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

You should see:
```
Uvicorn running on http://0.0.0.0:8000
```

**Keep this window open!** The server must stay running.

### 4. Open the Frontend

Open your browser and go to:
```
http://localhost:3000
```

(Or wherever your frontend is running - check your frontend setup)

---

## How to Use the System

### Enrolling a New Person

"Enrolling" means teaching the system who a person is.

1. Go to the **Enroll** page
2. Enter the person's name
3. Upload **at least 3 photos** of their face
   - Photos should be clear and well-lit
   - Different angles help
4. Click **Enroll**

The system will store their face information for future recognition.

### Recognizing a Person

1. Go to the **Recognize** or **Verify** page
2. Choose a **model** from the dropdown:
   - **Siamese Network** - Your custom trained model
   - **Facenet** - Industry standard model
   - **ArcFace** - Another industry model
3. Upload a photo or use your webcam
4. Click **Recognize**

The system will tell you:
- **Who** it thinks the person is
- **How confident** it is (percentage)
- Whether it met the **threshold** to be considered a match

---

## Understanding the Models

| Model | Description | Best For |
|-------|-------------|----------|
| Siamese Network | Custom trained model | Your specific use case |
| Facenet | Google's model | General purpose |
| ArcFace | State-of-art model | High accuracy |

**Confidence Score:**
- Higher % = More confident the match is correct
- The **threshold** (default 0.7) determines when it's considered a match

---

## Troubleshooting

### "No face detected"

**Cause:** The photo doesn't show a clear face.

**Solutions:**
- Use a brighter, clearer photo
- Make sure the face is visible
- Try a different photo

### "Unknown" for enrolled person

**Causes:**
1. Person needs to re-enroll
2. Photo quality is different from enrollment photos
3. Face angle is very different

**Solutions:**
- Re-enroll the person with new photos
- Use photos taken in similar lighting
- Try multiple photos

### Server won't start

```bash
# Make sure port 8000 isn't in use
netstat -ano | findstr :8000
```

If something is using it, either:
- Stop that program
- Or change the port: `uvicorn main:app --port 8001`

---

## Training Your Own Model

### The Data Structure

Your training images should be organized like this:

```
C:\Users\MENTOR\Desktop\last\
├── data\
│   ├── positive\     ← Photos of SAME person (for learning "same")
│   └── negative\     ← Photos of DIFFERENT people (for learning "different")
```

### Running Training

1. Open command prompt
2. Activate environment: `conda activate codeml`
3. Go to project folder: `cd C:\Users\MENTOR\Desktop\last`
4. Run training:
   ```bash
   python train_siamese.py
   ```

The trained model will be saved to:
```
backend\models\siamese_trained.h5
```

### Training Tips

- **More positive images** = Better at recognizing same person
- **More negative images** = Better at telling different people apart
- **Variety** in lighting, angles, expressions helps

---

## Project Structure

```
C:\Users\MENTOR\Desktop\last\
├── backend\
│   ├── main.py           ← API server (FastAPI)
│   ├── services\        ← Core functionality
│   │   ├── detection.py    ← Finds faces in images
│   │   ├── embedding.py    ← Extracts face features
│   │   ├── matching.py     ← Compares faces
│   │   └── database.py     ← Stores enrolled people
│   ├── models\          ← Trained AI models
│   └── database\        ← Stored face data
├── frontend\            ← Web interface
├── train_siamese.py     ← Training script
└── data\                ← Training images
```

---

## API Reference (For Developers)

If you want to use the system programmatically:

### Check Available Models
```
GET http://localhost:8000/models
```

### Enroll a Person
```
POST http://localhost:8000/enroll
Form Data:
  - name: "John Doe"
  - files: [photo1.jpg, photo2.jpg, photo3.jpg]
```

### Recognize a Face
```
POST http://localhost:8000/verify
Form Data:
  - file: photo.jpg
  - model: "Siamese"
  - threshold: 0.7
```

---

## Common Terms

| Term | Meaning |
|------|---------|
| **Embedding** | A list of numbers that represents a face |
| **Cosine Similarity** | How to compare two embeddings |
| **Threshold** | Minimum confidence to call it a match |
| **Enroll** | Add a new person to the system |
| **Recognize** | Find out who a person is |

---

## Getting Help

If something isn't working:

1. **Check the server console** for error messages
2. **Restart the server** (Ctrl+C, then run uvicorn again)
3. **Re-enroll** the person if recognition keeps failing
4. **Check the database** folder exists and has write permissions

---

## Quick Command Reference

```bash
# Activate environment
conda activate codeml

# Start backend server
cd C:\Users\MENTOR\Desktop\last\backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Train the model
cd C:\Users\MENTOR\Desktop\last
python train_siamese.py

# Check if server is running
curl http://localhost:8000/
```

---

## That's It!

You now know how to use this face recognition system. Start by enrolling a few people, then try recognizing them!
