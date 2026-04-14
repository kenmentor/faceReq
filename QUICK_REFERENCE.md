# Quick Reference Card

> **IMPORTANT: Install a Markdown extension to view this file properly:**
> - **VS Code**: "Markdown All in One"
> - **VS**: "Markdown Editor"
> - Or open in any browser

---

## Start the System

```bash
conda activate codeml
cd C:\Users\MENTOR\Desktop\last\backend
uvicorn main:app --reload --port 8000
```

Open browser: http://localhost:3000

---

## Basic Workflow

### Enroll (Teach who someone is)
1. Go to Enroll page
2. Enter name
3. Upload 3+ photos
4. Click Enroll

### Recognize (Find out who someone is)
1. Go to Recognize page
2. Select model
3. Upload photo or use webcam
4. Click Recognize

---

## Models Available

| Model | Use Case |
|-------|----------|
| Siamese | Your custom model |
| Facenet | General purpose |
| ArcFace | High accuracy |

---

## If Something Goes Wrong

| Problem | Solution |
|---------|----------|
| "No face detected" | Use clearer/better lit photo |
| "Unknown" for enrolled person | Re-enroll with new photos |
| Server won't start | Check port 8000, restart terminal |
| Import errors | Make sure `codeml` env is activated |

---

## File Locations

| File | Path |
|------|------|
| Backend code | `backend\main.py` |
| Trained model | `backend\models\siamese_trained.h5` |
| Training script | `train_siamese.py` |
| Training images | `data\positive\` and `data\negative\` |

---

## Important Notes

- Server must stay running while using the system
- Photos must contain a clear, visible face
- Minimum 3 photos needed for enrollment
- Siamese model uses Facenet under the hood
