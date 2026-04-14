import os
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

print("="*50)
print("SIAMESE NETWORK TRAINING")
print("="*50)

# Paths
POSITIVE_DIR = "C:/Users/MENTOR/Desktop/last/data/positive"
NEGATIVE_DIR = "C:/Users/MENTOR/Desktop/last/data/negative"
MODEL_DIR = "C:/Users/MENTOR/Desktop/last/backend/models"

os.makedirs(MODEL_DIR, exist_ok=True)

# Count and load images
pos_images = [f for f in os.listdir(POSITIVE_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))][:100]
neg_images = [f for f in os.listdir(NEGATIVE_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))][:200]

print(f"Positive: {len(pos_images)}, Negative: {len(neg_images)}")

IMG_SIZE = (100, 100)

def load_image(path):
    try:
        img = Image.open(path).convert('RGB')
        img = img.resize(IMG_SIZE)
        return np.array(img, dtype=np.float32) / 255.0
    except:
        return None

# Build model
class L2Norm(layers.Layer):
    def call(self, inputs):
        return tf.math.l2_normalize(inputs, axis=-1)

def create_model():
    inp = layers.Input(shape=(100, 100, 3))
    x = layers.Conv2D(32, 3, activation='relu', padding='same')(inp)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)
    
    x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)
    
    x = layers.Conv2D(128, 3, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling2D()(x)
    
    x = layers.Dense(128, activation='relu')(x)
    x = L2Norm()(x)
    
    emb_model = keras.Model(inp, x, name='embedding')
    
    in_a = keras.Input(shape=(100, 100, 3))
    in_b = keras.Input(shape=(100, 100, 3))
    
    emb_a = emb_model(in_a)
    emb_b = emb_model(in_b)
    
    diff = layers.Lambda(lambda v: tf.abs(v[0] - v[1]))([emb_a, emb_b])
    
    x = layers.Dense(64, activation='relu')(diff)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(32, activation='relu')(x)
    out = layers.Dense(1, activation='sigmoid')(x)
    
    siamese = keras.Model([in_a, in_b], out)
    return siamese, emb_model

# Create pairs
print("Creating pairs...")
pairs = []
labels = []

# Positive pairs
for i, f1 in enumerate(pos_images):
    for f2 in pos_images[i+1:i+5]:
        img1 = load_image(os.path.join(POSITIVE_DIR, f1))
        img2 = load_image(os.path.join(POSITIVE_DIR, f2))
        if img1 is not None and img2 is not None:
            pairs.append([img1, img2])
            labels.append(1)

# Negative pairs  
for i, f1 in enumerate(neg_images[:50]):
    for f2 in neg_images[i+1:i+51]:
        img1 = load_image(os.path.join(NEGATIVE_DIR, f1))
        img2 = load_image(os.path.join(NEGATIVE_DIR, f2))
        if img1 is not None and img2 is not None:
            pairs.append([img1, img2])
            labels.append(0)

print(f"Total pairs: {len(pairs)}")
pairs = np.array(pairs)
labels = np.array(labels)

# Split
split = int(len(labels) * 0.8)
X_a, X_b = pairs[:, 0], pairs[:, 1]

X_a_tr, X_a_te = X_a[:split], X_a[split:]
X_b_tr, X_b_te = X_b[:split], X_b[split:]
y_tr, y_te = labels[:split], labels[split:]

print(f"Train: {len(y_tr)}, Test: {len(y_te)}")

# Build and train
print("\nBuilding model...")
model, emb_model = create_model()
model.compile(optimizer=keras.optimizers.Adam(0.001), loss='binary_crossentropy', metrics=['accuracy'])

print("Training...")
model.fit([X_a_tr, X_b_tr], y_tr, validation_data=([X_a_te, X_b_te], y_te), epochs=15, batch_size=16, verbose=2)

# Save
save_path = os.path.join(MODEL_DIR, 'siamese_trained.h5')
emb_model.save(save_path)
print(f"\nSaved: {save_path}")

loss, acc = model.evaluate([X_a_te, X_b_te], y_te)
print(f"Test accuracy: {acc*100:.1f}%")