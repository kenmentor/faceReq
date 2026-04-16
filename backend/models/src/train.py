import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from tensorflow import keras
import numpy as np
from pathlib import Path
import json
from datetime import datetime

from model.src.model import create_siamese_network, ContrastiveLoss, L2Normalize, L1Dist
from model.src.dataset import FaceDataset, split_dataset

TRAINING_LOG_FILE = "model/training_progress.txt"

def log_progress(message):
    """Log progress to file and print"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    log_msg = f"[{timestamp}] {message}"
    print(log_msg)
    with open(TRAINING_LOG_FILE, "a") as f:
        f.write(log_msg + "\n")

def save_training_status(status, details=""):
    """Save current training status"""
    status_file = "model/training_status.json"
    data = {
        "status": status,
        "details": details,
        "timestamp": datetime.now().isoformat()
    }
    with open(status_file, "w") as f:
        json.dump(data, f, indent=2)


def train_model(
    positive_dir: str,
    negative_dir: str,
    num_pairs: int = 10000,
    epochs: int = 30,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    margin: float = 0.5,
    img_size: int = 100,
    embedding_dim: int = 128,
    train_ratio: float = 0.8,
    checkpoint_dir: str = 'model/checkpoints',
    model_path: str = 'model/trained_model.h5',
    use_pretrained: bool = False,
    freeze_pretrained: bool = True
):
    if os.path.exists(TRAINING_LOG_FILE):
        os.remove(TRAINING_LOG_FILE)
    
    log_progress("=" * 50)
    log_progress("FACE VERIFICATION TRAINING STARTED")
    log_progress("=" * 50)
    save_training_status("loading_dataset", "Loading dataset...")
    
    print("Loading dataset...")
    dataset = FaceDataset(positive_dir, negative_dir, img_size)
    
    print(f"Positive images: {dataset.get_positive_count()}")
    print(f"Negative images: {dataset.get_negative_count()}")
    
    print(f"Generating {num_pairs} pairs...")
    pairs, labels = dataset.generate_pairs(num_pairs)
    
    print(f"Positive pairs: {sum(labels)}")
    print(f"Negative pairs: {len(labels) - sum(labels)}")
    
    print("Splitting into train/validation...")
    train_pairs, train_labels, val_pairs, val_labels = split_dataset(pairs, labels, train_ratio)
    
    print(f"Training pairs: {len(train_pairs)}")
    print(f"Validation pairs: {len(val_pairs)}")
    
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    
    print("Creating model...")
    model = create_siamese_network(
        input_shape=(img_size, img_size, 3),
        embedding_dim=embedding_dim,
        use_pretrained=use_pretrained,
        freeze_pretrained=freeze_pretrained
    )
    
    if use_pretrained:
        print("\n=== Using Pretrained MobileNetV2 (frozen={}) ===".format(freeze_pretrained))
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    print("Model compiled!")
    model.summary()
    
    print("\nLoading training data (this may take a few minutes)...")
    print("Preprocessing images...")
    
    import time
    start = time.time()
    X_train_a, X_train_b, y_train = dataset.load_batch(train_pairs, train_labels, augment=True)
    print(f"Training data loaded in {time.time()-start:.1f}s: {len(X_train_a)} samples")
    
    start = time.time()
    X_val_a, X_val_b, y_val = dataset.load_batch(val_pairs, val_labels, augment=False)
    print(f"Validation data loaded in {time.time()-start:.1f}s: {len(X_val_a)} samples")
    
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(checkpoint_dir, 'best_model.keras'),
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        ),
        keras.callbacks.CSVLogger(
            os.path.join(checkpoint_dir, 'training_log.csv'),
            append=True
        )
    ]
    
    print("\nStarting training...")
    history = model.fit(
        [X_train_a, X_train_b],
        y_train,
        validation_data=([X_val_a, X_val_b], y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )
    
    model.save(model_path)
    print(f"Model saved to {model_path}")
    
    training_history = {
        'history': {k: [float(v) for v in vals] for k, vals in history.history.items()},
        'config': {
            'epochs': epochs,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'margin': margin,
            'img_size': img_size,
            'embedding_dim': embedding_dim,
            'num_pairs': num_pairs,
            'train_pairs': len(train_pairs),
            'val_pairs': len(val_pairs)
        },
        'timestamp': datetime.now().isoformat()
    }
    
    with open('model/training_history.json', 'w') as f:
        json.dump(training_history, f, indent=2)
    
    return model, history


def evaluate_model(model, positive_dir: str, negative_dir: str, num_test: int = 1000):
    """Evaluate the trained model."""
    from model.src.dataset import FaceDataset
    
    dataset = FaceDataset(positive_dir, negative_dir)
    test_pairs, test_labels = dataset.generate_pairs(num_test)
    
    X_test_a, X_test_b, y_test = dataset.load_batch(test_pairs, test_labels)
    
    predictions = model.predict([X_test_a, X_test_b], verbose=0).flatten()
    
    threshold = 0.5
    pred_labels = (predictions > threshold).astype(int)
    
    accuracy = np.mean(pred_labels == y_test)
    true_pos = np.sum((pred_labels == 1) & (y_test == 1))
    false_pos = np.sum((pred_labels == 1) & (y_test == 0))
    true_neg = np.sum((pred_labels == 0) & (y_test == 0))
    false_neg = np.sum((pred_labels == 0) & (y_test == 1))
    
    precision = true_pos / (true_pos + false_pos + 1e-10)
    recall = true_pos / (true_pos + false_neg + 1e-10)
    f1_score = 2 * (precision * recall) / (precision + recall + 1e-10)
    
    results = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1_score),
        'threshold': threshold,
        'test_samples': len(y_test)
    }
    
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1_score:.4f}")
    print("="*50)
    
    return results


if __name__ == "__main__":
    import sys
    
    positive_dir = "model/data/positive"
    negative_dir = "model/data/negative"
    
    print("="*50)
    print("FACE VERIFICATION TRAINING")
    print("="*50)
    
    model, history = train_model(
        positive_dir=positive_dir,
        negative_dir=negative_dir,
        num_pairs=20000,
        epochs=50,
        batch_size=32,
        learning_rate=0.0001,
        margin=0.3,
        embedding_dim=256
    )
    
    print("\nEvaluating model...")
    results = evaluate_model(model, positive_dir, negative_dir)
    
    print("\nTraining complete!")
