import os
import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict
import random

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


class FaceDataset:
    """
    Dataset class for face verification with both positive and negative pairs.
    """
    def __init__(self, positive_dir: str, negative_dir: str, img_size: int = 100):
        self.img_size = img_size
        self.positive_images = self._load_images(positive_dir)
        self.negative_images = self._load_images(negative_dir)
        
    def _load_images(self, data_dir: str) -> List[str]:
        """Load all images from directory."""
        images = []
        data_path = Path(data_dir)
        
        if not data_path.exists():
            return images
            
        for img_file in data_path.iterdir():
            if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                images.append(str(img_file))
        
        return images
    
    def get_positive_count(self) -> int:
        return len(self.positive_images)
    
    def get_negative_count(self) -> int:
        return len(self.negative_images)
    
    def generate_pairs(self, num_pairs: int = 10000) -> Tuple[List[Tuple[str, str]], List[int]]:
        """
        Generates positive and negative pairs.
        Positive: two images of the same person (from positive folder)
        Negative: two images of different people (positive vs negative)
        """
        pairs = []
        labels = []
        
        pos_images = self.positive_images
        neg_images = self.negative_images
        
        if len(pos_images) < 2:
            print("Warning: Not enough positive images for pairs!")
            return pairs, labels
        
        for _ in range(num_pairs):
            if random.random() < 0.5 and len(pos_images) >= 2:
                idx_a = random.randint(0, len(pos_images) - 1)
                idx_b = random.randint(0, len(pos_images) - 1)
                while idx_b == idx_a:
                    idx_b = random.randint(0, len(pos_images) - 1)
                
                pairs.append((pos_images[idx_a], pos_images[idx_b]))
                labels.append(1)
            else:
                img_a = random.choice(pos_images)
                img_b = random.choice(neg_images)
                
                pairs.append((img_a, img_b))
                labels.append(0)
        
        return pairs, labels
    
    def preprocess_image(self, image_path: str, detect_face: bool = False) -> np.ndarray:
        """Preprocess a single image - fast version."""
        img = cv2.imread(image_path)
        if img is None:
            return None
        
        img = cv2.resize(img, (self.img_size, self.img_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = (img - 0.5) / 0.5
        
        return img
    
    def load_batch(self, pairs: List[Tuple[str, str]], labels: List[int], 
                   augment: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Load a batch of pairs."""
        images_a = []
        images_b = []
        valid_labels = []
        
        for i, (img_path_a, img_path_b) in enumerate(pairs):
            img_a = self.preprocess_image(img_path_a)
            img_b = self.preprocess_image(img_path_b)
            
            if img_a is not None and img_b is not None:
                if augment:
                    img_a = self.augment_image(img_a)
                    img_b = self.augment_image(img_b)
                    
                images_a.append(img_a)
                images_b.append(img_b)
                valid_labels.append(labels[i])
        
        return np.array(images_a), np.array(images_b), np.array(valid_labels)
    
    def augment_image(self, img: np.ndarray) -> np.ndarray:
        """Apply data augmentation."""
        img = (img * 0.5 + 0.5)
        
        if random.random() < 0.5:
            img = np.fliplr(img)
        
        if random.random() < 0.4:
            brightness = random.uniform(0.7, 1.3)
            img = np.clip(img * brightness, 0, 1)
        
        if random.random() < 0.4:
            contrast = random.uniform(0.7, 1.3)
            mean = img.mean()
            img = np.clip((img - mean) * contrast + mean, 0, 1)
        
        if random.random() < 0.3:
            angle = random.uniform(-20, 20)
            h, w = img.shape[:2]
            M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
            img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        
        if random.random() < 0.2:
            noise = np.random.normal(0, 0.02, img.shape).astype(np.float32)
            img = np.clip(img + noise, 0, 1)
        
        if random.random() < 0.15:
            shift_x = random.randint(-8, 8)
            shift_y = random.randint(-8, 8)
            M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
            img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), borderMode=cv2.BORDER_REFLECT)
        
        img = (img - 0.5) / 0.5
        
        return img


def split_dataset(pairs: List, labels: List, train_ratio: float = 0.8) -> Tuple:
    """Split into train and validation."""
    num_train = int(len(pairs) * train_ratio)
    
    indices = list(range(len(pairs)))
    random.shuffle(indices)
    
    train_indices = indices[:num_train]
    val_indices = indices[num_train:]
    
    train_pairs = [pairs[i] for i in train_indices]
    train_labels = [labels[i] for i in train_indices]
    
    val_pairs = [pairs[i] for i in val_indices]
    val_labels = [labels[i] for i in val_indices]
    
    return train_pairs, train_labels, val_pairs, val_labels


if __name__ == "__main__":
    dataset = FaceDataset("model/data/positive", "model/data/negative")
    print(f"Positive images: {dataset.get_positive_count()}")
    print(f"Negative images: {dataset.get_negative_count()}")
    
    pairs, labels = dataset.generate_pairs(100)
    print(f"Generated {len(pairs)} pairs")
    print(f"Positive pairs (same person): {sum(labels)}")
    print(f"Negative pairs (different people): {len(labels) - sum(labels)}")
