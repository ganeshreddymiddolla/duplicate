"""
dog_breed_classifier.py

Enhanced Keras training script for 120-class Dog Breed Classification.
Features:
- Two-stage training (Feature Extraction -> Fine Tuning) for higher accuracy.
- Integrated Data Augmentation layers (Rotation, Zoom, Contrast).
- Label Smoothing to prevent overfitting.
- EfficientNetB0 backbone.
"""

from pathlib import Path
import csv
import argparse
import sys
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers

# Configuration
IMG_SIZE = 224
BATCH_SIZE = 32
AUTOTUNE = tf.data.AUTOTUNE

def read_labels(labels_csv: Path):
    ids, breeds = [], []
    with labels_csv.open('r', newline='') as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            if not row: continue
            ids.append(row[0].strip())
            breeds.append(row[1].strip() if len(row) > 1 else '')
    return ids, breeds

def verify_dataset(root: Path):
    labels_csv = root / 'labels.csv'
    train_dir = root / 'train'
    
    if not labels_csv.exists() or not train_dir.exists():
        raise FileNotFoundError("Dataset missing (checked 'labels.csv' and 'train/').")

    ids, breeds = read_labels(labels_csv)
    train_files = {p.stem: str(p) for p in train_dir.glob('*.jpg')}
    
    valid_paths = []
    valid_labels = []
    
    for img_id, breed in zip(ids, breeds):
        if img_id in train_files:
            valid_paths.append(train_files[img_id])
            valid_labels.append(breed)
            
    print(f"✓ Found {len(valid_paths)} valid images.")
    return valid_paths, valid_labels

def build_label_mapping(breeds):
    classes = sorted(list(set(breeds)))
    idx = {c: i for i, c in enumerate(classes)}
    return idx, classes

def preprocess_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    return img

def make_dataset(filepaths, labels, shuffle=True):
    ds = tf.data.Dataset.from_tensor_slices((filepaths, labels))
    ds = ds.map(lambda x, y: (preprocess_image(x), y), num_parallel_calls=AUTOTUNE)
    if shuffle:
        ds = ds.shuffle(1024)
    ds = ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)
    return ds

def build_enhanced_model(num_classes):
    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    
    # 1. Integrated Augmentation for Robustness
    x = layers.RandomFlip("horizontal")(inputs)
    x = layers.RandomRotation(0.15)(x)
    x = layers.RandomZoom(0.1)(x)
    x = layers.RandomContrast(0.1)(x)
    
    # 2. Base Model (EfficientNetB0)
    base_model = tf.keras.applications.EfficientNetB0(
        include_top=False, 
        input_tensor=x, 
        weights="imagenet"
    )
    
    # Freeze base model initially
    base_model.trainable = False
    
    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    
    model = models.Model(inputs, outputs)
    return base_model, model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=12, help="Epochs for initial training")
    parser.add_argument('--fine_tune_epochs', type=int, default=10, help="Epochs for fine tuning")
    args = parser.parse_args()

    root = Path('.')
    paths, labels_raw = verify_dataset(root)
    
    label2idx, classes = build_label_mapping(labels_raw)
    y_indices = [label2idx[l] for l in labels_raw]
    y_onehot = tf.keras.utils.to_categorical(y_indices, num_classes=len(classes))
    
    # Save classes
    with open('classes.txt', 'w') as f:
        f.write('\n'.join(classes))
    
    # Stratified Split
    train_paths, val_paths, train_y, val_y = train_test_split(
        paths, y_onehot, test_size=0.15, stratify=y_indices, random_state=42
    )
    
    train_ds = make_dataset(train_paths, train_y, shuffle=True)
    val_ds = make_dataset(val_paths, val_y, shuffle=False)
    
    print(f"Classes: {len(classes)} | Train: {len(train_paths)} | Val: {len(val_paths)}")
    
    # Build Model
    base_model, model = build_enhanced_model(len(classes))
    
    # --- PHASE 1: Feature Extraction (Train Top Layer) ---
    print("\n--- Phase 1: Training Top Layers ---")
    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        metrics=['accuracy']
    )
    
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint('best_model_phase1.keras', save_best_only=True, monitor='val_accuracy'),
        tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)
    ]
    
    model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=callbacks)
    
    # --- PHASE 2: Fine-Tuning ---
    print("\n--- Phase 2: Fine-Tuning EfficientNet ---")
    base_model.trainable = True
    
    # Freeze all except top 30 layers of base model for fine-tuning
    for layer in base_model.layers[:-30]:
        layer.trainable = False
        
    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-5), # Low Learning Rate is CRITICAL
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        metrics=['accuracy']
    )
    
    ft_callbacks = [
        tf.keras.callbacks.ModelCheckpoint('final_model.keras', save_best_only=True, monitor='val_accuracy'),
        tf.keras.callbacks.EarlyStopping(patience=4, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(factor=0.2, patience=2)
    ]
    
    model.fit(train_ds, validation_data=val_ds, epochs=args.fine_tune_epochs, callbacks=ft_callbacks)
    
    print("✓ Training Complete. Saved 'final_model.keras' and 'classes.txt'.")

if __name__ == '__main__':
    main()