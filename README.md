# Pawdentify Dog Breed Classifier

A single-file Keras/TensorFlow training script for 120-class dog breed classification.

## Dataset

- **Training images:** `train/` (10,222 JPG files)
- **Test images:** `test/` (10,357 JPG files)
- **Labels:** `labels.csv` (id, breed mapping)
- **Submission template:** `sample_submission.csv`

## Setup

### 1. Install Dependencies

```powershell
pip install -r requirements.txt
```

### 2. Verify Dataset (Optional)

Check for missing images or labels:

```powershell
python .\verify_dataset.py
```

Expected output:
```
labels.csv entries: 10222
train images: 10222
Missing in `train/`: 0
Missing in `labels.csv`: 0
```

## Training

### Quick Start (Development)
Train for 1 epoch to test the pipeline:

```powershell
python .\dog_breed_classifier.py --epochs 1 --batch_size 8
```

**Runtime:** ~27 minutes (CPU), ~5 minutes (GPU)

### Production Training
Train for 20 epochs with standard batch size:

```powershell
python .\dog_breed_classifier.py --epochs 20 --batch_size 16
```

**Runtime:** ~9 hours (CPU), ~45 minutes (GPU)

### Resume Training (No Starting Over!)

If training is interrupted, **resume from the best checkpoint**:

```powershell
python .\dog_breed_classifier.py --epochs 20 --batch_size 16 --resume
```

This loads `best_model.keras` and continues training for the specified number of additional epochs. All progress is saved automatically.

**Example workflow:**
```powershell
# Session 1: Train for 10 epochs
python .\dog_breed_classifier.py --epochs 10 --batch_size 16

# Session 2 (next day): Resume and train 10 more epochs
python .\dog_breed_classifier.py --epochs 10 --batch_size 16 --resume
# Total training = 20 epochs, no starting over!
```

### Training Options

```
--epochs INT            Number of training epochs (default: 8)
--batch_size INT        Batch size for training (default: 32)
--val_split FLOAT       Validation split ratio (default: 0.15)
--strict                Abort if dataset has missing ids/labels (default: no)
--resume                Resume from best_model.keras checkpoint (default: no)
```

Example:
```powershell
python .\dog_breed_classifier.py --epochs 20 --batch_size 16 --val_split 0.2 --resume
```

## Outputs

After training, the following files are created:

- **`best_model.keras`** — Best checkpoint (saved by ModelCheckpoint callback)
- **`final_model.keras`** — Final model after all epochs
- **`saved_model/`** — TensorFlow SavedModel format (for TF Lite / TF Serving)
- **`classes.txt`** — Ordered list of 120 breed class names

## Inference

### Load and Test a Trained Model

Use the provided inference script:

```powershell
python .\infer.py --model_path best_model.keras --image_path train\<image_id>.jpg
```

Example:
```powershell
python .\infer.py --model_path best_model.keras --image_path "train\1d4f3c53ba70bc0fdb6e43d4f1f8d21a.jpg"
```

**Output:**
```
Top 3 predictions:
  1. labrador_retriever: 0.8234
  2. golden_retriever: 0.1203
  3. chesapeake_bay_retriever: 0.0412
```

### Generate Kaggle Submission

Create predictions for all test images:

```powershell
python .\infer.py --model_path best_model.keras --output_csv submission.csv
```

This generates `submission.csv` with columns: `id`, and 120 breed probability columns (same format as `sample_submission.csv`).

## File Structure

```
dogs_breed/
├── README.md                      (this file)
├── requirements.txt               (dependencies)
├── labels.csv                     (train labels)
├── sample_submission.csv          (submission template)
├── verify_dataset.py              (dataset checker)
├── verify_dataset.ipynb           (dataset checker notebook)
├── dog_breed_classifier.py        (training script)
├── infer.py                       (inference/prediction script)
├── train/                         (10,222 training images)
├── test/                          (10,357 test images)
├── best_model.keras              (saved checkpoint, created after training)
├── final_model.keras             (final model, created after training)
├── saved_model/                  (TensorFlow SavedModel, created after training)
└── classes.txt                   (breed names, created after training)
```

## Performance Notes

- **Accuracy baseline (random):** ~0.83% (120 classes)
- **1-epoch smoke test:** ~8.67% validation accuracy (expected, untrained)
- **20-epoch training:** expected ~70–80% validation accuracy (varies with GPU/hyperparams)

## GPU Acceleration (Optional)

If you have NVIDIA GPU with CUDA installed:

```powershell
pip install tensorflow[and-cuda]
```

TensorFlow will automatically detect and use GPU. Training on GPU is **~5-10x faster** than CPU.

## Troubleshooting

### "FileNotFoundError: labels.csv not found"
Ensure you run commands from the `dogs_breed/` directory.

### "ValueError: Invalid filepath extension"
If using an older Keras version, the script uses `.keras` format (Keras 3+). Update:
```powershell
pip install --upgrade tensorflow keras
```

### Out of memory errors
Reduce `--batch_size`:
```powershell
python .\dog_breed_classifier.py --epochs 20 --batch_size 8
```

### Model training is very slow
- Use GPU: install `tensorflow[and-cuda]`
- Reduce epochs or batch size for testing

## References

- Model: EfficientNetB0 (pretrained on ImageNet)
- Data: Kaggle Dog Breed Identification dataset
- Framework: TensorFlow 2.10+, Keras 3+
