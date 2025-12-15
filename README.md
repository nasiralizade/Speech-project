# Audio Classification with CNN and GPT-2

This project implements an audio classification system using a hybrid architecture combining Convolutional Neural Networks (CNN) for feature extraction and GPT-2 for sequence modeling. The model is trained and evaluated on the ESC-50 environmental sound classification dataset.

## Overview

The project uses a two-stage approach:
1. **Feature Extraction**: CNNs extract spatial features from MFCC spectrograms
2. **Sequence Modeling**: GPT-2 processes the temporal sequence of features for classification

Two models are trained and compared:
- **Fine-tuned Model**: Uses pre-trained GPT-2 with gradual unfreezing
- **Baseline Model**: Uses GPT-2 architecture without pre-trained weights

## Architecture

### AudioClassifier Model
- **Input**: MFCC spectrograms (2 channels × 100 time steps × 40 frequency bins)
- **CNN Layers**:
  - Conv2d(2 → 64) + BatchNorm + ReLU + MaxPool
  - Conv2d(64 → 128) + BatchNorm + ReLU + MaxPool
  - Output shape: (batch, 128, 25, 10)
- **Projection Layer**: Projects CNN features to GPT-2 embedding dimension
- **GPT-2 Transformer**: Processes sequence of 25 time-step tokens
- **Classification Head**: 
  - Linear(n_embd → 256) + ReLU
  - Linear(256 → num_classes)

### Training Strategy
- **Gradual Unfreezing**: GPT-2 layers are progressively unfrozen during training
  - Epoch 5: Unfreeze layer 0
  - Epoch 10: Unfreeze layer 1
  - Epoch 15: Unfreeze layer 2
- **Learning Rates**: 
  - GPT-2: 2e-5
  - Classification head: 1e-3
- **Optimization**: AdamW with weight decay (1e-2)
- **Scheduler**: ReduceLROnPlateau (patience=5, factor=0.5)
- **Early Stopping**: Stops if validation accuracy doesn't improve for 500 epochs

## Dataset

The project uses the **ESC-50** dataset:
- 50 environmental sound classes
- 2000 audio clips (5 seconds each)
- 40 examples per class
- Sample rate: 16 kHz

### Expected Directory Structure
```
data/
├── ESC-50/
│   ├── audio/          # Audio .wav files
│   └── meta/
│       └── esc50.csv   # Metadata file
└── preprocessed/       # Processed features (generated)
```

## Installation

### Requirements
```bash
pip install -r src/requirements
```

### Dependencies
- torch==2.0.1
- transformers==4.31.0
- librosa==0.10.1
- torchaudio==2.0.2
- numpy==1.24.3
- scikit-learn==1.3.0
- matplotlib==3.7.2
- pandas==2.0.3

## Usage

### 1. Data Preprocessing

Extract MFCC features and apply augmentation:

```bash
python src/preprocess.py
```

This will:
- Load ESC-50 audio files
- Extract 40 MFCC coefficients
- Apply SpecAugment for data augmentation
- Create 5-fold cross-validation splits
- Save preprocessed data to `data/preprocessed/preprocessed.pt`

**Features**:
- MFCC extraction with 40 coefficients
- Normalization (mean=0, std=1)
- SpecAugment: frequency and time masking
- Padding/truncating to fixed length (100 frames)
- 2-channel input: original + augmented

### 2. Training

Train both fine-tuned and baseline models:

```bash
python src/train.py
```

**Training Process**:
- Trains the fine-tuned model with pre-trained GPT-2
- Trains the baseline model without pre-trained weights
- Saves best models to `models/best_model.pt` and `models/best_baseline_model.pt`
- Uses early stopping based on validation accuracy
- Displays training/validation metrics per epoch

**Output**:
```
Epoch 1/20: Train Loss=3.2145, Train Acc=0.3421, Val Acc=0.4123, LR=0.002000
Best Model Updated: Acc=0.4123
```

### 3. Evaluation

Evaluate trained models on test set:

```bash
python src/evaluate.py
```

**Evaluation Metrics**:
- Accuracy
- Precision (weighted)
- Recall (weighted)
- F1-score (weighted)
- Confusion matrices (saved to `notebooks/`)

**Output**:
```
Fine-tuned Model: Acc=0.8234, Prec=0.8156, Rec=0.8234, F1=0.8187
Baseline Model: Acc=0.7512, Prec=0.7423, Rec=0.7512, F1=0.7461
```

## Model Details

### Device Support
The code automatically detects and uses the best available device:
1. MPS (Apple Silicon)
2. CUDA (NVIDIA GPUs)
3. CPU (fallback)

### Hyperparameters
- **Batch size**: 16
- **Number of epochs**: 20 (with early stopping)
- **Learning rate (GPT-2)**: 2e-5
- **Learning rate (head)**: 1e-3
- **Weight decay**: 1e-2
- **Patience**: 500 epochs
- **Number of classes**: 50

### Data Augmentation
- **SpecAugment**:
  - Frequency masking: up to 5 bins
  - Time masking: up to 10 frames
- **Noise injection**: Gaussian noise (σ=0.005)
- **Pitch shift**: ±2 semitones
- **Time stretch**: 0.8× to 1.2× speed

## Project Structure

```
Speech-project/
├── src/
│   ├── model.py         # Model architecture definitions
│   ├── train.py         # Training script
│   ├── evaluate.py      # Evaluation and metrics
│   ├── preprocess.py    # Data preprocessing
│   ├── test.py          # CNN architecture test
│   └── requirements     # Python dependencies
├── data/
│   ├── ESC-50/          # Dataset (not included)
│   └── preprocessed/    # Processed features (generated)
├── models/              # Saved model checkpoints (generated)
└── notebooks/           # Confusion matrices (generated)
```

## Results

The model achieves competitive performance on the ESC-50 dataset through:
- Pre-trained GPT-2 knowledge transfer
- Gradual unfreezing strategy
- SpecAugment and other augmentation techniques
- Early stopping to prevent overfitting

Results are saved in:
- Model checkpoints: `models/`
- Confusion matrices: `notebooks/`

## Notes

- The GPT-2 backbone is initially frozen and gradually unfrozen during training
- The baseline model uses the same architecture but without pre-trained weights
- Training is optimized with different learning rates for different components
- ReduceLROnPlateau scheduler adjusts learning rate based on validation performance

## License

This project is for educational and research purposes.
