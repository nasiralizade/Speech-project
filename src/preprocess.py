import os
import torch
import librosa
import numpy as np
from sklearn.model_selection import KFold
import torchaudio
import pandas as pd


def load_esc50(data_dir, sr=16000):
    audio_files, labels = [], []
    print(f"Scanning directory: {data_dir}")
    if not os.path.exists(data_dir):
        print(f"Error: Directory {data_dir} does not exist")
        return audio_files, labels

    # Load ESC-50 metadata
    meta_path = os.path.join(os.path.dirname(data_dir), 'meta', 'esc50.csv')
    if not os.path.exists(meta_path):
        print(f"Error: Metadata file {meta_path} not found")
        return audio_files, labels

    metadata = pd.read_csv(meta_path)
    print(f"Loaded metadata with {len(metadata)} entries")

    # Create a dictionary mapping filenames to category IDs
    label_map = {row['filename']: row['category'] for _, row in metadata.iterrows()}
    category_to_id = {cat: idx for idx, cat in enumerate(sorted(set(metadata['category'])))}

    for root, _, files in os.walk(data_dir):
        print(f"Checking directory: {root}")
        for file in files:
            if file.lower().endswith('.wav'):
                if file in label_map:
                    category = label_map[file]
                    label = category_to_id[category]  # 0-based label
                    if label < 0 or label >= 50:
                        print(f"Warning: Invalid label {label} for file {file}, skipping")
                        continue
                    audio_files.append(os.path.join(root, file))
                    labels.append(label)
                else:
                    print(f"Warning: File {file} not found in metadata, skipping")
    print(f"Found {len(audio_files)} audio files with valid labels")
    if labels:
        print(f"Label distribution: {np.bincount(labels, minlength=50)}")
    return audio_files, labels


def extract_mfcc(audio_path, sr=16000, n_mfcc=40):
    try:
        audio, _ = librosa.load(audio_path, sr=sr)
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
        return mfcc.T
    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        return None


def augment_audio(audio, sr=16000):
    noise = np.random.normal(0, 0.005, len(audio))
    audio = audio + noise
    audio = librosa.effects.pitch_shift(audio, sr=sr, n_steps=np.random.uniform(-2, 2))
    audio = librosa.effects.time_stretch(audio, rate=np.random.uniform(0.8, 1.2))
    return audio


def normalize_features(features):
    mean = np.mean(features, axis=0)
    std = np.std(features, axis=0) + 1e-6
    return (features - mean) / std


def spec_augment(mfcc, freq_mask_param=5, time_mask_param=10):
    augmented = np.copy(mfcc)
    num_freqs = augmented.shape[1]
    f = np.random.randint(0, freq_mask_param)
    f0 = np.random.randint(0, num_freqs - f)
    augmented[:, f0:f0 + f] = 0
    num_frames = augmented.shape[0]
    t = np.random.randint(0, time_mask_param)
    t0 = np.random.randint(0, num_frames - t)
    augmented[t0:t0 + t, :] = 0
    return augmented


def preprocess_data(data_dir, output_dir, sr=16000, n_mfcc=40, max_len=100):
    audio_files, labels = load_esc50(data_dir)
    if not audio_files:
        raise ValueError("No audio files found. Check directory path and file extensions.")

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    folds = []

    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(audio_files)):
        print(f"Processing fold {fold_idx + 1}")
        X_train_files, X_test_files = np.array(audio_files)[train_idx], np.array(audio_files)[test_idx]
        y_train, y_test = np.array(labels)[train_idx], np.array(labels)[test_idx]

        X_train_orig, X_train_aug = [], []
        for audio_path in X_train_files:
            mfcc = extract_mfcc(audio_path, sr, n_mfcc)
            if mfcc is None:
                continue
            mfcc = normalize_features(mfcc)
            aug_mfcc = spec_augment(mfcc)
            X_train_orig.append(mfcc)
            X_train_aug.append(aug_mfcc)

        X_test_orig, X_test_aug = [], []
        for audio_path in X_test_files:
            mfcc = extract_mfcc(audio_path, sr, n_mfcc)
            if mfcc is None:
                continue
            mfcc = normalize_features(mfcc)
            aug_mfcc = spec_augment(mfcc)
            X_test_orig.append(mfcc)
            X_test_aug.append(aug_mfcc)

        if not X_train_orig or not X_test_orig:
            print(f"Warning: Empty feature set in fold {fold_idx + 1}, skipping")
            continue

        X_train_orig = [x[:max_len] if x.shape[0] > max_len else np.pad(x, ((0, max_len - x.shape[0]), (0, 0))) for x in
                        X_train_orig]
        X_train_aug = [x[:max_len] if x.shape[0] > max_len else np.pad(x, ((0, max_len - x.shape[0]), (0, 0))) for x in
                       X_train_aug]
        X_test_orig = [x[:max_len] if x.shape[0] > max_len else np.pad(x, ((0, max_len - x.shape[0]), (0, 0))) for x in
                       X_test_orig]
        X_test_aug = [x[:max_len] if x.shape[0] > max_len else np.pad(x, ((0, max_len - x.shape[0]), (0, 0))) for x in
                      X_test_aug]

        X_train = np.stack([X_train_orig, X_train_aug], axis=1)
        X_test = np.stack([X_test_orig, X_test_aug], axis=1)

        fold_data = {
            'X_train': torch.tensor(X_train, dtype=torch.float32),
            'y_train': torch.tensor(y_train, dtype=torch.long),
            'X_test': torch.tensor(X_test, dtype=torch.float32),
            'y_test': torch.tensor(y_test, dtype=torch.long)
        }
        folds.append(fold_data)
        print(f"Fold {fold_idx + 1}: {len(y_train)} train samples, {len(y_test)} test samples")
        print(f"Fold {fold_idx + 1} y_train label counts: {np.bincount(y_train, minlength=50)}")

    if not folds:
        raise ValueError("No valid folds created. Check audio processing.")

    os.makedirs(output_dir, exist_ok=True)
    torch.save(folds, os.path.join(output_dir, 'folds.pt'))
    print(f"Saved {len(folds)} folds to {output_dir}/folds.pt")
    return folds


if __name__ == '__main__':
    preprocess_data('data/ESC-50/audio', 'data/preprocessed')