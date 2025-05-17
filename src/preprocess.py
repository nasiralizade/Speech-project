import os
import torch
import librosa
import numpy as np
import torchaudio
from sklearn.model_selection import train_test_split


def load_esc50(data_dir, sr=16000):
    audio_files, labels = [], []
    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.endswith('.wav'):
                audio_files.append(os.path.join(root, file))
                label = int(file.split('-')[-1].split('.')[0])  # ESC-50 label
                labels.append(label)
    return audio_files, labels


def extract_mfcc(audio_path, sr=16000, n_mfcc=40):
    audio, _ = librosa.load(audio_path, sr=sr)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    return mfcc.T  # Shape: (time_steps, n_mfcc)


def augment_audio(audio, sr=16000):
    # Random background noise
    noise = np.random.normal(0, 0.005, len(audio))
    audio = audio + noise
    # Pitch shift
    audio = librosa.effects.pitch_shift(audio, sr=sr, n_steps=np.random.uniform(-2, 2))
    # Time stretch
    audio = librosa.effects.time_stretch(audio, rate=np.random.uniform(0.8, 1.2))
    return audio
def normalize_features(features):
    mean = np.mean(features, axis=0)
    std = np.std(features, axis=0) + 1e-6  # prevent division by zero
    return (features - mean) / std


def preprocess_data(data_dir, output_dir, sr=16000, n_mfcc=40, max_len=100):
    audio_files, labels = load_esc50(data_dir)
    features = []

    for audio_path in audio_files:
        # Extract MFCC and normalize
        mfcc = extract_mfcc(audio_path, sr, n_mfcc)
        mfcc = normalize_features(mfcc)

        # Apply SpecAugment
        spec_augmented_mfcc = spec_augment(mfcc)
        features.append((mfcc, spec_augmented_mfcc))

    # Pad/truncate MFCCs to fixed length
    padded_features = []
    for mfcc, aug_mfcc in features:
        mfcc = mfcc[:max_len] if mfcc.shape[0] > max_len else np.pad(mfcc, ((0, max_len - mfcc.shape[0]), (0, 0)))
        aug_mfcc = aug_mfcc[:max_len] if aug_mfcc.shape[0] > max_len else np.pad(aug_mfcc,
                                                                                 ((0, max_len - aug_mfcc.shape[0]),
                                                                                  (0, 0)))
        padded_features.append(np.stack([mfcc, aug_mfcc]))

    # Prepare dataset splits
    X = np.array(padded_features)
    y = np.array(labels)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    # Save preprocessed data
    os.makedirs(output_dir, exist_ok=True)
    torch.save({'X_train': X_train, 'y_train': y_train, 'X_val': X_val, 'y_val': y_val,
                'X_test': X_test, 'y_test': y_test}, os.path.join(output_dir, 'preprocessed.pt'))

    return X_train, y_train, X_val, y_val, X_test, y_test





def spec_augment(mfcc, freq_mask_param=5, time_mask_param=10):
    augmented = np.copy(mfcc)

    # Frequency masking
    num_freqs = augmented.shape[1]
    f = np.random.randint(0, freq_mask_param)
    f0 = np.random.randint(0, num_freqs - f)
    augmented[:, f0:f0 + f] = 0

    # Time masking
    num_frames = augmented.shape[0]
    t = np.random.randint(0, time_mask_param)
    t0 = np.random.randint(0, num_frames - t)
    augmented[t0:t0 + t, :] = 0

    return augmented


if __name__ == '__main__':
    preprocess_data('/Users/nasiralizade/Project/Speech/data/ESC-50/audio', 'data/preprocessed')