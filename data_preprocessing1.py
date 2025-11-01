import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
from pathlib import Path
import os
import matplotlib.pyplot as plt

#10 core commands
LABELS = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"]

# My Path to the downloaded dataset
DATA_PATH = Path("./speech_commands_data")

# Spectrogram parameters
SAMPLE_RATE = 16000
N_FFT = 1024
HOP_LENGTH = 256


def preprocess_audio(filepath, label):
    audio, sr = tf.audio.decode_wav(tf.io.read_file(filepath), desired_channels=1)
    audio = tf.squeeze(audio, axis=-1)
    audio = tf.cast(audio, tf.float32)

    # Pad or truncate audio to 1 second (16000 samples)
    audio_len = tf.shape(audio)[0]
    if audio_len < SAMPLE_RATE:
        padding = tf.zeros(SAMPLE_RATE - audio_len, dtype=tf.float32)
        audio = tf.concat([audio, padding], axis=0)
    elif audio_len > SAMPLE_RATE:
        audio = audio[:SAMPLE_RATE]

    # Computing STFT
    stft = tf.signal.stft(audio, frame_length=N_FFT, frame_step=HOP_LENGTH)
    spectrogram = tf.abs(stft)

    # Normalize spectrogram
    spectrogram = tf.math.log(spectrogram + 1e-6)  # Adding small epsilon for log stability
    mean = tf.reduce_mean(spectrogram)
    std = tf.math.reduce_std(spectrogram)
    spectrogram = (spectrogram - mean) / (std + 1e-6)  # Adding small epsilon for std stability

    # Convert label string to integer ID
    label_id = tf.argmax(tf.cast(tf.equal(LABELS, tf.cast(label, tf.string)), tf.int32))

    return spectrogram, label_id


def create_dataset_from_filepaths(batch_size=32):
    all_audio_paths = []
    all_audio_labels = []

    print("Collecting file paths...")
    for label_name in LABELS:
        label_dir = DATA_PATH / label_name
        if not label_dir.exists():
            print(f"Warning: Directory for label \'{label_name}\' not found at {label_dir}")
            continue
        for filepath in label_dir.glob("*.wav"):
            all_audio_paths.append(str(filepath))
            all_audio_labels.append(label_name)

    if not all_audio_paths:
        print("No audio files found. Please ensure the dataset is downloaded and extracted correctly.")
        return None, None, None, None

    # TensorFlow Dataset from file paths and labels
    dataset = tf.data.Dataset.from_tensor_slices((all_audio_paths, all_audio_labels))

    # Shuffle and split the dataset
    dataset = dataset.shuffle(len(all_audio_paths), reshuffle_each_iteration=False)

    train_size = int(0.8 * len(all_audio_paths))
    val_size = int(0.1 * len(all_audio_paths))

    train_ds = dataset.take(train_size)
    val_ds = dataset.skip(train_size).take(val_size)
    test_ds = dataset.skip(train_size + val_size)

    AUTOTUNE = tf.data.AUTOTUNE

    # Map the preprocessing function to the datasets
    train_ds = train_ds.map(preprocess_audio, num_parallel_calls=AUTOTUNE)
    val_ds = val_ds.map(preprocess_audio, num_parallel_calls=AUTOTUNE)
    test_ds = test_ds.map(preprocess_audio, num_parallel_calls=AUTOTUNE)

    # I explicitly defined the padded_shapes for batching to maintain fixed time dimension
    # sequence_length = (SAMPLE_RATE - N_FFT) // HOP_LENGTH + 1 = (16000 - 1024) // 256 + 1 = 59
    # features = N_FFT // 2 + 1 = 1024 // 2 + 1 = 513
    padded_shapes = ([59, 513], [])  # Spectrogram shape (time_frames, features), and empty shape for label

    train_ds = train_ds.padded_batch(batch_size, padded_shapes=padded_shapes, drop_remainder=True).prefetch(AUTOTUNE)
    val_ds = val_ds.padded_batch(batch_size, padded_shapes=padded_shapes, drop_remainder=True).prefetch(AUTOTUNE)
    test_ds = test_ds.padded_batch(batch_size, padded_shapes=padded_shapes, drop_remainder=True).prefetch(AUTOTUNE)

    for spectrograms, labels in train_ds.take(1):
        spec = spectrograms[0].numpy()
        label_id = labels[0].numpy()

        plt.figure(figsize=(10, 4))
        plt.imshow(spec.T, aspect='auto', origin='lower', cmap='viridis')
        plt.title(f"Spectrogram - Label: {LABELS[label_id]} ({label_id})")
        plt.xlabel("Time frames")
        plt.ylabel("Frequency bins")
        plt.colorbar(label="Log Magnitude")
        plt.tight_layout()
        plt.show()

    print(
        f"Dataset created: Train size={train_size}, Val size={val_size}, Test size={len(all_audio_paths) - train_size - val_size}")
    print(f"Labels: {LABELS}")
    return train_ds, val_ds, test_ds, {label: i for i, label in enumerate(LABELS)}


if __name__ == '__main__':
    train_ds, val_ds, test_ds, label_map = create_dataset_from_filepaths()
    if train_ds:
        print("Successfully created datasets.")
        for spec, label in train_ds.take(1):
            print(f"Example spectrogram shape: {spec.shape}, label: {label.numpy()}")
    else:
        print("Failed to create datasets.")
