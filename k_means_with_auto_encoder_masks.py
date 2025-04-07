#!/usr/bin/env python3
"""
SEM Image Segmentation with Autoencoder and K-means Clustering

This script processes SEM images of battery materials using SAM masks, an autoencoder for feature extraction, and K-means clustering for segmentation.

Segmentation Categories:
- Voids (black): Empty spaces between particles
- Carbon Black (gray): Conductive carbon additive particles
- LFP (green): Lithium iron phosphate active material
- NaCl (white): Sodium chloride particles

Usage:
    python script.py image.png masks.pkl

Output:
    A side-by-side comparison of the original and segmented images is displayed and saved as "segmentation_comparison.png".
"""

import os
import sys
import pickle
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.cluster import KMeans
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.optimizers import Adam


def load_image(image_path):
    """Load a grayscale SEM image."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Image not found at {image_path}")
    return img


def load_sam_masks(pkl_path):
    """Load SAM masks from a pickle file."""
    try:
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
        masks = [item["segmentation"] for item in data if isinstance(item, dict) and "segmentation" in item]
        print(f"Loaded {len(masks)} masks from {os.path.basename(pkl_path)}")
        return masks
    except Exception as e:
        print(f"Error loading pickle file: {e}")
        return None


def preprocess_image(img):
    """Enhance image quality using median filtering, CLAHE, and Gaussian blur."""
    img_median = cv2.medianBlur(img, 3)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_clahe = clahe.apply(img_median)
    img_enhanced = cv2.GaussianBlur(img_clahe, (3, 3), 0)
    return img_enhanced


def build_autoencoder(input_shape=(32, 32, 1)):
    """Build and compile a convolutional autoencoder for feature extraction."""
    inp = Input(shape=input_shape)

    # Encoder
    x = Conv2D(32, (3, 3), activation="relu", padding="same")(inp)
    x = MaxPooling2D((2, 2), padding="same")(x)
    x = Conv2D(16, (3, 3), activation="relu", padding="same")(x)
    encoded = MaxPooling2D((2, 2), padding="same")(x)

    # Decoder
    x = Conv2D(16, (3, 3), activation="relu", padding="same")(encoded)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), activation="relu", padding="same")(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(1, (3, 3), activation="sigmoid", padding="same")(x)

    autoencoder = Model(inp, decoded)
    autoencoder.compile(optimizer=Adam(), loss="mse")
    encoder = Model(inp, encoded)  # Extractor model

    return autoencoder, encoder


def extract_and_process_regions(image, masks, encoder, target_size=(32, 32)):
    """Extract masked regions, resize them, and encode features."""
    features = []
    valid_masks = []

    for mask in masks:
        coords = np.where(mask)
        if coords[0].size == 0:
            continue

        # Extract bounding box
        y_min, y_max = coords[0].min(), coords[0].max()
        x_min, x_max = coords[1].min(), coords[1].max()
        cropped = image[y_min:y_max + 1, x_min:x_max + 1]
        cropped[~mask[y_min:y_max + 1, x_min:x_max + 1]] = 0  # Apply mask

        # Pad to square and resize
        height, width = cropped.shape
        max_dim = max(height, width)
        padded = np.zeros((max_dim, max_dim), dtype=cropped.dtype)
        y_offset = (max_dim - height) // 2
        x_offset = (max_dim - width) // 2
        padded[y_offset:y_offset + height, x_offset:x_offset + width] = cropped
        resized = cv2.resize(padded, target_size)

        # Normalize and encode
        resized = np.expand_dims(resized.astype("float32") / 255.0, axis=(0, -1))
        latent_features = encoder.predict(resized).flatten()
        features.append(latent_features)
        valid_masks.append(mask)

    return np.array(features), valid_masks


def perform_clustering(features, n_clusters=4):
    """Apply K-means clustering to the extracted features."""
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    return kmeans.fit_predict(features)


def create_segmentation_map(image, masks, labels):
    """Generate a segmentation map based on clustering labels."""
    segmentation = np.zeros_like(image, dtype=np.uint8)
    for mask, label in zip(masks, labels):
        segmentation[mask.astype(bool)] = label
    return segmentation


def visualize_results(original_image, segmentation_map, output_path):
    """Display and save the original and segmented images side by side."""
    cmap_seg = ListedColormap(["black", "gray", "green", "white"])
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(original_image, cmap="gray")
    axes[0].set_title("Original SEM Image")
    axes[1].imshow(segmentation_map, cmap=cmap_seg, vmin=0, vmax=3)
    axes[1].set_title("Segmented Image")
    plt.savefig(output_path)
    plt.show()


def main():
    """Main function to run the SEM image segmentation pipeline."""
    if len(sys.argv) < 3:
        print("Usage: python script.py image.png masks.pkl")
        sys.exit(1)

    image_path, pkl_path = sys.argv[1], sys.argv[2]
    img = load_image(image_path)
    enhanced_img = preprocess_image(img)
    masks = load_sam_masks(pkl_path)
    if not masks:
        sys.exit(1)

    autoencoder, encoder = build_autoencoder()
    features, valid_masks = extract_and_process_regions(enhanced_img, masks, encoder)
    labels = perform_clustering(features)
    segmentation_map = create_segmentation_map(enhanced_img, valid_masks, labels)
    visualize_results(img, segmentation_map, "segmentation_comparison.png")


if __name__ == "__main__":
    main()
