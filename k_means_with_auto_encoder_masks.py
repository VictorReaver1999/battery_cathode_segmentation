#!/usr/bin/env python3
"""
SAM-Enhanced SEM Image Segmentation with Autoencoder and K-means Clustering

This script combines three powerful techniques for battery material segmentation:
1. SAM (Segment Anything Model) for precise particle boundary detection
2. Autoencoder for learning rich feature representations of particles
3. K-means clustering for unsupervised classification of materials

The segmentation classifies pixels into four categories:
- Voids (black): Empty spaces between particles
- Carbon Black (gray): Conductive carbon additive particles
- LFP (green): Lithium iron phosphate active material
- NaCl (white): Sodium chloride particles

The autoencoder learns to compress and reconstruct particle images, creating a
feature vector that captures important visual characteristics beyond simple
intensity values. K-means clustering then groups particles with similar features.

Requirements:
  - OpenCV (cv2)
  - NumPy
  - Matplotlib
  - scikit-learn
  - TensorFlow (tf.keras)

Usage example (from command line):
  python k_means_with_auto_encoder_masks.py image.png image.pkl
"""

import os
import sys
import pickle
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.cluster import KMeans

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.optimizers import Adam


def load_image(image_path):
    """
    Load SEM image as grayscale.

    Parameters:
    -----------
    image_path : str
        Path to the SEM image file

    Returns:
    --------
    ndarray
        Grayscale SEM image as a NumPy array

    Raises:
    -------
    FileNotFoundError
        If the specified image file cannot be found
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Image not found at {image_path}")
    return img


def load_sam_masks(pkl_path):
    """
    Load SAM masks from a pickle file.

    This function expects the pickle to contain a list of dictionaries,
    where each dictionary has a key 'segmentation' holding a boolean array.

    Parameters:
    -----------
    pkl_path : str
        Path to the pickle file containing SAM masks

    Returns:
    --------
    list
        List of boolean masks representing segmented regions

    Raises:
    -------
    Exception
        If there's an error loading or parsing the pickle file
    """
    try:
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
        masks = []
        for item in data:
            if isinstance(item, dict) and "segmentation" in item:
                masks.append(item["segmentation"])
        print(f"Loaded {len(masks)} masks from {os.path.basename(pkl_path)}")
        return masks
    except Exception as e:
        print(f"Error loading pickle file: {e}")
        return None


def preprocess_image(img):
    """
    Enhance image quality using median filtering, CLAHE and Gaussian blur.

    Parameters:
    -----------
    img : ndarray
        Input grayscale SEM image

    Returns:
    --------
    ndarray
        Enhanced image with improved contrast and reduced noise
    """
    img_median = cv2.medianBlur(img, 3)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_clahe = clahe.apply(img_median)
    img_enhanced = cv2.GaussianBlur(img_clahe, (3, 3), 0)
    return img_enhanced


def build_autoencoder(input_shape=(64, 64, 1)):
    """
    Build a simple convolutional autoencoder for feature extraction.

    The encoder portion compresses the input image to a lower-dimensional
    latent representation, which is used for feature extraction.

    Parameters:
    -----------
    input_shape : tuple
        Shape of the input images (height, width, channels)

    Returns:
    --------
    tuple (Model, Model)
        (autoencoder, encoder) - The full autoencoder and the encoder portion
    """
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

    # Create a separate model for the encoder
    encoder = Model(inp, encoded)

    return autoencoder, encoder


def train_autoencoder_on_patches(image, masks, autoencoder, target_size=(64, 64), epochs=10, batch_size=8):
    """
    Extract patches from the masked regions and train the autoencoder.

    This function:
    1. Extracts image patches from regions identified by SAM masks
    2. Resizes each patch to a consistent target size
    3. Trains the autoencoder to reconstruct these patches

    Parameters:
    -----------
    image : ndarray
        Input grayscale SEM image
    masks : list
        List of boolean masks from SAM segmentation
    autoencoder : Model
        Keras autoencoder model to be trained
    target_size : tuple, optional
        Size to which patches will be resized (width, height)
    epochs : int, optional
        Number of training epochs (default: 10)
    batch_size : int, optional
        Batch size for training (default: 8)

    Returns:
    --------
    Model
        Trained autoencoder model
    """
    patches = []
    for mask in masks:
        coords = np.where(mask)
        if coords[0].size == 0:
            continue
        y_min, y_max = coords[0].min(), coords[0].max()
        x_min, x_max = coords[1].min(), coords[1].max()
        patch = image[y_min: y_max + 1, x_min: x_max + 1]
        # Resize to target_size
        patch_resized = cv2.resize(patch, target_size)
        patches.append(patch_resized)
    if len(patches) == 0:
        print("No valid patches found for training autoencoder")
        return autoencoder
    patches = np.array(patches, dtype="float32") / 255.0
    patches = np.expand_dims(patches, axis=-1)  # shape: (n, H, W, 1)
    print(f"Training autoencoder on {patches.shape[0]} patches ...")
    autoencoder.fit(patches, patches, epochs=epochs, batch_size=batch_size, verbose=1)
    return autoencoder


def extract_features_from_mask(image, mask, encoder, target_size=(64, 64)):
    """
    Extract features from a specific masked region using the autoencoder.

    For a given SAM mask region, this function computes:
    - Basic intensity statistics (mean, std, median)
    - A latent feature vector from the autoencoder encoder

    Parameters:
    -----------
    image : ndarray
        Input grayscale SEM image
    mask : ndarray
        Boolean mask identifying a region of interest
    encoder : Model
        Trained encoder part of the autoencoder
    target_size : tuple, optional
        Size to which patches will be resized (width, height)

    Returns:
    --------
    tuple (ndarray, tuple) or (None, None)
        (feature_vector, bbox) - Feature vector and bounding box of the region
        Returns (None, None) if the mask is empty
    """
    coords = np.where(mask)
    if coords[0].size == 0:
        return None, None
    y_min, y_max = coords[0].min(), coords[0].max()
    x_min, x_max = coords[1].min(), coords[1].max()
    patch = image[y_min: y_max + 1, x_min: x_max + 1]
    if patch.size == 0:
        return None, None

    # Basic intensity features
    mean_intensity = np.mean(patch)
    std_intensity = np.std(patch)
    median_intensity = np.median(patch)

    # Resize patch to match encoder input and normalize
    patch_resized = cv2.resize(patch, target_size)
    patch_resized = patch_resized.astype("float32") / 255.0
    patch_resized = np.expand_dims(patch_resized, axis=-1)  # shape: (H, W, 1)
    patch_resized = np.expand_dims(patch_resized, axis=0)  # shape: (1, H, W, 1)

    # Get latent features
    latent = encoder.predict(patch_resized)
    latent_features = latent.flatten()

    # Concatenate features: [mean, std, median] + latent vector
    feature_vector = np.concatenate(([mean_intensity, std_intensity, median_intensity], latent_features))
    bbox = (x_min, y_min, x_max, y_max)
    return feature_vector, bbox


def main():
    """
    Main function for SEM image segmentation with autoencoder and K-means.

    This function:
    1. Loads the SEM image and SAM masks from command-line arguments
    2. Preprocesses the image with CLAHE and Gaussian blur
    3. Trains an autoencoder using particle patches
    4. Extracts feature vectors for each particle
    5. Applies K-means clustering to classify particles
    6. Creates a segmentation map based on the classification
    7. Visualizes and saves the results

    Command-line arguments:
    - image.png: Path to the SEM image
    - image.pkl: Path to the SAM masks pickle file
    """
    if len(sys.argv) < 3:
        print("Usage: python k_means_with_auto_encoder_masks.py image.png masks.pkl")
        sys.exit(1)

    image_path = sys.argv[1]
    pkl_path = sys.argv[2]
    output_dir = "autoencoder_results_new"
    os.makedirs(output_dir, exist_ok=True)

    # Step 1: Load and preprocess image
    print("Loading image ...")
    img = load_image(image_path)
    enhanced_img = preprocess_image(img)

    # Step 2: Load SAM masks
    masks = load_sam_masks(pkl_path)
    if masks is None or len(masks) == 0:
        print("No valid SAM masks loaded!")
        sys.exit(1)

    # (Optional) You might want to remove regions (e.g. metadata) here.
    # For simplicity, we assume the image has only useful data.

    # Step 3: Create and (optionally) train the autoencoder on patches from the image regions.
    ae_input_shape = (64, 64, 1)
    autoencoder, encoder = build_autoencoder(input_shape=ae_input_shape)
    autoencoder = train_autoencoder_on_patches(enhanced_img, masks, autoencoder, target_size=(64, 64),
                                               epochs=10, batch_size=8)

    # Step 4: Extract features for every valid mask
    feature_list = []
    valid_masks = []
    bbox_list = []
    for mask in masks:
        feat, bbox = extract_features_from_mask(enhanced_img, mask, encoder, target_size=(64, 64))
        if feat is not None:
            feature_list.append(feat)
            valid_masks.append(mask)
            bbox_list.append(bbox)
    if len(feature_list) == 0:
        print("No features extracted!")
        sys.exit(1)
    features = np.array(feature_list)
    print(f"Extracted features from {features.shape[0]} regions.")

    # Step 5: Run clustering (k-means) on the feature vectors; choose 4 clusters.
    kmeans = KMeans(n_clusters=4, random_state=42)
    cluster_labels = kmeans.fit_predict(features)
    print("Clustering complete.")

    # Step 6: Map clusters to physical material classes using average intensity.
    # We expect: lowest intensity = Voids (class 0), then Carbon Black (1), LFP (2), and highest = NaCl (3).
    cluster_intensity = {}
    for cl in np.unique(cluster_labels):
        indices = np.where(cluster_labels == cl)[0]
        # The first feature (index 0) is the mean intensity.
        avg_int = np.mean(features[indices, 0])
        cluster_intensity[cl] = avg_int
    # Sort clusters by mean intensity (lowest first)
    sorted_clusters = sorted(cluster_intensity.items(), key=lambda x: x[1])
    mapping = {}
    # According to the expected ordering:
    # 0: Voids, 1: Carbon Black, 2: LFP, 3: NaCl.
    for i, (cl, _) in enumerate(sorted_clusters):
        mapping[cl] = i
    print("Cluster mapping (cluster_label -> material class):", mapping)
    material_labels = ["Voids", "Carbon Black", "LFP", "NaCl"]

    # Step 7: Create final segmentation mask
    segmentation = np.zeros_like(enhanced_img, dtype=np.uint8)
    for i, mask in enumerate(valid_masks):
        # Determine the final material label after mapping
        final_label = mapping[cluster_labels[i]]
        segmentation[mask.astype(bool)] = final_label

    # (Optionally, for regions where no valid mask exists, you could add additional segmentation logic.)

    # Step 8: Visualize results
    cmap_seg = ListedColormap(["black", "gray", "green", "white"])
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(enhanced_img, cmap="gray")
    axes[0].set_title("Enhanced SEM Image")
    axes[0].axis("off")
    axes[1].imshow(segmentation, cmap=cmap_seg, vmin=0, vmax=3)
    axes[1].set_title("Segmented (Autoencoder + Clustering)")
    axes[1].axis("off")
    plt.suptitle(os.path.basename(image_path))
    plt.tight_layout()
    plt.show()

    # Optionally, save the segmentation result
    seg_out_path = os.path.join(output_dir, "segmentation_result.png")
    cv2.imwrite(seg_out_path, segmentation)
    print(f"Segmentation result saved to {seg_out_path}")


if __name__ == "__main__":
    main()