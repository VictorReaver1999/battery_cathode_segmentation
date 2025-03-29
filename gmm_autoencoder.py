#!/usr/bin/env python3

"""
SAM-Enhanced Battery Material Segmentation with Gaussian Mixture Models

This module combines Segment Anything Model (SAM) masks with statistical Gaussian Mixture Model (GMM)
classification to perform sophisticated material segmentation in SEM images of battery materials.
The pipeline includes automatic metadata removal, feature extraction using both traditional image
processing and deep learning autoencoders, and probabilistic classification with GMMs.

Key Features:
- Automatic detection and removal of microscope metadata bars
- Integration of SAM-generated masks for precise particle boundaries
- Hybrid feature extraction combining traditional and deep learning approaches
- Probabilistic classification with adjustable GMM components
- Comprehensive post-processing and visualization capabilities
- CSV export for quantitative analysis

Classes:
    None

Functions:
    load_sem_image: Load SEM image from file
    detect_metadata_bar: Detect and mask microscope metadata regions
    preprocess_image: Enhance image quality through filtering and contrast adjustment
    load_sam_masks: Load segmentation masks from SAM output
    build_autoencoder: Construct convolutional autoencoder for feature learning
    extract_particle_patches: Extract image regions for autoencoder training
    extract_mask_features: Calculate traditional image features from mask regions
    train_autoencoder: Train deep feature extractor on particle patches
    extract_autoencoder_features: Generate deep features using trained encoder
    train_gmm: Train Gaussian Mixture Model on extracted features
    classify_with_gmm: Classify particles using trained GMM
    segment_with_gmm_sam: Main segmentation pipeline
    visualize_segmentation: Generate comparative visualization of results
    calculate_material_stats: Calculate quantitative material distribution statistics
    export_to_csv: Export segmentation results to CSV format
    main: Command-line interface handler

Dependencies:
    OpenCV, scikit-learn, TensorFlow, matplotlib, scipy, scikit-image
"""

import os
import sys
import pickle
import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import pandas as pd
from scipy import ndimage
from skimage import filters, measure, morphology
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras import layers, models


def load_sem_image(image_path):
    """
    Load and preprocess a scanning electron microscope (SEM) image.

    Args:
        image_path (str): Path to the SEM image file (PNG format recommended)

    Returns:
        numpy.ndarray: Grayscale image as 2D numpy array with values in [0, 255]

    Raises:
        FileNotFoundError: If specified image file cannot be loaded
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Image not found at {image_path}")
    return img


def detect_metadata_bar(img, threshold_percent=0.1):
    """
    Detect and mask microscope metadata bar at image bottom.

        Uses edge detection and texture analysis to identify metadata regions containing
        microscope parameters and measurement information.

        Args:
            img (numpy.ndarray): Input grayscale image
            threshold_percent (float, optional): Percentage of image height to analyze
                for metadata detection. Defaults to 0.1 (10% of image height).

        Returns:
            numpy.ndarray: Boolean mask where True indicates valid image regions
    """
    h, w = img.shape

    # Start by assuming the metadata bar is at the bottom of the image
    metadata_mask = np.ones_like(img, dtype=bool)

    # Check for horizontal lines that might indicate metadata bar
    # Use horizontal gradient to detect abrupt changes
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)

    # Look at the bottom part of the image
    bottom_region_height = int(h * threshold_percent)
    bottom_region = img[-bottom_region_height:]

    # Check for text-like features in the bottom region
    horizontal_sum = np.sum(np.abs(sobelx[-bottom_region_height:]), axis=1)
    vertical_sum = np.sum(np.abs(sobely[-bottom_region_height:]), axis=1)

    # Normalize the sums
    horizontal_sum = horizontal_sum / w
    vertical_sum = vertical_sum / w

    # Find rows with high horizontal and vertical gradients (typical for text)
    text_threshold = np.max(horizontal_sum) * 0.3
    potential_text_rows = np.where(horizontal_sum > text_threshold)[0]

    # If we found potential text rows, set the mask accordingly
    if len(potential_text_rows) > 0:
        # Find the first row with significant text features from the bottom
        first_text_row = potential_text_rows[0]
        metadata_height = min(80, bottom_region_height)  # At least 80 pixels or what we checked

        # Set the bottom part as metadata (mask = False)
        metadata_mask[-metadata_height:, :] = False

        print(f"Detected metadata bar of height {metadata_height} pixels")
    else:
        # Fallback: use a fixed height for metadata bar
        metadata_mask[-80:, :] = False
        print(f"Using default metadata height of 80 pixels")

    return metadata_mask


def preprocess_image(img):
    """Enhance image quality through multi-stage processing pipeline.

    Processing steps:
    1. Median filtering for noise reduction
    2. CLAHE (Contrast Limited Adaptive Histogram Equalization) for local contrast enhancement
    3. Gaussian blurring for high-frequency noise reduction

    Args:
        img (numpy.ndarray): Raw grayscale SEM image

    Returns:
        numpy.ndarray: Processed image with enhanced features
    """
    # Apply median filter to reduce noise
    img_median = cv2.medianBlur(img, 3)

    # Apply CLAHE for better contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_clahe = clahe.apply(img_median)

    # Apply slight Gaussian blur
    img_enhanced = cv2.GaussianBlur(img_clahe, (3, 3), 0)

    return img_enhanced


def load_sam_masks(pickle_path):
    """
    Load segmentation masks generated by Segment Anything Model (SAM).

    Args:
        pickle_path (str): Path to SAM output pickle file containing mask data

    Returns:
        list[numpy.ndarray] or None: List of boolean masks or None if loading fails

    Raises:
        IOError: If pickle file structure is invalid or cannot be read
    """
    try:
        with open(pickle_path, 'rb') as f:
            data = pickle.load(f)

        # Extract segmentation masks
        masks = []
        for item in data:
            if isinstance(item, dict) and 'segmentation' in item:
                masks.append(item['segmentation'])

        print(f"Loaded {len(masks)} masks from SAM")
        return masks

    except Exception as e:
        print(f"Error loading pickle file: {e}")
        return None


def build_autoencoder(input_shape=(64, 64, 1)):
    """
    Construct convolutional autoencoder with residual connections.

    Architecture Details:
    - Encoder: 3 convolutional blocks with residual connections and max pooling
    - Latent space: 128-dimensional dense representation
    - Decoder: Transposed convolutions for image reconstruction

    Args:
        input_shape (tuple, optional): Input tensor shape. Defaults to (64, 64, 1).

    Returns:
        tuple: (autoencoder_model, encoder_model) Keras model instances
    """
    # Encoder
    inputs = layers.Input(shape=input_shape)

    # First block with residual connection
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)

    # Second block with residual connection
    skip1 = x
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.add([x, layers.Conv2D(64, (1, 1))(skip1)])  # Residual connection
    x = layers.MaxPooling2D((2, 2), padding='same')(x)

    # Third block
    skip2 = x
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.add([x, layers.Conv2D(128, (1, 1))(skip2)])  # Residual connection
    x = layers.MaxPooling2D((2, 2), padding='same')(x)

    # Latent representation
    x = layers.Flatten()(x)
    latent = layers.Dense(128, activation='relu')(x)

    # Decoder (for training only)
    x = layers.Dense(8 * 8 * 128, activation='relu')(latent)
    x = layers.Reshape((8, 8, 128))(x)

    x = layers.Conv2DTranspose(128, (3, 3), strides=2, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv2DTranspose(64, (3, 3), strides=2, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv2DTranspose(32, (3, 3), strides=2, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)

    outputs = layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

    # Define models for autoencoder and encoder
    autoencoder = models.Model(inputs, outputs)
    encoder = models.Model(inputs, latent)

    # Compile autoencoder
    autoencoder.compile(optimizer='adam', loss='mse')

    return autoencoder, encoder


def extract_particle_patches(img, masks, patch_size=64, metadata_mask=None):
    """
    Extract fixed-size image patches centered on detected particles.

    Args:
        img (numpy.ndarray): Source grayscale image
        masks (list[numpy.ndarray]): List of boolean masks from SAM
        patch_size (int, optional): Output patch size. Defaults to 64.
        metadata_mask (numpy.ndarray, optional): Region mask to exclude. Defaults to None.

    Returns:
        numpy.ndarray: Array of extracted patches shaped (N, patch_size, patch_size)
    """
    patches = []

    for mask in masks:
        # Skip masks with no pixels or that overlap with metadata
        if np.sum(mask) == 0:
            continue

        if metadata_mask is not None:
            # Skip if mask significantly overlaps with metadata region
            if np.sum(mask & (~metadata_mask)) > 0.2 * np.sum(mask):
                continue

        # Get particle bounding box
        y_indices, x_indices = np.where(mask)
        if len(y_indices) == 0:
            continue

        y_min, y_max = np.min(y_indices), np.max(y_indices)
        x_min, x_max = np.min(x_indices), np.max(x_indices)

        # Calculate center
        center_y = (y_min + y_max) // 2
        center_x = (x_min + x_max) // 2

        # Define patch boundaries
        half_size = patch_size // 2
        y1 = max(0, center_y - half_size)
        y2 = min(img.shape[0], center_y + half_size)
        x1 = max(0, center_x - half_size)
        x2 = min(img.shape[1], center_x + half_size)

        # Skip patches that overlap with metadata
        if metadata_mask is not None and not np.all(metadata_mask[y1:y2, x1:x2]):
            continue

        # Extract the patch
        patch = img[y1:y2, x1:x2]

        # Resize if necessary
        if patch.shape[0] != patch_size or patch.shape[1] != patch_size:
            patch = cv2.resize(patch, (patch_size, patch_size))

        patches.append(patch)

    return np.array(patches)


def extract_mask_features(img, enhanced_img, mask, metadata_mask):
    """
    Calculate comprehensive feature set for a masked region.

    Extracted features include:
    - Intensity statistics (mean, std, median, min/max)
    - Texture features (GLCM-based complexity)
    - Shape features (area, compactness)
    - Enhanced image characteristics

    Args:
        img (numpy.ndarray): Original grayscale image
        enhanced_img (numpy.ndarray): Preprocessed image
        mask (numpy.ndarray): Boolean mask defining region of interest
        metadata_mask (numpy.ndarray): Valid region mask

    Returns:
        dict or None: Feature dictionary or None for invalid regions
    """
    # Apply the metadata mask to the particle mask
    valid_mask = mask & metadata_mask

    # Skip empty masks or masks with too few pixels
    if np.sum(valid_mask) < 50:
        return None

    # Extract region of interest from original and enhanced images
    masked_img = img[valid_mask]
    masked_enhanced = enhanced_img[valid_mask]

    # Calculate intensity features
    mean_intensity = np.mean(masked_img)
    std_intensity = np.std(masked_img)
    median_intensity = np.median(masked_img)
    min_intensity = np.min(masked_img)
    max_intensity = np.max(masked_img)

    # Enhanced image features
    enhanced_mean = np.mean(masked_enhanced)
    enhanced_std = np.std(masked_enhanced)

    # Calculate texture features using GLCM
    texture_complexity = std_intensity / (mean_intensity + 1e-10)

    # Calculate shape features
    area = np.sum(valid_mask)

    # Create contour for perimeter calculation
    contour = valid_mask.astype(np.uint8) * 255
    contours, _ = cv2.findContours(contour, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 0:
        perimeter = cv2.arcLength(contours[0], True)
    else:
        perimeter = 0

    # Calculate compactness (circularity)
    compactness = 4 * np.pi * area / (perimeter * perimeter + 1e-10) if perimeter > 0 else 0

    # Return feature dictionary
    features = {
        'mean_intensity': mean_intensity,
        'std_intensity': std_intensity,
        'median_intensity': median_intensity,
        'min_intensity': min_intensity,
        'max_intensity': max_intensity,
        'enhanced_mean': enhanced_mean,
        'enhanced_std': enhanced_std,
        'texture_complexity': texture_complexity,
        'area': area,
        'compactness': compactness
    }

    return features


def train_autoencoder(img, enhanced_img, masks, metadata_mask, epochs=50):
    """
    Train convolutional autoencoder on particle image patches.

    Args:
        img (numpy.ndarray): Original grayscale image
        enhanced_img (numpy.ndarray): Preprocessed image
        masks (list[numpy.ndarray]): SAM-generated masks
        metadata_mask (numpy.ndarray): Valid region mask
        epochs (int, optional): Training iterations. Defaults to 50.

    Returns:
        tensorflow.keras.Model or None: Trained encoder model or None if insufficient data
    """
    # Extract patches for training
    print("Extracting patches for autoencoder training...")
    patch_size = 64
    patches = extract_particle_patches(enhanced_img, masks, patch_size, metadata_mask)

    if len(patches) < 10:
        print("Warning: Not enough particles detected for autoencoder training")
        return None

    # Normalize patches
    patches_norm = patches.astype('float32') / 255.0
    patches_norm = patches_norm.reshape(-1, patch_size, patch_size, 1)

    # Build and train the autoencoder
    print(f"Training autoencoder on {len(patches_norm)} patches for {epochs} epochs")
    autoencoder, encoder = build_autoencoder(input_shape=(patch_size, patch_size, 1))

    # Train the autoencoder
    autoencoder.fit(
        patches_norm, patches_norm,
        epochs=epochs,
        batch_size=16,
        shuffle=True,
        validation_split=0.2,
        verbose=1
    )

    return encoder


def extract_autoencoder_features(img, mask, encoder, metadata_mask, patch_size=64):
    """
    Generate deep feature vector for a masked region using trained encoder.

    Args:
        img (numpy.ndarray): Preprocessed input image
        mask (numpy.ndarray): Region of interest mask
        encoder (tensorflow.keras.Model): Trained feature extractor
        metadata_mask (numpy.ndarray): Valid region mask
        patch_size (int, optional): Input size for encoder. Defaults to 64.

    Returns:
        numpy.ndarray or None: 128-dimensional feature vector or None for invalid regions
    """
    # Apply metadata mask
    valid_mask = mask & metadata_mask

    # Skip if mask is too small
    if np.sum(valid_mask) < 50:
        return None

    # Get mask center
    y_indices, x_indices = np.where(valid_mask)
    y_min, y_max = np.min(y_indices), np.max(y_indices)
    x_min, x_max = np.min(x_indices), np.max(x_indices)

    center_y = (y_min + y_max) // 2
    center_x = (x_min + x_max) // 2

    # Define patch boundaries
    half_size = patch_size // 2
    y1 = max(0, center_y - half_size)
    y2 = min(img.shape[0], center_y + half_size)
    x1 = max(0, center_x - half_size)
    x2 = min(img.shape[1], center_x + half_size)

    # Extract patch
    patch = img[y1:y2, x1:x2].astype('float32') / 255.0

    # Resize if necessary
    if patch.shape[0] != patch_size or patch.shape[1] != patch_size:
        patch = cv2.resize(patch, (patch_size, patch_size))

    # Reshape for encoder
    patch = patch.reshape(1, patch_size, patch_size, 1)

    # Get feature vector
    features = encoder.predict(patch, verbose=0)[0]

    return features


def train_gmm(mask_features, n_components=4, covariance_type='full'):

    """
    Train Gaussian Mixture Model on extracted particle features.

    Args:
        mask_features (list[dict]): List of feature dictionaries from extract_mask_features
        n_components (int, optional): Number of GMM components. Defaults to 4.
        covariance_type (str, optional): Covariance matrix type. Defaults to 'full'.

    Returns:
        tuple or None: (GMM model, StandardScaler, cluster mapping) or None on failure
    """
    # Extract the relevant features for GMM training \
    feature_matrix = []

    # For each mask with features
    for features in mask_features:
        if features is not None:
            # Create feature vector (focusing on intensity-related features)
            feature_vector = [
                features['mean_intensity'],
                features['enhanced_mean'],
                features['std_intensity'],
                features['texture_complexity']
            ]
            feature_matrix.append(feature_vector)

    # Convert to numpy array
    feature_matrix = np.array(feature_matrix)

    # Check if we have enough data
    if len(feature_matrix) < n_components:
        print(f"Warning: Only {len(feature_matrix)} valid particles for GMM training, needed {n_components}")
        if len(feature_matrix) == 0:
            return None
        n_components = min(len(feature_matrix), 3)  # Reduced number of components

    # Normalize features
    scaler = StandardScaler()
    feature_matrix_scaled = scaler.fit_transform(feature_matrix)

    # Train GMM
    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type=covariance_type,
        random_state=42,
        max_iter=200
    )
    gmm.fit(feature_matrix_scaled)

    # Order the components by the mean intensity of their center
    # This helps align GMM clusters with the expected materials
    centers = gmm.means_

    # Transform the centers back to original scale for intensity comparison
    centers_original = scaler.inverse_transform(centers)

    # Sort by the first feature (mean_intensity)
    sorted_idx = np.argsort(centers_original[:, 0])

    # Map cluster indices to material classes based on intensity
    # 0: void (not used here since voids are defined by thresholding)
    # 1: carbon black (darkest)
    # 2: LFP (medium)
    # 3: NaCl (brightest)
    cluster_to_material = {}

    if n_components == 3:
        # We have 3 clusters: carbon black, LFP, NaCl
        for i, idx in enumerate(sorted_idx):
            cluster_to_material[idx] = i + 1  # Start from class 1 (carbon black)
    else:
        # Handle case with 4 clusters
        # The darkest cluster might be voids or carbon black
        # Check the mean intensity of the darkest cluster
        darkest_idx = sorted_idx[0]
        darkest_intensity = centers_original[darkest_idx, 0]

        if darkest_intensity < 40:  # If very dark, it's void
            # Skip the darkest cluster and assign 1, 2, 3 to the rest
            cluster_to_material[darkest_idx] = 0  # void
            for i, idx in enumerate(sorted_idx[1:4]):
                cluster_to_material[idx] = i + 1
        else:
            # All clusters are materials
            for i, idx in enumerate(sorted_idx):
                if i < 3:  # Only use first 3 clusters (we expect 3 materials)
                    cluster_to_material[idx] = i + 1

    # Store the scaler with the GMM for later use
    return gmm, scaler, cluster_to_material


def classify_with_gmm(particle_features, gmm, scaler, cluster_to_material):
    """Classify particle features using trained GMM model.

    Args:
        particle_features (dict): Feature dictionary from extract_mask_features
        gmm (GaussianMixture): Trained GMM instance
        scaler (StandardScaler): Feature normalizer
        cluster_to_material (dict): Cluster to material class mapping

    Returns:
        tuple or None: (material_class, confidence) or None for invalid input
    """
    if particle_features is None:
        return None

    # Create feature vector for GMM classification
    feature_vector = [
        particle_features['mean_intensity'],
        particle_features['enhanced_mean'],
        particle_features['std_intensity'],
        particle_features['texture_complexity']
    ]

    # Reshape and scale
    feature_vector = np.array(feature_vector).reshape(1, -1)
    feature_vector_scaled = scaler.transform(feature_vector)

    # Get GMM probability and prediction
    prob = gmm.predict_proba(feature_vector_scaled)[0]
    cluster = np.argmax(prob)

    # Convert to material class
    if cluster in cluster_to_material:
        material = cluster_to_material[cluster]
        confidence = prob[cluster]
        return material, confidence
    else:
        # Fallback: classify based on intensity only
        mean_intensity = particle_features['mean_intensity']

        if mean_intensity < 60:
            return 1, 0.8  # Carbon Black
        elif mean_intensity < 150:
            return 2, 0.8  # LFP
        else:
            return 3, 0.8  # NaCl


def segment_with_gmm_sam(image_path, masks_path=None, output_dir=None,
                         void_threshold=10, epochs=50, gmm_components=4):
    """
    Main segmentation pipeline integrating SAM masks and GMM classification.

    Processing Steps:
    1. Image loading and metadata detection
    2. Feature extraction (traditional + deep learning)
    3. GMM training and classification
    4. Post-processing and visualization
    5. Results export

    Args:
        image_path (str): Path to SEM image
        masks_path (str, optional): Path to SAM masks pickle file
        output_dir (str, optional): Output directory for results
        void_threshold (int, optional): Intensity threshold for void detection. Defaults to 10.
        epochs (int, optional): Autoencoder training epochs. Defaults to 50.
        gmm_components (int, optional): Number of GMM components. Defaults to 4.

    Returns:
        tuple: (segmentation_map, statistics_dict)
    """
    # If masks_path is not provided, infer it from image_path
    if masks_path is None:
        base_name = os.path.splitext(image_path)[0]
        masks_path = f"{base_name}.pkl"

    print(f"Processing {os.path.basename(image_path)}")
    print(f"Using mask file: {os.path.basename(masks_path)}")
    print(f"Parameters: Void threshold={void_threshold}, GMM components={gmm_components}")

    # Create output directory if needed
    if output_dir is not None and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Step 1: Load image and masks
    img = load_sem_image(image_path)

    # Detect and mask the metadata bar
    metadata_mask = detect_metadata_bar(img)

    # Enhance image
    enhanced_img = preprocess_image(img)

    # Load SAM masks
    masks = load_sam_masks(masks_path)

    if not masks:
        print(f"Error: No valid masks found in {masks_path}")
        return None, None

    # Step 2: Prepare data for GMM
    print("Extracting features from particles...")

    # First, create a combined particle mask from all SAM masks
    particle_mask = np.zeros_like(img, dtype=bool)
    for mask in masks:
        particle_mask = particle_mask | mask

    # Apply metadata mask to particle mask
    particle_mask = particle_mask & metadata_mask

    # Step 3: Train autoencoder (optional but recommended)
    use_autoencoder = (epochs > 0)
    encoder = None

    if use_autoencoder:
        encoder = train_autoencoder(img, enhanced_img, masks, metadata_mask, epochs)

    # Step 4: Extract features from each mask
    mask_features = []
    valid_masks = []

    for mask in masks:
        # Extract basic features
        features = extract_mask_features(img, enhanced_img, mask, metadata_mask)

        if features is not None:
            # Add autoencoder features if available
            if encoder is not None:
                ae_features = extract_autoencoder_features(enhanced_img, mask, encoder, metadata_mask)
                if ae_features is not None:
                    features['autoencoder'] = ae_features

            mask_features.append(features)
            valid_masks.append(mask & metadata_mask)

    # Step 5: Train GMM on the features
    print(f"Training GMM with {len(mask_features)} particles and {gmm_components} components...")
    gmm_result = train_gmm(mask_features, n_components=gmm_components)

    if gmm_result is None:
        print("Error: GMM training failed! Falling back to intensity-based classification...")
        # Create a segmentation using traditional thresholding
        segmentation = np.zeros_like(img, dtype=np.uint8)

        # Mark voids
        segmentation[img < void_threshold] = 0

        # Classify particles using intensity thresholds
        for i, (mask, features) in enumerate(zip(valid_masks, mask_features)):
            if features is None:
                continue

            mean_intensity = features['mean_intensity']

            if mean_intensity < 60:
                segmentation[mask] = 1  # Carbon Black
            elif mean_intensity < 150:
                segmentation[mask] = 2  # LFP
            else:
                segmentation[mask] = 3  # NaCl
    else:
        # Unpack GMM results
        gmm, scaler, cluster_to_material = gmm_result

        # Step 6: Create initial segmentation using GMM-classified masks
        segmentation = np.zeros_like(img, dtype=np.uint8)

        # First, identify clear voids by thresholding
        void_regions = (img < void_threshold) & metadata_mask & (~particle_mask)
        segmentation[void_regions] = 0

        # Classify each mask using GMM
        gmm_labels = []
        gmm_confidences = []

        for i, (mask, features) in enumerate(zip(valid_masks, mask_features)):
            if features is None:
                gmm_labels.append(None)
                gmm_confidences.append(0)
                continue

            # Classify with GMM
            result = classify_with_gmm(features, gmm, scaler, cluster_to_material)

            if result is None:
                gmm_labels.append(None)
                gmm_confidences.append(0)
                continue

            material, confidence = result
            gmm_labels.append(material)
            gmm_confidences.append(confidence)

            # Apply label to segmentation
            segmentation[mask] = material

        # Step 7: Classify remaining unclassified regions
        unclassified = (segmentation == 0) & (~void_regions) & metadata_mask

        if np.any(unclassified):
            print(f"Classifying {np.sum(unclassified)} unclassified pixels...")

            # Create a distance map to each material
            distance_maps = np.zeros((img.shape[0], img.shape[1], 3))

            for material in range(1, 4):
                material_mask = (segmentation == material)
                if np.any(material_mask):
                    distance_maps[:, :, material - 1] = ndimage.distance_transform_edt(~material_mask)

            # Get typical intensities for each material from GMM means
            if gmm is not None:
                means = scaler.inverse_transform(gmm.means_)
                mean_intensities = []

                for cluster in range(gmm_components):
                    if cluster in cluster_to_material and cluster_to_material[cluster] > 0:
                        material = cluster_to_material[cluster]
                        intensity = means[cluster, 0]  # Mean intensity feature
                        mean_intensities.append((material, intensity))

                # Sort by material class
                mean_intensities.sort()

                # Get locations of unclassified pixels
                unclass_y, unclass_x = np.where(unclassified)

                # For each unclassified pixel
                for i in range(len(unclass_y)):
                    y, x = unclass_y[i], unclass_x[i]
                    intensity = img[y, x]

                    # Get distances to each material
                    distances = distance_maps[y, x]

                    # Weight distances by intensity difference
                    weighted_distances = np.zeros(3)

                    for j, (material, mean_int) in enumerate(mean_intensities):
                        # Calculate distance penalty based on intensity difference
                        int_diff = abs(intensity - mean_int)
                        int_penalty = int_diff / 255.0

                        # Adjust distance by intensity similarity
                        weighted_distances[material - 1] = distances[material - 1] * (1 + int_penalty)

                    # Assign to material with smallest weighted distance
                    material = np.argmin(weighted_distances) + 1
                    segmentation[y, x] = material

    # Step 8: Post-process the segmentation for coherent regions
    print("Post-processing segmentation for coherent regions...")

    # Apply morphological operations to smooth boundaries
    # First clean up each material separately
    processed = segmentation.copy()

    for material in range(1, 4):  # Skip voids
        material_mask = (processed == material)

        # Close small gaps
        closed = morphology.closing(material_mask, morphology.disk(2))

        # Remove small isolated regions (size depends on material)
        min_size = 20 if material != 1 else 50  # Larger for carbon black
        cleaned = morphology.remove_small_objects(closed, min_size=min_size)

        # Fill small holes
        filled = ndimage.binary_fill_holes(cleaned)

        # Update segmentation
        processed[material_mask & (~filled)] = 0  # Removed regions become void
        processed[(~material_mask) & filled & metadata_mask] = material  # Added regions

    # Ensure metadata area is blacked out
    processed[~metadata_mask] = 0

    # Step 9: Calculate material statistics
    stats = calculate_material_stats(processed, metadata_mask)

    # Step 10: Visualize results
    fig = visualize_segmentation(img, processed, image_path, metadata_mask)

    # Step 11: Save results if output directory is provided
    if output_dir:
        base_name = os.path.splitext(os.path.basename(image_path))[0]

        # Save visualization
        output_image_path = os.path.join(output_dir, f"{base_name}_gmm_sam_segmentation.png")
        fig.savefig(output_image_path, dpi=300, bbox_inches='tight')
        print(f"Saved segmentation visualization to: {output_image_path}")

        # Export to CSV
        output_csv_path = os.path.join(output_dir, f"{base_name}_gmm_sam_segmentation.csv")
        export_to_csv(processed, image_path, output_csv_path, metadata_mask)
        print(f"Saved segmentation data to: {output_csv_path}")

    plt.close(fig)  # Close the figure to avoid display in notebooks

    return processed, stats


def visualize_segmentation(original_img, segmented, image_path, metadata_mask=None):
    """Generate comparative visualization of original and segmented images.

    Args:
        original_img (numpy.ndarray): Source SEM image
        segmented (numpy.ndarray): Segmentation class map
        image_path (str): Source image path for labeling
        metadata_mask (numpy.ndarray, optional): Valid region mask. Defaults to None.

    Returns:
        matplotlib.figure.Figure: Generated figure object
    """
    material_labels = ["Voids", "Carbon Black", "LFP", "NaCl"]
    colors = ['black', 'gray', 'green', 'white']
    cmap = ListedColormap(colors)

    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))

    # Create a copy of the original for visualization
    vis_img = original_img.copy()

    # If we have a metadata mask, apply it
    if metadata_mask is not None:
        # Make a black area where metadata is
        vis_img = vis_img.copy()
        vis_img[~metadata_mask] = 0

    # Plot original image
    ax1.imshow(vis_img, cmap='gray')
    ax1.set_title("Original SEM Image", fontsize=14)
    ax1.axis('off')

    # Create visualization copy of segmentation
    vis_seg = segmented.copy()

    # If we have a metadata mask, set metadata area to class 0 (voids/black)
    if metadata_mask is not None:
        vis_seg[~metadata_mask] = 0

    # Plot segmented image
    ax2.imshow(vis_seg, cmap=cmap, vmin=0, vmax=3)
    ax2.set_title("GMM+SAM Segmentation", fontsize=14)
    ax2.axis('off')

    # Add filename as figure title
    plt.suptitle(f"Battery Material Analysis: {os.path.basename(image_path)}", fontsize=16)

    # Create legend
    legend_handles = [plt.Rectangle((0, 0), 1, 1, color=colors[i]) for i in range(len(material_labels))]
    fig.legend(legend_handles,
               [f"{i}: {material_labels[i]}" for i in range(len(material_labels))],
               loc="lower center", ncol=len(material_labels), frameon=False, fontsize=12)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)

    return fig


def calculate_material_stats(segmented, metadata_mask=None):
    """Calculate quantitative material distribution statistics.

        Args:
            segmented (numpy.ndarray): Segmentation class map
            metadata_mask (numpy.ndarray, optional): Valid region mask. Defaults to None.

        Returns:
            dict: Statistics containing pixel counts and percentages per class
    """
    material_labels = ["Voids", "Carbon Black", "LFP", "NaCl"]

    # Only count valid pixels (not in metadata area)
    if metadata_mask is not None:
        valid_pixels = segmented[metadata_mask]
        total_pixels = len(valid_pixels)
    else:
        valid_pixels = segmented.flatten()
        total_pixels = segmented.size

    # Count pixels in each class
    class_counts = np.bincount(valid_pixels.flatten(), minlength=4)
    class_percentages = class_counts / total_pixels * 100

    # Create statistics dictionary
    stats = {
        "pixel_counts": {material_labels[i]: int(class_counts[i]) for i in range(4)},
        "percentages": {material_labels[i]: float(class_percentages[i]) for i in range(4)},
        "total_pixels": int(total_pixels)
    }

    # Print results
    print("\nMaterial Distribution Analysis:")
    print("-" * 40)
    for i, material in enumerate(material_labels):
        print(f"{material}: {class_percentages[i]:.2f}% ({class_counts[i]} pixels)")
    print("-" * 40)

    return stats


def export_to_csv(segmented, image_path, output_path, metadata_mask=None):
    """Export segmentation results to CSV format with pixel-level annotations.

        Args:
            segmented (numpy.ndarray): Segmentation class map
            image_path (str): Source image path for metadata
            output_path (str): Destination CSV file path
            metadata_mask (numpy.ndarray, optional): Valid region mask. Defaults to None.
    """
    material_labels = ["Voids", "Carbon Black", "LFP", "NaCl"]
    h, w = segmented.shape

    # Create coordinate grids
    y_indices, x_indices = np.indices((h, w))

    # If we have a metadata mask, only include valid pixels
    if metadata_mask is not None:
        # Get positions of valid pixels
        valid_y = y_indices[metadata_mask]
        valid_x = x_indices[metadata_mask]
        valid_labels = segmented[metadata_mask]

        # Create corresponding material labels
        material_column = [material_labels[label] for label in valid_labels]

        # Create pixel coordinates
        pixels = [f"({x}, {y})" for x, y in zip(valid_x, valid_y)]

        # Create DataFrame
        df = pd.DataFrame({
            "ImageName": [os.path.basename(image_path)] * len(valid_y),
            "Pixel": pixels,
            "Class": valid_labels,
            "Material": material_column,
            "X": valid_x,
            "Y": valid_y
        })
    else:
        # Include all pixels
        x_flat = x_indices.flatten()
        y_flat = y_indices.flatten()
        labels_flat = segmented.flatten()

        # Create material label column
        material_column = [material_labels[label] for label in labels_flat]

        # Create pixel coordinates
        pixels = [f"({x}, {y})" for x, y in zip(x_flat, y_flat)]

        # Create DataFrame
        df = pd.DataFrame({
            "ImageName": [os.path.basename(image_path)] * len(x_flat),
            "Pixel": pixels,
            "Class": labels_flat,
            "Material": material_column,
            "X": x_flat,
            "Y": y_flat
        })

    # Save to CSV
    df.to_csv(output_path, index=False)


def main():
    """Command-line interface for segmentation pipeline.

        System Exit Codes:
            0: Success
            1: Input file error
    """
    parser = argparse.ArgumentParser(description='SEM Image Segmentation with GMM, Autoencoder, and SAM Masks')

    parser.add_argument('image_path', help='Path to the SEM image file (.png)')
    parser.add_argument('--masks',
                        help='Path to SAM masks pickle file (.pkl) - if not provided, will use same base name as image')
    parser.add_argument('--output', help='Output directory for results', default='gmm_sam_results')
    parser.add_argument('--void-threshold', type=int, default=10,
                        help='Threshold for void detection (lower = fewer voids)')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs for autoencoder training (0 to skip)')
    parser.add_argument('--gmm-components', type=int, default=4, help='Number of GMM components')

    args = parser.parse_args()

    # Validate image path
    if not os.path.isfile(args.image_path):
        print(f"Error: Image file not found: {args.image_path}")
        return 1

    # If masks path not provided, infer it from image path
    masks_path = args.masks
    if masks_path is None:
        base_name = os.path.splitext(args.image_path)[0]
        masks_path = f"{base_name}.pkl"

    if not os.path.isfile(masks_path):
        print(f"Error: Masks file not found: {masks_path}")
        return 1

    # Run the segmentation
    segment_with_gmm_sam(
        args.image_path,
        masks_path=masks_path,
        output_dir=args.output,
        void_threshold=args.void_threshold,
        epochs=args.epochs,
        gmm_components=args.gmm_components
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())