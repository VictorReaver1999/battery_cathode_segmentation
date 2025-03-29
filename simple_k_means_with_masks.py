"""
Simple K-means SEM Image Segmentation with SAM Masks

This script performs segmentation of SEM images of battery materials using:
1. SAM (Segment Anything Model) masks for particle detection
2. Simple intensity thresholds for material classification

The segmentation classifies pixels into four categories:
- Voids (black): Areas with no particles, typically very dark
- Carbon Black (gray): Darkest particles
- LFP (red): Medium-brightness particles
- NaCl (white): Brightest particles

The script automatically detects and removes the microscope metadata bar.

Usage:
    python simple_k_means_with_masks.py image.png [options]

Requirements:
    - OpenCV
    - NumPy
    - Matplotlib
    - SAM mask files (.pkl) with the same base name as the input image
"""

import os
import sys
import pickle
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def load_sem_image(image_path):
    """
    Load the SEM image in grayscale.

    Parameters:
    -----------
    image_path : str
        Path to the input SEM image file

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


def detect_metadata_bar(img, metadata_height=80):
    """
    Mask the metadata bar at the bottom of the image.

    Parameters:
    -----------
    img : ndarray
        Input SEM image
    metadata_height : int, optional
        Height of the metadata bar in pixels (default: 80)

    Returns:
    --------
    ndarray
        Boolean mask where True indicates valid image area (not metadata)
    """
    h, w = img.shape
    metadata_mask = np.ones_like(img, dtype=bool)
    metadata_mask[-metadata_height:, :] = False
    return metadata_mask


def preprocess_image(img):
    """
    Enhance the SEM image for better segmentation.

    Parameters:
    -----------
    img : ndarray
        Input SEM image

    Returns:
    --------
    ndarray
        Enhanced image with improved contrast
    """
    # Apply CLAHE for contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_clahe = clahe.apply(img)
    return img_clahe


def load_sam_masks(image_path):
    """
    Load SAM masks from the corresponding .pkl file.

    Parameters:
    -----------
    image_path : str
        Path to the input SEM image file. The function will look for a .pkl file
        with the same base name.

    Returns:
    --------
    list
        List of boolean masks from the SAM segmentation

    Raises:
    -------
    FileNotFoundError
        If the corresponding mask file cannot be found
    """
    base_name = os.path.splitext(image_path)[0]
    pickle_path = f"{base_name}.pkl"
    if not os.path.exists(pickle_path):
        raise FileNotFoundError(f"Mask file not found: {pickle_path}")

    with open(pickle_path, 'rb') as f:
        data = pickle.load(f)

    masks = [item['segmentation'] for item in data if 'segmentation' in item]
    return masks


def segment_image(image_path, output_dir, void_threshold=40, carbon_threshold=100, lfp_threshold=170):
    """
    Segment the SEM image using SAM masks and intensity thresholds.

    Parameters:
    -----------
    image_path : str
        Path to the SEM image file
    output_dir : str
        Directory to save the segmentation results
    void_threshold : int, optional
        Intensity threshold for void detection (default: 40)
    carbon_threshold : int, optional
        Intensity threshold for carbon black detection (default: 100)
    lfp_threshold : int, optional
        Intensity threshold for LFP detection (default: 170)

    Returns:
    --------
    ndarray
        The segmented image as a NumPy array with values:
        0 = Voids, 1 = Carbon Black, 2 = LFP, 3 = NaCl
    """
    # Load the SEM image
    img = load_sem_image(image_path)

    # Mask the metadata bar
    metadata_mask = detect_metadata_bar(img)

    # Preprocess the image
    img_preprocessed = preprocess_image(img)

    # Load SAM masks
    masks = load_sam_masks(image_path)

    # Initialize segmentation map
    segmentation = np.zeros_like(img, dtype=np.uint8)

    # Apply SAM masks to classify regions
    for mask in masks:
        masked_area = mask & metadata_mask
        mean_intensity = np.mean(img_preprocessed[masked_area])

        # Classify based on intensity thresholds
        if mean_intensity < carbon_threshold:
            label = 1  # Carbon Black
        elif mean_intensity < lfp_threshold:
            label = 2  # LFP
        else:
            label = 3  # NaCl

        segmentation[masked_area] = label

    # Classify voids (regions not covered by SAM masks)
    voids = (segmentation == 0) & metadata_mask & (img_preprocessed < void_threshold)
    segmentation[voids] = 0  # Voids

    # Ensure metadata area is black
    segmentation[~metadata_mask] = 0

    # Save the segmentation result
    save_segmentation(image_path, img, segmentation, output_dir)

    return segmentation


def save_segmentation(image_path, original_img, segmentation, output_dir):
    """
    Save the segmented image with the specified colors and side-by-side comparison.

    Parameters:
    -----------
    image_path : str
        Path to the original SEM image file
    original_img : ndarray
        Original SEM image
    segmentation : ndarray
        Segmentation result
    output_dir : str
        Directory to save the output visualization
    """
    # Define colors for each class
    material_colors = ['black', 'gray', 'red', 'white']
    cmap = ListedColormap(material_colors)

    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Plot the original and segmented images side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))

    # Original image
    ax1.imshow(original_img, cmap='gray')
    ax1.set_title("Original Image", fontsize=14)
    ax1.axis('off')

    # Segmented image
    ax2.imshow(segmentation, cmap=cmap, vmin=0, vmax=3)
    ax2.set_title("Segmented Image", fontsize=14)
    ax2.axis('off')

    # Add a legend
    material_labels = ["Voids", "Carbon Black", "LFP", "NaCl"]
    legend_handles = [plt.Rectangle((0, 0), 1, 1, color=material_colors[i]) for i in range(4)]
    fig.legend(legend_handles,
               [f"{i}: {material_labels[i]}" for i in range(4)],
               loc="lower center", ncol=4, frameon=False, fontsize=12)

    # Save the figure
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    output_path = os.path.join(output_dir, f"{base_name}_segmented_comparison.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f"Segmented image saved to: {output_path}")


if __name__ == "__main__":
    import argparse

    # Command-line argument parser
    parser = argparse.ArgumentParser(description="SEM Image Segmentation with SAM Masks")
    parser.add_argument("image_path", type=str, help="Path to the SEM image")
    parser.add_argument("--output_dir", type=str, default="segmentation_results", help="Directory to save results")
    parser.add_argument("--void_threshold", type=int, default=40, help="Threshold for void classification")
    parser.add_argument("--carbon_threshold", type=int, default=100, help="Threshold for carbon black classification")
    parser.add_argument("--lfp_threshold", type=int, default=170, help="Threshold for LFP classification")

    args = parser.parse_args()

    # Run the segmentation
    segment_image(
        args.image_path,
        args.output_dir,
        void_threshold=args.void_threshold,
        carbon_threshold=args.carbon_threshold,
        lfp_threshold=args.lfp_threshold
    )