"""
SAM-Enhanced Battery Material Segmentation

This script performs segmentation of SEM images of battery materials using
SAM (Segment Anything Model) masks combined with intensity-based classification.
It provides a robust approach for identifying and analyzing the distribution of
different materials in battery electrode samples.

Features:
- Combines SAM masks with intensity-based classification
- Identifies 4 material types: Voids, Carbon Black, LFP, NaCl
- Uses SAM masks to precisely identify particle boundaries
- Includes tunable parameters to control void detection
- Automatically removes microscope metadata bar
- Outputs visual segmentation results and material statistics

The algorithm uses a multi-step process:
1. Pre-processing the SEM image to enhance contrast
2. Loading SAM-generated particle masks
3. Classifying each particle based on intensity statistics
4. Creating a coherent segmentation with post-processing
5. Calculating material distribution statistics

Usage:
    python autoencoder_debugged.py image.png masks.pkl [options]

Requirements:
    - OpenCV
    - NumPy
    - Matplotlib
    - scipy
    - scikit-image
    - pandas
"""

import os
import pickle
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import pandas as pd
from scipy import ndimage
from skimage import filters, measure


def load_sem_image(image_path):
    """
    Load and return the original SEM image in grayscale format.

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


def detect_metadata_bar(img, threshold_percent=0.1):
    """
    Detect and create a mask for the microscope metadata bar at the bottom of the image.

    This function analyzes image gradients to identify text-like features that typically
    appear in the metadata bar at the bottom of SEM images.

    Parameters:
    -----------
    img : ndarray
        Original grayscale image
    threshold_percent : float
        Percentage of image height to check for abrupt changes

    Returns:
    --------
    metadata_mask : ndarray
        Binary mask where True indicates pixels to keep (not metadata)
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

    return metadata_mask


def preprocess_image(img):
    """
    Apply preprocessing to enhance image features.

    This function applies multiple image processing techniques to improve
    image quality for segmentation:
    1. Median filtering to reduce noise
    2. CLAHE for better contrast
    3. Gaussian blur for smoothing

    Parameters:
    -----------
    img : ndarray
        Input grayscale SEM image

    Returns:
    --------
    ndarray
        Enhanced image with improved contrast and reduced noise
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
    Load segmentation masks from SAM pickle file.

    Parameters:
    -----------
    pickle_path : str
        Path to the pickle file containing SAM masks

    Returns:
    --------
    list or None
        List of boolean masks representing segmented regions, or None if loading fails

    Notes:
    ------
    The pickle file should contain a list of dictionaries where each dictionary
    has a 'segmentation' key containing a mask array
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


def extract_mask_features(img, mask):
    """
    Extract features from a specific mask region.

    Calculates various features from the region defined by the mask:
    - Intensity statistics (mean, std, median)
    - Texture features
    - Shape features (area, perimeter, compactness)

    Parameters:
    -----------
    img : ndarray
        Input grayscale image
    mask : ndarray
        Boolean mask defining the region of interest

    Returns:
    --------
    dict or None
        Dictionary of features, or None if the mask is empty
    """
    # Create binary mask
    binary_mask = mask.astype(np.uint8)

    # Extract region of interest
    masked_img = cv2.bitwise_and(img, img, mask=binary_mask)

    # Calculate intensity features
    non_zero_indices = np.where(binary_mask > 0)
    if len(non_zero_indices[0]) == 0:  # Empty mask
        return None

    # Intensity statistics
    pixel_values = img[non_zero_indices]
    mean_intensity = np.mean(pixel_values)
    std_intensity = np.std(pixel_values)
    median_intensity = np.median(pixel_values)

    # Calculate texture features
    # Simple texture: standard deviation of intensity within mask
    texture_complexity = std_intensity / (mean_intensity + 1e-10)

    # Calculate shape features
    area = np.sum(binary_mask)
    perimeter = measure.perimeter(binary_mask)
    compactness = 4 * np.pi * area / (perimeter * perimeter + 1e-10) if perimeter > 0 else 0

    # Return feature dictionary
    features = {
        'mean_intensity': mean_intensity,
        'median_intensity': median_intensity,
        'std_intensity': std_intensity,
        'texture_complexity': texture_complexity,
        'area': area,
        'compactness': compactness
    }

    return features


def classify_mask_by_intensity(features, thresholds):
    """
    Classify mask based on intensity features.

    Parameters:
    -----------
    features : dict
        Dictionary of features extracted from a mask region
    thresholds : dict
        Dictionary of threshold values for classification:
        - 'voids': threshold for void detection
        - 'carbon_black': threshold for carbon black
        - 'lfp': threshold for LFP

    Returns:
    --------
    int or None
        Class label (0=void, 1=carbon black, 2=LFP, 3=NaCl), or None if features is None
    """
    if features is None:
        return None

    mean_intensity = features['mean_intensity']

    # Classify based on intensity
    if mean_intensity < thresholds['voids']:
        return 0  # Voids
    elif mean_intensity < thresholds['carbon_black']:
        return 1  # Carbon Black
    elif mean_intensity < thresholds['lfp']:
        return 2  # LFP
    else:
        return 3  # NaCl (brightest)


def create_segmentation_from_masks(img_shape, masks, labels):
    """
    Create segmentation image from masks and their labels.

    This function handles overlapping masks by prioritizing non-overlapping regions
    and using a voting scheme for overlapping regions.

    Parameters:
    -----------
    img_shape : tuple
        Shape of the output segmentation image (height, width)
    masks : list
        List of boolean masks for each detected region
    labels : list
        List of class labels corresponding to each mask

    Returns:
    --------
    ndarray
        Segmentation image with pixel values representing class labels
    """
    segmentation = np.zeros(img_shape, dtype=np.uint8)

    # Create a count map to resolve overlaps
    count_map = np.zeros(img_shape, dtype=np.uint8)

    # First pass: generate overlap counts
    for mask in masks:
        count_map[mask] += 1

    # Second pass: assign labels, prioritizing non-overlapping regions
    # For overlapping regions, we'll take the average of classifications
    overlap_map = np.zeros((img_shape[0], img_shape[1], 4), dtype=np.float32)

    for i, (mask, label) in enumerate(zip(masks, labels)):
        if label is not None:  # Skip unclassified masks
            overlap_map[mask, label] += 1

    # For each pixel, select the class with the highest count
    for i in range(img_shape[0]):
        for j in range(img_shape[1]):
            if np.sum(overlap_map[i, j]) > 0:
                segmentation[i, j] = np.argmax(overlap_map[i, j])

    return segmentation


def improve_segmentation(segmentation, img):
    """
    Apply post-processing to improve segmentation quality.

    This function applies various morphological operations to:
    - Remove small isolated regions
    - Fill holes in larger regions
    - Smooth boundaries between regions

    Parameters:
    -----------
    segmentation : ndarray
        Initial segmentation image with class labels
    img : ndarray
        Original grayscale image for reference

    Returns:
    --------
    ndarray
        Improved segmentation with smoother boundaries and fewer artifacts
    """
    # Create a copy of the segmentation
    improved = segmentation.copy()

    # Fill small holes in each material class
    for class_id in range(4):
        binary_mask = (improved == class_id)

        # Remove very small objects
        min_size = 20 if class_id != 1 else 50  # Larger for carbon black
        cleaned = ndimage.binary_opening(binary_mask, structure=np.ones((3, 3)))
        cleaned = ndimage.binary_closing(cleaned, structure=np.ones((3, 3)))

        # Remove small objects
        labeled, num = ndimage.label(cleaned)
        sizes = ndimage.sum(cleaned, labeled, range(1, num + 1))
        mask_sizes = sizes < min_size
        remove_pixels = mask_sizes[labeled - 1]
        cleaned[remove_pixels] = False

        # Update the improved segmentation
        improved[binary_mask & (~cleaned)] = 0  # Default to void for removed small regions
        improved[cleaned & (~binary_mask)] = class_id  # Add regions that were filled

    # Apply median filter for smoother boundaries
    improved = ndimage.median_filter(improved, size=3)

    return improved


def visualize_segmentation(original_img, segmented, image_path, metadata_mask=None):
    """
    Create visualization of the segmentation results.

    Parameters:
    -----------
    original_img : ndarray
        Original grayscale SEM image
    segmented : ndarray
        Segmentation image with class labels
    image_path : str
        Path to the original image file (used for title)
    metadata_mask : ndarray, optional
        Boolean mask where True indicates valid image area (not metadata)

    Returns:
    --------
    matplotlib.figure.Figure
        Figure object containing the visualization
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
    ax2.set_title("SAM-Enhanced Segmentation", fontsize=14)
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
    """
    Calculate statistics about material distribution.

    Parameters:
    -----------
    segmented : ndarray
        Segmentation image with class labels
    metadata_mask : ndarray, optional
        Boolean mask where True indicates valid image area (not metadata)

    Returns:
    --------
    dict
        Dictionary containing:
        - pixel_counts: count of pixels for each material
        - percentages: percentage of each material
        - total_pixels: total number of valid pixels analyzed
    """
    material_labels = ["Voids", "Carbon Black", "LFP", "NaCl"]

    # Create a mask of valid pixels (excluding metadata)
    if metadata_mask is not None:
        valid_mask = metadata_mask
        valid_pixels = np.sum(valid_mask)
        segmented_valid = segmented[valid_mask]
        total_pixels = valid_pixels
    else:
        segmented_valid = segmented.flatten()
        total_pixels = segmented.size

    # Count pixels in each class
    class_counts = np.bincount(segmented_valid.flatten(), minlength=4)
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
    """
    Export segmentation results to CSV.

    Parameters:
    -----------
    segmented : ndarray
        Segmentation image with class labels
    image_path : str
        Path to the original image file
    output_path : str
        Path where the CSV file will be saved
    metadata_mask : ndarray, optional
        Boolean mask where True indicates valid image area (not metadata)

    Notes:
    ------
    The CSV file contains columns for:
    - ImageName: name of the original image
    - Pixel: coordinate of each pixel
    - Class: numeric class label
    - Material: text name of the material
    - X, Y: separate coordinate values
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
    print(f"CSV output saved as '{output_path}'")


def segment_sem_with_sam(image_path, masks_path, output_dir=None, void_threshold=10):
    """
    Main function for SEM image segmentation using SAM masks.

    This function orchestrates the entire segmentation process:
    1. Loading and preprocessing the image
    2. Detecting and masking the metadata bar
    3. Loading SAM masks
    4. Applying dynamic threshold determination
    5. Classifying particles based on intensity
    6. Creating and improving the segmentation
    7. Calculating statistics and saving results

    Parameters:
    -----------
    image_path : str
        Path to the SEM image
    masks_path : str
        Path to the SAM pickle file with masks
    output_dir : str, optional
        Directory to save results
    void_threshold : int, optional
        Threshold for void classification (lower = fewer voids), default=10

    Returns:
    --------
    tuple (ndarray, dict) or (None, None)
        (segmentation, stats) - Final segmentation and statistics dictionary,
        or (None, None) if processing fails
    """
    print(f"Processing {os.path.basename(image_path)} with SAM masks")

    # Create output directory if needed
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Step 1: Load image and masks
    img = load_sem_image(image_path)

    # Detect and mask out microscope metadata bar
    metadata_mask = detect_metadata_bar(img)

    enhanced_img = preprocess_image(img)
    masks = load_sam_masks(masks_path)

    if masks is None or len(masks) == 0:
        print("Error: No valid masks found!")
        return None, None

    # CRITICAL CHANGE #1: Create a combined particle mask from all SAM masks
    particle_mask = np.zeros_like(img, dtype=bool)
    for mask in masks:
        particle_mask = particle_mask | mask

    # Apply metadata mask to particle mask
    particle_mask = particle_mask & metadata_mask

    # Step 2: Get image statistics
    # Only consider pixels that are not in metadata area
    valid_pixels = enhanced_img[metadata_mask]

    img_stats = {
        'min': np.min(valid_pixels),
        'max': np.max(valid_pixels),
        'mean': np.mean(valid_pixels),
        'std': np.std(valid_pixels)
    }

    # CRITICAL CHANGE #2: Use dynamic thresholding based on image histogram
    hist, bins = np.histogram(valid_pixels, bins=256, range=(0, 256))
    cumsum = np.cumsum(hist)

    # Find more appropriate thresholds based on histogram
    # Use lower threshold for voids - only the darkest 5-10% should be voids
    void_percentile = 0.05  # Bottom 5% intensity = voids
    carbon_percentile = 0.30  # Bottom 30% intensity = carbon black
    lfp_percentile = 0.70  # Up to 70% intensity = LFP

    total_valid_pixels = valid_pixels.size
    void_idx = np.searchsorted(cumsum, total_valid_pixels * void_percentile)
    carbon_idx = np.searchsorted(cumsum, total_valid_pixels * carbon_percentile)
    lfp_idx = np.searchsorted(cumsum, total_valid_pixels * lfp_percentile)

    thresholds = {
        'voids': void_threshold,  # Use the parameter value
        'carbon_black': bins[carbon_idx],
        'lfp': bins[lfp_idx]
    }

    print(
        f"Dynamic thresholds: Voids < {thresholds['voids']}, Carbon < {thresholds['carbon_black']}, LFP < {thresholds['lfp']}")

    # Step 3: Extract features and classify each mask
    mask_features = []
    mask_labels = []

    # Create initial segmentation map
    segmentation = np.zeros_like(img, dtype=np.uint8)

    # CRITICAL CHANGE #3: Only classify as void if extremely dark AND not in any mask
    # First mark all potential voids (very dark regions)
    potential_voids = (enhanced_img < thresholds['voids'])

    # Then exclude any that are part of a SAM mask
    true_voids = potential_voids & (~particle_mask)

    # Also exclude metadata area from voids
    true_voids = true_voids & metadata_mask

    # Set these as class 0 (voids)
    segmentation[true_voids] = 0

    # Now process each SAM mask and classify it
    for i, mask in enumerate(masks):
        # Ensure the mask doesn't overlap with metadata area
        masked_area = mask & metadata_mask

        features = extract_mask_features(enhanced_img, masked_area)

        if features is None:
            mask_labels.append(None)
            continue

        # Use mean intensity for classification
        mean_intensity = features['mean_intensity']

        # Classify based on intensity
        if mean_intensity < thresholds['carbon_black']:
            label = 1  # Carbon Black
        elif mean_intensity < thresholds['lfp']:
            label = 2  # LFP
        else:
            label = 3  # NaCl

        mask_labels.append(label)

        # Apply this label to the segmentation where mask is True
        # and not already classified as void
        segmentation[masked_area & (~true_voids)] = label

    # CRITICAL CHANGE #4: Classify remaining unclassified regions
    # Areas that are not voids, not in any mask, and in valid image area
    unclassified = (segmentation == 0) & (~true_voids) & metadata_mask

    if np.any(unclassified):
        print(f"Classifying {np.sum(unclassified)} unclassified pixels")

        # Create a distance transform from particle edges
        distance = ndimage.distance_transform_edt(~particle_mask)

        # Get locations of unclassified pixels
        unclass_y, unclass_x = np.where(unclassified)

        # For each unclassified pixel
        for i in range(len(unclass_y)):
            y, x = unclass_y[i], unclass_x[i]

            # If close to a particle (within 5 pixels), use its intensity
            if distance[y, x] < 5:
                intensity = enhanced_img[y, x]

                if intensity < thresholds['carbon_black']:
                    segmentation[y, x] = 1  # Carbon Black
                elif intensity < thresholds['lfp']:
                    segmentation[y, x] = 2  # LFP
                else:
                    segmentation[y, x] = 3  # NaCl

    # Step 5: Improve segmentation with post-processing
    final_segmentation = improve_segmentation(segmentation, enhanced_img)

    # Make sure metadata area is set to void
    final_segmentation[~metadata_mask] = 0

    # Step 6: Calculate material statistics
    stats = calculate_material_stats(final_segmentation, metadata_mask)

    # Step 7: Visualize results
    fig = visualize_segmentation(img, final_segmentation, image_path, metadata_mask)

    # Save results if output directory is specified
    if output_dir:
        base_name = os.path.splitext(os.path.basename(image_path))[0]

        # Save visualization
        fig.savefig(os.path.join(output_dir, f"{base_name}_sam_segmented.png"),
                    dpi=300, bbox_inches='tight')

        # Export CSV
        csv_path = os.path.join(output_dir, f"{base_name}_sam_segmentation.csv")
        export_to_csv(final_segmentation, image_path, csv_path, metadata_mask)

    return final_segmentation, stats


if __name__ == "__main__":
    # Example usage
    image_path = "NaCl_Komposit_60_min_fluid039.png"  # Use your image path
    masks_path = "NaCl_Komposit_60_min_fluid039.pkl"  # Use your masks path
    output_dir = "sam_autoencoder_segmentation_results"

    # Run the segmentation with a lower void threshold (try values between 5-15)
    # Lower values will classify fewer areas as voids
    segmentation, stats = segment_sem_with_sam(
        image_path,
        masks_path,
        output_dir=output_dir,
        void_threshold=10  # Adjust this value as needed - lower = fewer voids
    )

    plt.show()