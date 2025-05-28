import sys
import os
import pickle
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.optimizers import Adam
from sklearn.cluster import KMeans


def load_mask_from_pkl(pkl_file, mask_index=0):
    """Load a specific mask from a pickle file"""
    with open(pkl_file, 'rb') as f:
        data = pickle.load(f)

    print(f"Total masks in file: {len(data)}")
    if mask_index >= len(data):
        print(f"Warning: Requested mask index {mask_index} exceeds available masks. Using mask 0.")
        mask_index = 0

    mask = data[mask_index]['segmentation']
    return mask, data[mask_index]


def preprocess_image(img):
    """Enhance image quality using median filtering, CLAHE, and Gaussian blur."""
    img_median = cv2.medianBlur(img, 3)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_clahe = clahe.apply(img_median)
    img_enhanced = cv2.GaussianBlur(img_clahe, (3, 3), 0)
    return img_enhanced


def build_autoencoder(input_shape=(32, 32, 1)):
    """Build a convolutional autoencoder for feature extraction."""
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


def extract_region(image, mask, target_size=(32, 32)):
    """Extract and process a masked region for feature extraction."""
    coords = np.where(mask)
    if coords[0].size == 0:
        return None, None, None, None

    # Get bounding box
    y_min, y_max = coords[0].min(), coords[0].max()
    x_min, x_max = coords[1].min(), coords[1].max()

    # Original region shape
    region_shape = (y_max - y_min + 1, x_max - x_min + 1)

    # Extract region and apply mask
    cropped = image[y_min:y_max + 1, x_min:x_max + 1].copy()
    mask_region = mask[y_min:y_max + 1, x_min:x_max + 1]
    masked = cropped.copy()
    masked[~mask_region] = 0

    # Pad to square
    height, width = masked.shape
    max_dim = max(height, width)
    padded = np.zeros((max_dim, max_dim), dtype=masked.dtype)
    y_offset = (max_dim - height) // 2
    x_offset = (max_dim - width) // 2
    padded[y_offset:y_offset + height, x_offset:x_offset + width] = masked

    # Resize to target size
    resized = cv2.resize(padded, target_size)

    # Normalize for the encoder
    normalized = resized.astype("float32") / 255.0

    return (y_min, y_max, x_min, x_max), masked, padded, normalized


def visualize_feature_maps(encoder, normalized_image):
    """Visualize the feature maps from the autoencoder's encoder."""
    normalized_image_expanded = np.expand_dims(normalized_image, axis=(0, -1))
    feature_maps = encoder.predict(normalized_image_expanded)
    feature_maps = np.squeeze(feature_maps)

    # Visualize a subset of feature maps
    num_features = min(16, feature_maps.shape[-1])
    plt.figure(figsize=(8, 8))
    for i in range(num_features):
        plt.subplot(4, 4, i + 1)
        plt.imshow(feature_maps[:, :, i], cmap='viridis')
        plt.axis('off')
        plt.title(f"Feature {i + 1}")
    plt.tight_layout()
    return feature_maps


def visualize_mask_transformations(image_path, pkl_path, mask_index=0, output_dir='mask_transformations_kmeans'):
    """Visualize the transformations a mask undergoes in the K-means workflow."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load image and mask
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Error: Could not load image {image_path}")
        return

    enhanced_img = preprocess_image(img)
    mask, mask_metadata = load_mask_from_pkl(pkl_path, mask_index)

    # Extract the mask region
    bbox, masked_region, padded_square, normalized = extract_region(enhanced_img, mask)
    if bbox is None:
        print("Error: Empty mask region")
        return

    # Build autoencoder for feature extraction
    autoencoder, encoder = build_autoencoder()

    # Create figure for all transformations
    plt.figure(figsize=(15, 12))

    # 1. Original SEM image with mask overlay
    plt.subplot(2, 3, 1)
    plt.title("1. Original Image with Mask Overlay")
    plt.imshow(img, cmap='gray')
    y_min, y_max, x_min, x_max = bbox
    mask_overlay = np.zeros_like(img, dtype=np.uint8)
    mask_overlay[mask] = 255
    mask_rgb = np.zeros((*img.shape, 3), dtype=np.uint8)
    mask_rgb[:, :, 1] = mask_overlay  # Green channel for mask
    plt.imshow(mask_rgb, alpha=0.3)
    plt.axis('off')

    # 2. Bounding Box Extraction
    plt.subplot(2, 3, 2)
    plt.title(f"2. Bounding Box Extraction\n{y_max - y_min + 1}x{x_max - x_min + 1}")
    bbox_img = img.copy()
    cv2.rectangle(bbox_img, (x_min, y_min), (x_max, y_max), 255, 2)
    plt.imshow(bbox_img, cmap='gray')
    plt.axis('off')

    # 3. Masked Region
    plt.subplot(2, 3, 3)
    plt.title("3. Masked Region\n(Background pixels set to 0)")
    plt.imshow(masked_region, cmap='gray')
    plt.axis('off')

    # 4. Padded Square
    plt.subplot(2, 3, 4)
    plt.title(f"4. Padded to Square\n{padded_square.shape[0]}x{padded_square.shape[1]}")
    plt.imshow(padded_square, cmap='gray')
    plt.axis('off')

    # 5. Resized and Normalized
    plt.subplot(2, 3, 5)
    plt.title(f"5. Resized to 32x32\nNormalized [0,1]")
    plt.imshow(normalized, cmap='gray')
    plt.axis('off')

    # Save the main transformations
    plt.tight_layout()
    main_output_file = os.path.join(output_dir, f"mask_{mask_index}_main_transformations.png")
    plt.savefig(main_output_file)

    # Feature extraction visualization (separate figure)
    feature_maps = visualize_feature_maps(encoder, normalized)
    feature_output_file = os.path.join(output_dir, f"mask_{mask_index}_feature_maps.png")
    plt.savefig(feature_output_file)

    # Simulate clustering result with 4 clusters
    # Since we only have one sample, we'll just assign a random cluster
    # In the real workflow, this would be based on the feature vector
    cluster_labels = ["Void (0)", "Carbon Black (1)", "LFP (2)", "NaCl (3)"]
    cluster_colors = ['black', 'gray', 'green', 'white']
    assigned_label = np.random.randint(0, 4)  # Random assignment for demonstration

    plt.figure(figsize=(8, 6))
    plt.subplot(1, 2, 1)
    plt.title(f"Cluster Assignment\n{cluster_labels[assigned_label]}")
    plt.imshow(np.ones((100, 100)) * assigned_label, cmap='gray',
               vmin=0, vmax=3)
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title("Material Segmentation Colors")
    for i, (label, color) in enumerate(zip(cluster_labels, cluster_colors)):
        plt.plot([0, 1], [i, i], color=color, linewidth=20)
        plt.text(1.1, i, label, va='center')
    plt.axis('off')
    plt.xlim(0, 3)
    plt.ylim(-0.5, 3.5)

    cluster_output_file = os.path.join(output_dir, f"mask_{mask_index}_cluster_result.png")
    plt.savefig(cluster_output_file)

    print(f"Visualizations saved to {output_dir}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python visualize_kmeans_transformations.py image.png masks.pkl [mask_index]")
        sys.exit(1)

    image_path = sys.argv[1]
    pkl_path = sys.argv[2]
    mask_index = int(sys.argv[3]) if len(sys.argv) > 3 else 0

    visualize_mask_transformations(image_path, pkl_path, mask_index)
