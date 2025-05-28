import os
import sys
import pickle
import numpy as np
import cv2
import matplotlib.pyplot as plt


def load_mask_from_pkl(pkl_file, mask_index=0):
    """Load a specific mask from a pickle file"""
    with open(pkl_file, 'rb') as f:
        data = pickle.load(f)

    print(f"Total masks in file: {len(data)}")
    print(f"Mask keys: {data[mask_index].keys()}")

    mask = data[mask_index]['segmentation']
    area = data[mask_index]['area']
    stability = data[mask_index]['stability_score'] if 'stability_score' in data[mask_index] else 'N/A'
    bbox = data[mask_index]['bbox'] if 'bbox' in data[mask_index] else 'N/A'

    print(f"Mask shape: {mask.shape}")
    print(f"Mask area: {area}")
    print(f"Mask stability score: {stability}")
    print(f"Mask bounding box: {bbox}")
    print(f"Mask type: {mask.dtype}")
    print(f"Mask unique values: {np.unique(mask)}")

    return mask, data[mask_index]


def resize_mask(mask, target_height, target_width):
    """Resize a mask to fit within target dimensions while maintaining aspect ratio"""
    mask_h, mask_w = mask.shape

    # Calculate scaling factors
    scale_h = target_height / mask_h
    scale_w = target_width / mask_w
    scale = min(scale_h, scale_w)

    # Calculate new dimensions
    new_width = int(mask_w * scale)
    new_height = int(mask_h * scale)

    # Resize using nearest neighbor interpolation to preserve binary values
    resized_mask = cv2.resize(mask.astype(np.uint8), (new_width, new_height),
                              interpolation=cv2.INTER_NEAREST)

    return resized_mask


def visualize_mask_transformations(pkl_file, mask_index=0, output_dir='mask_transformations'):
    """Extract a mask from PKL file and visualize its transformations"""
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load mask from PKL file
    original_mask, mask_metadata = load_mask_from_pkl(pkl_file, mask_index)

    # 1. Original mask visualization
    plt.figure(figsize=(12, 10))

    plt.subplot(2, 3, 1)
    plt.title(f"1. Original Mask\nShape: {original_mask.shape}")
    plt.imshow(original_mask, cmap='gray')

    # 2. Binary conversion (True/False to 0/255)
    binary_mask = (original_mask > 0).astype(np.uint8) * 255
    plt.subplot(2, 3, 2)
    plt.title(f"2. Binary Conversion\n(Values: {np.unique(binary_mask)})")
    plt.imshow(binary_mask, cmap='gray')

    # 3. Resize if large (let's assume target is 200x200)
    target_h, target_w = 200, 200

    if original_mask.shape[0] > target_h or original_mask.shape[1] > target_w:
        resized_mask = resize_mask(original_mask, target_h, target_w)
        resizing_status = "Resized (too large)"
    else:
        resized_mask = original_mask
        resizing_status = "Original size (no resize needed)"

    plt.subplot(2, 3, 3)
    plt.title(f"3. Resize Check\n{resizing_status}\nNew shape: {resized_mask.shape}")
    plt.imshow(resized_mask, cmap='gray')

    # 4. Bounding box visualization
    if 'bbox' in mask_metadata:
        bbox = mask_metadata['bbox']

        # Create a colored representation with bounding box
        viz_mask = np.zeros((*original_mask.shape, 3), dtype=np.uint8)
        viz_mask[original_mask] = [0, 255, 0]  # Green for mask

        # Draw bounding box if available
        try:
            # Check if bbox is in the expected format and convert to integers
            if len(bbox) == 4:
                x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                x_end, y_end = x + w, y + h
                # Draw red rectangle
                cv2.rectangle(viz_mask, (x, y), (x_end, y_end), (255, 0, 0), 2)
            else:
                print(f"Warning: Bounding box format unexpected. Got: {bbox}")
        except Exception as e:
            print(f"Error drawing bounding box: {e}")

        plt.subplot(2, 3, 4)
        plt.title(f"4. Bounding Box\nBBox: {bbox}")
        plt.imshow(viz_mask)
    else:
        plt.subplot(2, 3, 4)
        plt.title("4. Bounding Box\n(Not available in metadata)")
        plt.imshow(original_mask, cmap='gray')

    # 5. Placement simulation (on black canvas)
    canvas_h, canvas_w = max(400, original_mask.shape[0]), max(400, original_mask.shape[1])
    canvas = np.zeros((canvas_h, canvas_w), dtype=np.uint8)

    # Random position for demonstration
    mask_h, mask_w = resized_mask.shape
    start_y = np.random.randint(0, canvas_h - mask_h)
    start_x = np.random.randint(0, canvas_w - mask_w)

    # Place the mask
    binary_resized = (resized_mask > 0).astype(np.uint8) * 255
    canvas[start_y:start_y + mask_h, start_x:start_x + mask_w] = binary_resized

    plt.subplot(2, 3, 5)
    plt.title(f"5. Random Placement\nPosition: ({start_x}, {start_y})")
    plt.imshow(canvas, cmap='gray')

    # 6. Final labeled output
    plt.subplot(2, 3, 6)
    plt.title("6. Labeled Result\n(Simulated output)")

    # Colorize the canvas for visualization
    colored_canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    colored_canvas[canvas > 0] = [255, 0, 0]  # Red for this mask
    plt.imshow(colored_canvas)

    # Save the visualization
    output_file = os.path.join(output_dir, f"mask_{mask_index}_transformations.png")
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    plt.close()
    print(f"Visualization saved to {output_file}")

    # Also save individual mask images for further analysis
    cv2.imwrite(os.path.join(output_dir, f"mask_{mask_index}_original.png"),
                (original_mask * 255).astype(np.uint8))
    cv2.imwrite(os.path.join(output_dir, f"mask_{mask_index}_binary.png"),
                binary_mask)
    cv2.imwrite(os.path.join(output_dir, f"mask_{mask_index}_placed.png"),
                canvas)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python visualize_mask_transformations.py path_to_pkl_file [mask_index]")
        sys.exit(1)

    pkl_file = sys.argv[1]
    mask_index = int(sys.argv[2]) if len(sys.argv) > 2 else 0

    visualize_mask_transformations(pkl_file, mask_index)
