"""
Encoder-only CNN for battery material segmentation in SEM images.

This module implements a simplified convolutional neural network architecture 
for segmenting battery material components (LFP, carbon black, NaCl, and voids)
in scanning electron microscope (SEM) images. The architecture uses only the 
encoding path of a traditional segmentation network, followed by direct upsampling
to the original image size.

The model processes SEM images to classify each pixel as one of four material classes,
using a fully convolutional approach that maintains spatial information throughout
the network while progressively increasing feature depth.
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import cv2


def build_encoder_only_cnn(input_shape=(512, 512, 1), num_classes=4):
    """
    Build an encoder-only CNN for semantic segmentation of battery materials.

    This model uses a series of convolutional blocks to progressively downsample
    the input image while increasing feature depth. The final feature map is then 
    classified using 1x1 convolutions, and upsampled back to the original image
    size using bilinear interpolation.

    Parameters
    ----------
    input_shape : tuple
        Shape of the input images (height, width, channels).
        Default is (512, 512, 1) for grayscale SEM images.
    num_classes : int
        Number of material classes to segment.
        Default is 4 (LFP, carbon black, NaCl, and voids).

    Returns
    -------
    tf.keras.Model
        Compiled Keras model for semantic segmentation.

    Notes
    -----
    The architecture progressively reduces spatial dimensions by a factor of 8
    (through three max-pooling operations) while increasing feature depth to 512.
    This results in each position in the feature map having a receptive field of
    approximately 64x64 pixels in the original image.
    """
    # Input layer
    inputs = layers.Input(input_shape, name='sem_image_input')

    # Encoder path (downsampling)
    # Block 1: Input (512x512x1) -> Output (256x256x64)
    # First level of feature extraction captures basic edge and texture information
    conv1 = layers.Conv2D(64, 3, activation='relu', padding='same',
                          name='block1_conv1')(inputs)
    conv1 = layers.BatchNormalization(name='block1_bn1')(conv1)
    conv1 = layers.Conv2D(64, 3, activation='relu', padding='same',
                          name='block1_conv2')(conv1)
    pool1 = layers.MaxPooling2D(pool_size=(2, 2), name='block1_pool')(conv1)

    # Block 2: Input (256x256x64) -> Output (128x128x128)
    # Second level extracts more complex patterns and material textures
    conv2 = layers.Conv2D(128, 3, activation='relu', padding='same',
                          name='block2_conv1')(pool1)
    conv2 = layers.BatchNormalization(name='block2_bn1')(conv2)
    conv2 = layers.Conv2D(128, 3, activation='relu', padding='same',
                          name='block2_conv2')(conv2)
    pool2 = layers.MaxPooling2D(pool_size=(2, 2), name='block2_pool')(conv2)

    # Block 3: Input (128x128x128) -> Output (64x64x256)
    # Third level captures material-specific features and local arrangements
    conv3 = layers.Conv2D(256, 3, activation='relu', padding='same',
                          name='block3_conv1')(pool2)
    conv3 = layers.BatchNormalization(name='block3_bn1')(conv3)
    conv3 = layers.Conv2D(256, 3, activation='relu', padding='same',
                          name='block3_conv2')(conv3)
    pool3 = layers.MaxPooling2D(pool_size=(2, 2), name='block3_pool')(conv3)

    # Block 4 (bottleneck): Input (64x64x256) -> Output (64x64x512)
    # Final level extracts high-level features with large receptive fields
    # Each position now has a deep feature vector representing a 64x64 region
    conv4 = layers.Conv2D(512, 3, activation='relu', padding='same',
                          name='block4_conv1')(pool3)
    conv4 = layers.BatchNormalization(name='block4_bn1')(conv4)
    conv4 = layers.Conv2D(512, 3, activation='relu', padding='same',
                          name='block4_conv2')(conv4)
    drop4 = layers.Dropout(0.5, name='block4_dropout')(conv4)

    # Classification layer: Input (64x64x512) -> Output (64x64xnum_classes)
    # 1x1 convolution maps deep features to class probabilities at each position
    # This implements the "1px wide and thousands deep" concept mentioned by supervisor
    class_predictions = layers.Conv2D(num_classes, 1, padding='same',
                                      name='classification_conv')(drop4)

    # Bilinear upsampling: Input (64x64xnum_classes) -> Output (512x512xnum_classes)
    # Simple resizing back to original dimensions without adding parameters
    outputs = layers.UpSampling2D(size=(8, 8), interpolation='bilinear',
                                  name='upsampling')(class_predictions)
    outputs = layers.Activation('softmax', name='segmentation_output')(outputs)

    # Create model
    model = models.Model(inputs=[inputs], outputs=[outputs],
                         name='EncoderOnlyCNN_BatterySegmentation')
    return model


def load_dataset(image_dir, mask_dir, img_size=(512, 512)):
    """
    Load and preprocess a dataset of SEM images and corresponding segmentation masks.

    Parameters
    ----------
    image_dir : str
        Directory containing the SEM images.
    mask_dir : str
        Directory containing the segmentation masks.
    img_size : tuple
        Target size for resizing images and masks. Default is (512, 512).

    Returns
    -------
    tuple
        (X, y) where:
        - X is a numpy array of shape (n_samples, height, width, 1) containing normalized images
        - y is a numpy array of shape (n_samples, height, width, n_classes) containing one-hot encoded masks

    Notes
    -----
    Images are normalized to [0,1] range.
    Masks should contain integer values (0, 1, 2, 3) representing different materials.
    These are converted to one-hot encoding for training.

    Examples
    --------
    >>> X, y = load_dataset('sem_images/', 'masks/', img_size=(256, 256))
    >>> print(f"Loaded {X.shape[0]} images of shape {X.shape[1:]}")
    """
    # Get file paths - make sure they're sorted for correct image-mask pairing
    image_paths = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir)
                          if f.endswith(('.png', '.tif', '.jpg'))])
    mask_paths = sorted([os.path.join(mask_dir, f) for f in os.listdir(mask_dir)
                         if f.endswith(('.png', '.tif', '.jpg'))])

    # Verify that we have matching numbers of images and masks
    if len(image_paths) != len(mask_paths):
        raise ValueError(f"Mismatch between number of images ({len(image_paths)}) "
                         f"and masks ({len(mask_paths)})")

    images = []
    masks = []

    # Load and preprocess images and masks
    for img_path, mask_path in zip(image_paths, mask_paths):
        # Load image as grayscale 
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Could not load image: {img_path}")

        # Resize to target dimensions
        img = cv2.resize(img, img_size)
        img = img / 255.0  # Normalize to [0,1]

        # Load mask (values should be 0-3, representing different materials)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Could not load mask: {mask_path}")

        # Resize using nearest neighbor to preserve label values
        mask = cv2.resize(mask, img_size, interpolation=cv2.INTER_NEAREST)

        # Convert to one-hot encoding for training
        mask_one_hot = tf.keras.utils.to_categorical(mask, num_classes=4)

        # Add channel dimension to image
        images.append(img[..., np.newaxis])
        masks.append(mask_one_hot)

    return np.array(images), np.array(masks)


def weighted_categorical_crossentropy(class_weights):
    """
    Create a weighted categorical crossentropy loss function for imbalanced classes.

    Parameters
    ----------
    class_weights : array_like
        Array of weights for each class, shape (num_classes,)

    Returns
    -------
    function
        Loss function that can be passed to model.compile()

    Notes
    -----
    This function addresses class imbalance by weighting the contribution
    of each class to the loss function. Classes that appear less frequently
    in the training data receive higher weights.
    """

    def loss(y_true, y_pred):
        # Clip predictions to avoid numerical instability
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)

        # Calculate weighted cross-entropy
        loss = y_true * tf.math.log(y_pred)
        loss = -tf.reduce_sum(loss * class_weights, axis=-1)
        return tf.reduce_mean(loss)

    return loss


def main():
    """
    Main execution function for training the battery material segmentation model.

    This function:
    1. Loads and preprocesses the dataset
    2. Splits data into training and validation sets
    3. Creates and compiles the model
    4. Trains the model with data augmentation
    5. Saves the trained model

    Command-line usage:
    $ python battery_segmentation.py

    Notes
    -----
    Modify image_dir and mask_dir paths before running the script.
    """
    # Set paths
    image_dir = "path/to/sem_images"
    mask_dir = "path/to/segmentation_masks"

    # Load dataset
    X, y = load_dataset(image_dir, mask_dir)
    print(f"Dataset loaded: {X.shape[0]} images of shape {X.shape[1:]}")

    # Split into training and validation sets (80% train, 20% validation)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Training set: {X_train.shape[0]} images")
    print(f"Validation set: {X_val.shape[0]} images")

    # Create data augmentation pipeline to improve generalization
    data_augmentation = tf.keras.Sequential([
        # Flip augmentations
        layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
        # Rotation augmentation - slight rotations to account for imaging variations
        layers.experimental.preprocessing.RandomRotation(0.2),
        # Contrast augmentation - helps model be robust to imaging conditions
        layers.experimental.preprocessing.RandomContrast(0.2),
    ], name="data_augmentation")

    # Build and compile model
    model = build_encoder_only_cnn(input_shape=(512, 512, 1), num_classes=4)

    # Model summary - print architectural details
    model.summary()

    # Calculate class weights to handle imbalanced data
    # Sum the occurrences of each class across all training masks
    class_weights = np.sum(y_train, axis=(0, 1, 2))  # Sum over samples, height, width
    # Invert to give higher weight to less frequent classes
    class_weights = 1.0 / (class_weights + 1e-6)  # Add epsilon to avoid division by zero
    # Normalize weights to sum to number of classes
    class_weights = class_weights / np.sum(class_weights) * 4

    print(f"Class weights: {class_weights}")

    # Compile model with weighted loss and metrics
    model.compile(
        optimizer=Adam(learning_rate=1e-4),  # Start with conservative learning rate
        loss=weighted_categorical_crossentropy(class_weights),
        metrics=[
            'accuracy',  # Overall pixel accuracy
            tf.keras.metrics.MeanIoU(num_classes=4),  # Intersection over Union
            tf.keras.metrics.Recall(),  # Important for smaller particle detection
        ]
    )

    # Define callbacks for training process monitoring and management
    callbacks = [
        # Save best model based on validation performance
        tf.keras.callbacks.ModelCheckpoint(
            'best_battery_segmentation_model.h5',
            save_best_only=True,
            monitor='val_mean_io_u',
            mode='max'
        ),
        # Stop training early if validation metrics plateau
        tf.keras.callbacks.EarlyStopping(
            patience=10,  # Number of epochs to wait for improvement 
            restore_best_weights=True,
            monitor='val_mean_io_u',
            mode='max'
        ),
        # Reduce learning rate when performance plateaus
        tf.keras.callbacks.ReduceLROnPlateau(
            factor=0.1,  # Reduce LR by factor of 10
            patience=5,  # Wait 5 epochs before reducing
            min_lr=1e-6  # Don't go below this LR
        ),
        # Log training metrics for visualization
        tf.keras.callbacks.TensorBoard(
            log_dir='./logs',
            histogram_freq=1  # Log histograms of weights each epoch
        )
    ]

    # Apply data augmentation during training
    def train_generator(X_train, y_train, batch_size=8):
        """Generate batches of augmented training data."""
        while True:
            # Randomly select batch_size samples
            indices = np.random.choice(len(X_train), batch_size)
            batch_x = X_train[indices]
            batch_y = y_train[indices]

            # Apply augmentation to images only (not masks)
            batch_x_aug = data_augmentation(batch_x)

            yield batch_x_aug, batch_y

    # Train model
    # Use generator for data augmentation on the fly
    train_gen = train_generator(X_train, y_train, batch_size=8)
    steps_per_epoch = len(X_train) // 8

    history = model.fit(
        train_gen,
        steps_per_epoch=steps_per_epoch,
        validation_data=(X_val, y_val),
        epochs=50,  # Maximum number of epochs (early stopping may reduce this)
        callbacks=callbacks,
        verbose=1  # Show progress bar
    )

    # Plot training history
    plt.figure(figsize=(12, 4))

    # Plot loss
    plt.subplot(1, 3, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Plot accuracy
    plt.subplot(1, 3, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # Plot IoU
    plt.subplot(1, 3, 3)
    plt.plot(history.history['mean_io_u'], label='Training IoU')
    plt.plot(history.history['val_mean_io_u'], label='Validation IoU')
    plt.title('Mean IoU Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Mean IoU')
    plt.legend()

    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300)
    plt.close()

    # Save final model
    model.save('battery_sem_segmentation_model.h5')
    print("Training complete! Model saved to 'battery_sem_segmentation_model.h5'")

    # Generate example predictions for visualization
    print("Generating example predictions...")
    n_samples = min(5, len(X_val))
    predictions = model.predict(X_val[:n_samples])

    visualize_results(X_val[:n_samples], y_val[:n_samples], predictions,
                      save_path='example_predictions.png')


def visualize_results(images, true_masks, predictions, save_path=None):
    """
    Visualize segmentation results compared to ground truth.

    Parameters
    ----------
    images : array_like
        Input images of shape (n_samples, height, width, 1)
    true_masks : array_like
        Ground truth masks of shape (n_samples, height, width, n_classes)
    predictions : array_like
        Model predictions of shape (n_samples, height, width, n_classes)
    save_path : str, optional
        Path to save the visualization. If None, only displays the plot.

    Notes
    -----
    Creates a visualization with 3 columns:
    1. Original SEM image
    2. Ground truth segmentation mask
    3. Predicted segmentation mask
    """
    n_samples = len(images)

    # Define colors for different material classes
    class_colors = [
        [255, 0, 0],  # LFP - Red
        [0, 0, 255],  # Carbon Black - Blue
        [255, 255, 0],  # NaCl - Yellow
        [0, 255, 0]  # Void - Green
    ]
    class_names = ["LFP", "Carbon Black", "NaCl", "Void"]

    fig, axes = plt.subplots(n_samples, 3, figsize=(15, 5 * n_samples))

    for i in range(n_samples):
        # Original image
        axes[i, 0].imshow(images[i, ..., 0], cmap='gray')
        axes[i, 0].set_title('SEM Image')
        axes[i, 0].axis('off')

        # Ground truth mask
        gt_mask = np.argmax(true_masks[i], axis=-1)
        gt_colored = np.zeros((gt_mask.shape[0], gt_mask.shape[1], 3), dtype=np.uint8)
        for c in range(len(class_colors)):
            gt_colored[gt_mask == c] = class_colors[c]
        axes[i, 1].imshow(gt_colored)
        axes[i, 1].set_title('Ground Truth')
        axes[i, 1].axis('off')

        # Prediction
        pred_mask = np.argmax(predictions[i], axis=-1)
        pred_colored = np.zeros((pred_mask.shape[0], pred_mask.shape[1], 3), dtype=np.uint8)
        for c in range(len(class_colors)):
            pred_colored[pred_mask == c] = class_colors[c]
        axes[i, 2].imshow(pred_colored)
        axes[i, 2].set_title('Prediction')
        axes[i, 2].axis('off')

    # Add a legend
    handles = [plt.Rectangle((0, 0), 1, 1, color=[c / 255 for c in color]) for color in class_colors]
    fig.legend(handles, class_names, loc='lower center', ncol=4, bbox_to_anchor=(0.5, 0))

    plt.tight_layout(rect=[0, 0.05, 1, 1])  # Adjust layout to make room for legend
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
    plt.show()


if __name__ == "__main__":
    main()
