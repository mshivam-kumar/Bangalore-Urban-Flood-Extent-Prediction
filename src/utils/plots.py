import time
import os
import numpy as np

import matplotlib
matplotlib.use('Agg')  # Non-GUI backend for headless environments
import matplotlib.pyplot as plt

import torch
import pandas as pd
from skimage.transform import resize
import matplotlib.colors as mcolors
from scipy import ndimage
from matplotlib.patches import Patch
from sklearn.metrics import confusion_matrix

import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

from src.utils.log_config import setup_log #or from log_config import setup_log # since plots.py is in the same directory as log_config.py
from config import project_config



def plot_sample_visualizations(predictions, targets, masks, sample_indices=None, output_dir = None):
    """
    Plot predicted vs. ground truth water depths for specified samples.

    Args:
        predictions (np.ndarray): Denormalized predictions (N, 1, H, W) or (N, H, W).
        targets (np.ndarray): Denormalized ground truth (N, 1, H, W) or (N, H, W).
        masks (np.ndarray): Masks (N, 1, H, W) or (N, H, W).
        output_dir (str): Directory to save the plots.
        sample_indices (list, optional): Indices of samples to plot. If None, plot all.
    """
    log = setup_log(__name__, project_config.MODEL_PIPELINE_LOG)

    if sample_indices is None:
        sample_indices = range(len(predictions))

    for idx in sample_indices:
        # Squeeze channel dimension if present (e.g., (1, H, W) -> (H, W))
        pred = np.squeeze(predictions[idx])  # Handles (1, H, W) or (H, W)
        target = np.squeeze(targets[idx])    # Handles (1, H, W) or (H, W)

        # Verify shape
        if pred.ndim != 2 or target.ndim != 2:
            log.error(f"Invalid shape for sample {idx}: pred={pred.shape}, target={target.shape}")
            continue

        plt.figure(figsize=(15, 5))
        # Predicted water depth
        plt.subplot(1, 2, 1)
        plt.imshow(pred, cmap='Blues', vmin=0, 
                   vmax=max(pred.max(), 0.01))
        plt.title(f"Predicted Water Depth (Sample {idx})")
        plt.colorbar(label='Depth')
        plt.axis('off')
        # Ground truth water depth
        plt.subplot(1, 2, 2)
        plt.imshow(target, cmap='Blues', vmin=0, 
                   vmax=max(target.max(), 0.01))
        plt.title(f"Ground Truth Water Depth (Sample {idx})")
        plt.colorbar(label='Depth')
        plt.axis('off')
        plt.tight_layout()

        eval_visualization_dir = output_dir +  '/eval_sample_visualizations'
        os.makedirs(eval_visualization_dir, exist_ok=True)
        output_file = os.path.join(eval_visualization_dir, f'evaluation_starting_sample_patch_{idx}.png')
        plt.savefig(output_file)
        log.info(f"Saved sample visualization to {output_file}")

        # plt.show()

        plt.close()


# Refine below plot function
def train_and_validation_loss_curve(epoch, train_loss, val_loss, output_dir = None):
    log = setup_log(__name__, project_config.MODEL_PIPELINE_LOG)

    plt.figure(figsize=(10, 6))
    # plt.plot(history['train_loss'], label='Training Loss')
    # plt.plot(history['val_loss'], label='Validation Loss')
    plt.plot(train_loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    # Save the figure
    plt.tight_layout()
    # train_and_validation_loss_plot_dir = output_dir + '/train_validation_loss_plots'  
    # Now we can't use + like used in above line. Since we are using project_config paths those are not strings. So to concatenate just use / between two paths. Or we can convert to str() before concatenation
    #  The error is because output_dir is a PosixPath object (from Python’s pathlib), and in Python you can’t concatenate it with a string using +.
    train_and_validation_loss_plot_dir = output_dir / 'train_validation_loss_plots'
    os.makedirs(train_and_validation_loss_plot_dir, exist_ok=True)
    plt.savefig(train_and_validation_loss_plot_dir / f'train_val_loss_curve_{epoch}.png')
    log.info(f"Saved train and validation loss plot to {train_and_validation_loss_plot_dir / f'train_val_loss_curve_{epoch}.png'}")

    # plt.show()
    
    plt.close()



def test_loss_curve(epoch, test_loss_list, x_cord_text = "Batch Index", num_of_pixels=0, output_dir = None):
    log = setup_log(__name__, project_config.MODEL_PIPELINE_LOG)

    plt.figure(figsize=(10, 6))
    if(x_cord_text == "Batch Index"):
        plt.plot(range(len(test_loss_list)), test_loss_list, label='Per-Batch MSE Loss')

    else: # "Pixel Index"
        plt.plot(range(len(test_loss_list)), test_loss_list, label='Per-Pixel MSE Loss')

    # plt.plot(range(len(batch_losses)), batch_losses, marker='o', linestyle='-', label='Per-Batch MSE Loss')
    plt.xlabel(x_cord_text)
    plt.ylabel('Average MSE Loss')
    if(x_cord_text == "Batch Index"):
        plt.legend() # With pixels data, getting warning: creating legend with loc="best" can be slow with large amounts of data.
    plt.title('Evaluation Loss Curve')


    test_loss_plot_dir = output_dir + '/test_loss_plots'
    os.makedirs(test_loss_plot_dir, exist_ok=True)
    

    if(x_cord_text == "Batch Index"):
        batches_eval_loss_plot_dir = test_loss_plot_dir + f'/batches_evaluation_loss_plots'
        os.makedirs(batches_eval_loss_plot_dir, exist_ok=True)
        # plt.savefig(batches_eval_loss_plot_dir + f'/batches_eval_loss_epoch_{epoch}.png')
        plt.savefig(batches_eval_loss_plot_dir + f'/batches_eval_loss.png')
        log.info(f"Saved batch evaluation loss plot to {batches_eval_loss_plot_dir + f'/batches_eval_loss_epoch_{epoch}.png'}")
    else: # "Pixel Index"
        pixels_eval_loss_plot_dir = test_loss_plot_dir + f'/pixels_evaluation_loss_plots'
        os.makedirs(pixels_eval_loss_plot_dir, exist_ok=True)
        # plt.savefig(pixels_eval_loss_plot_dir + f'/{num_of_pixels}_pixels_eval_loss_epoch_{epoch}.png')
        plt.savefig(pixels_eval_loss_plot_dir + f'/{num_of_pixels}_pixels_eval_loss_epoch.png')
        log.info(f"Saved pixel evaluation loss plot to {pixels_eval_loss_plot_dir + f'/{num_of_pixels}_pixels_eval_loss_epoch.png'}")

    # plt.show()

    plt.close()




def pixel_image_plot(actual, predicted, number_of_images = 4, output_dir = None):
    """
    Visualize actual and predicted pixel values as images with a difference map.

    Args:
        actual (array-like or tensor): Actual pixel values (2D, 3D, or 4D).
        predicted (array-like or tensor): Predicted pixel values (2D, 3D, or 4D).
        save_path (str): Path to save the plot.
        number_of_images (int): Number of images to plot (default: 4).
    """
    log = setup_log(__name__, project_config.MODEL_PIPELINE_LOG)

    # Convert tensors to NumPy arrays
    if isinstance(actual, torch.Tensor):
        actual = actual.cpu().numpy()
    if isinstance(predicted, torch.Tensor):
        predicted = predicted.cpu().numpy()

    # Convert to NumPy arrays
    actual = np.array(actual)
    predicted = np.array(predicted)

    # Debugging: Print initial shapes
    log.info(f"Initial shapes - actual: {actual.shape}, predicted: {predicted.shape}")

    # Squeeze channel dimension if present in predicted or actual
    if actual.ndim == 4 and actual.shape[1] == 1:
        actual = actual.squeeze(1)
    if predicted.ndim == 4 and predicted.shape[1] == 1:
        predicted = predicted.squeeze(1)

    # Debugging: Print shapes after squeezing
    log.info(f"Shapes after squeezing - actual: {actual.shape}, predicted: {predicted.shape}")

    # Verify shapes match
    if actual.shape != predicted.shape:
        raise ValueError(f"Actual and predicted must have the same shape. Got {actual.shape} and {predicted.shape}")

    # Ensure number_of_images does not exceed available images
    number_of_images = min(number_of_images, actual.shape[0])

    fig = plt.figure(figsize=(18, 4 * number_of_images))
    gs = fig.add_gridspec(number_of_images, 3, height_ratios=[1] * number_of_images)

    # Plot image comparisons
    for i in range(number_of_images):
        pred = predicted[i]
        target = actual[i]

        # Calculate difference map
        diff = np.abs(pred - target)

        # Display prediction
        ax1 = fig.add_subplot(gs[i, 0])
        im1 = ax1.imshow(pred, cmap='Blues')
        ax1.set_title(f"Prediction {i+1} (meters)")
        ax1.axis('off')
        plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

        # Display ground truth
        ax2 = fig.add_subplot(gs[i, 1])
        im2 = ax2.imshow(target, cmap='Blues')
        ax2.set_title(f"Ground Truth {i+1} (meters)")
        ax2.axis('off')
        plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

        # Threshold the diff array to make it binary (0 or 1)
        threshold = 0.1
        binary_diff = (diff > threshold).astype(int)
        # Display difference
        ax3 = fig.add_subplot(gs[i, 2])
        im3 = ax3.imshow(binary_diff, cmap='hot')
        ax3.set_title(f"Absolute Difference (> {threshold} m)")
        ax3.axis('off')
        plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)

    plt.tight_layout()



    pixel_image_plot_dir = output_dir + '/pixel_image_plots'
    os.makedirs(pixel_image_plot_dir, exist_ok=True)

    plt.savefig(pixel_image_plot_dir + f"/till_{number_of_images}_images.png")
    log.info(f"Saved batch evaluation loss plot to {pixel_image_plot_dir}/till_{number_of_images}images.png")

    # plt.show()

    plt.close()


def pixel_scatter_plot(all_targets, all_preds, max_points=100, output_dir = None):
    log = setup_log(__name__, project_config.MODEL_PIPELINE_LOG)

    # Convert tensors to NumPy and flatten
    if isinstance(all_targets, torch.Tensor):
        all_targets = all_targets.cpu().numpy()
    if isinstance(all_preds, torch.Tensor):
        all_preds = all_preds.cpu().numpy()
    actual = all_targets.flatten()
    predicted = all_preds.flatten()

    # Verify inputs
    if actual.shape != predicted.shape:
        raise ValueError(f"Actual and predicted must have the same shape. Got {actual.shape} and {predicted.shape}")
    if actual.ndim != 1 or predicted.ndim != 1:
        raise ValueError("Actual and predicted must be 1D arrays")

    # Downsample for large data
    # max_points = 10000
    if len(actual) > max_points:
        np.random.seed(42)
        indices = np.random.choice(len(actual), size=max_points, replace=False)
        actual = actual[indices]
        predicted = predicted[indices]
        log.info(f"Downsampled data to {max_points} points for plotting")

    # Create scatter plot
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=actual, y=predicted, color='blue', label='Predictions', s=10, alpha=0.5)
    plt.plot([actual.min(), actual.max()], [actual.min(), actual.max()], color='green', linestyle='--', label='Ideal (Actual = Predicted)')

    # Customize plot
    plt.xlabel('Actual Pixel Values')
    plt.ylabel('Predicted Pixel Values')
    plt.title(f'Scatter Plot: Predicted vs Actual Pixel Values. Downsampled to {max_points} pixel points')
    plt.legend()
    plt.grid(True)

    pixel_scatter_plot_dir = output_dir + '/pixel_scatter_plots'
    os.makedirs(pixel_scatter_plot_dir, exist_ok=True)
    plt.savefig(pixel_scatter_plot_dir + f"/till_{max_points}_points.png")
    log.info(f"Saved batch evaluation loss plot to {pixel_scatter_plot_dir} + /till_{max_points}_points.png")

    # plt.show()

    plt.close()


def MAE_plot(all_targets, all_preds, output_dir = None):
    log = setup_log(__name__, project_config.MODEL_PIPELINE_LOG)

    # Convert to NumPy
    # actual = all_targets.cpu().numpy()  
    # predicted = all_preds.cpu().numpy()


    actual = to_numpy(all_targets)  # to_numpy function is defined above
    predicted = to_numpy(all_preds)
    # log.info("Inside MAE_plot: actual.shape:", actual.shape)
    # log.info("Inside MAE_plot: predicted.shape:", predicted.shape)

    # Calculate MAE per image
    mae_per_image = np.mean(np.abs(predicted - actual), axis=(1, 2, 3))  # Mean over channels, height, width

    # Plot MAE
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(mae_per_image)), mae_per_image, marker='o', color='blue', label='MAE')
    plt.xlabel('Image Index')
    plt.ylabel('Mean Absolute Error')
    plt.title('MAE per Image')
    plt.legend()
    plt.grid(True)

    error_plot_dir = output_dir + '/error_plots'
    os.makedirs(error_plot_dir, exist_ok=True)
    plt.savefig(error_plot_dir + f'/mae.png')
    log.info(f"Saved batch evaluation loss plot to {error_plot_dir + f'/mae.png'}")

    # plt.show()

    plt.close()


def error_heatmap_plot(actual, predicted, batch_index=0, output_dir = None):
    """
    Visualize absolute error between actual and predicted as a binary heatmap.

    Args:
        actual (array-like or tensor): Actual pixel values (2D, 3D, or 4D).
        predicted (array-like or tensor): Predicted pixel values (2D, 3D, or 4D).
        save_path (str): Path to save the plot.
        batch_index (int): Index of the image to plot if batched (default: 0).
    """
    log = setup_log(__name__, project_config.MODEL_PIPELINE_LOG)

    # Convert tensors to NumPy arrays
    if isinstance(actual, torch.Tensor):
        actual = actual.cpu().numpy()
    if isinstance(predicted, torch.Tensor):
        predicted = predicted.cpu().numpy()

    # Convert to NumPy arrays
    actual = np.array(actual)
    predicted = np.array(predicted)

    # Debugging: Print initial shapes
    log.info(f"Initial shapes - actual: {actual.shape}, predicted: {predicted.shape}")

    # Squeeze channel dimension if present
    if actual.ndim == 4 and actual.shape[1] == 1:
        actual = actual.squeeze(1)
    if predicted.ndim == 4 and predicted.shape[1] == 1:
        predicted = predicted.squeeze(1)

    # Debugging: Print shapes after squeezing
    log.info(f"Shapes after squeezing - actual: {actual.shape}, predicted: {predicted.shape}")

    # Verify shapes match
    if actual.shape != predicted.shape:
        raise ValueError(f"Actual and predicted must have the same shape. Got {actual.shape} and {predicted.shape}")

    # Handle input dimensions
    if actual.ndim == 3:
        # 3D: (batch, height, width)
        if batch_index >= actual.shape[0]:
            raise ValueError(f"batch_index {batch_index} exceeds batch size {actual.shape[0]}")
        actual = actual[batch_index]  # Shape: (256, 256)
        predicted = predicted[batch_index]
    elif actual.ndim == 2:
        # 2D: (height, width)
        pass
    else:
        raise ValueError("Expected 2D or 3D inputs after squeezing.")

    # Calculate absolute error
    abs_diff = np.abs(predicted - actual)

    # Plot error heatmap
    plt.figure(figsize=(6, 6))
    # Threshold the diff array to make it binary (0 or 1)
    threshold = 0.1  # Adjust as needed
    binary_diff = (abs_diff > threshold).astype(int)
    im = plt.imshow(binary_diff, cmap='hot', vmin=0, vmax=1)
    plt.title(f'Absolute Error Heatmap (>{threshold} m)')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.colorbar(im, label='Error (0: ≤0.1m, 1: >0.1m)')
    plt.grid(False)
    plt.tight_layout()

    # Save the plot
    error_plot_dir = output_dir + '/error_plots'
    os.makedirs(error_plot_dir, exist_ok=True)
    plt.savefig(error_plot_dir + f'/error_heatmap.png')
    log.info(f"Saved batch evaluation loss plot to {error_plot_dir + f'/error_heatmap.png'}")

    # plt.show()

    plt.close()

def to_numpy(x):
    if hasattr(x, 'cpu'):
        return x.cpu().numpy() # if already in numpy then will give error that 'numpy.ndarray' object has no attribute 'cpu'
        # return x.cpu().numpy().flatten() # if already in numpy then will give error that 'numpy.ndarray' object has no attribute 'cpu'
    return np.asarray(x)

def error_histogram_plot(all_targets, all_preds, output_dir = None):
    log = setup_log(__name__, project_config.MODEL_PIPELINE_LOG)

    # Convert to NumPy and flatten
    actual = to_numpy(all_targets).flatten() # to_numpy function is defined above
    predicted = to_numpy(all_preds).flatten()
    residuals = predicted - actual

    # Create figure with three subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

    # Plot histograms
    ax1.hist(actual, bins=50, alpha=0.5, color='green', label='Actual')
    ax1.set_title('Actual Pixel Values')
    ax1.set_xlabel('Value')
    ax1.set_ylabel('Frequency')
    ax1.legend()

    ax2.hist(predicted, bins=50, alpha=0.5, color='blue', label='Predicted')
    ax2.set_title('Predicted Pixel Values')
    ax2.set_xlabel('Value')
    ax2.set_ylabel('Frequency')
    ax2.legend()

    ax3.hist(residuals, bins=50, alpha=0.5, color='red', label='Residuals')
    ax3.set_title('Residuals (Predicted - Actual)')
    ax3.set_xlabel('Residual')
    ax3.set_ylabel('Frequency')
    ax3.legend()

    # Adjust layout
    plt.tight_layout()


    # Save the plot
    error_plot_dir = output_dir + '/error_plots'
    os.makedirs(error_plot_dir, exist_ok=True)
    plt.savefig(error_plot_dir + f'/error_histogram.png')
    log.info(f"Saved batch evaluation loss plot to {error_plot_dir + f'/error_histogram.png'}")
    
    # plt.show()

    plt.close()


def plot_metrics_barplot(metrics, output_dir = None):
    log = setup_log(__name__, project_config.MODEL_PIPELINE_LOG)

    plt.figure(figsize=(10, 6))
    plt.bar(metrics.keys(), metrics.values(), color='skyblue')
    plt.ylabel('Metric Value')
    plt.title('Model Evaluation Metrics Comparison')
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    metrics_bar_plot_dir = output_dir + '/metric_bar_plots'
    os.makedirs(metrics_bar_plot_dir, exist_ok=True)
    plt.savefig(metrics_bar_plot_dir + f'/metrics_bar_plot.png')
    log.info(f"Saved metrics bar plot to {metrics_bar_plot_dir + f'/metric_bar_plot.png'}")
    
    # plt.show()    
    plt.close()


def confusion_matrix_plot_normal(all_targets, all_preds, threshold, output_dir = None):
    """
    Plot a simple confusion matrix for binary classification (Flood vs Non-Flood) with percentages.
    
    Parameters:
    -----------
    all_targets : numpy.ndarray or torch.Tensor
        Ground truth array
    all_preds : numpy.ndarray or torch.Tensor
        Prediction array
    threshold : float
        Threshold for binary classification
    save_path : str
        Path to save the confusion matrix plot
    """
    log = setup_log(__name__, project_config.MODEL_PIPELINE_LOG)

    # Convert to NumPy arrays if inputs are PyTorch tensors
    if isinstance(all_targets, torch.Tensor):
        all_targets = all_targets.cpu().numpy()
    if isinstance(all_preds, torch.Tensor):
        all_preds = all_preds.cpu().numpy()

    # Convert to binary and flatten
    actual_binary = (all_targets > threshold).flatten()  # Shape: (17039360,)
    predicted_binary = (all_preds > threshold).flatten()  # Shape: (17039360,)

    # Compute confusion matrix
    # Threshold
    pred_binary = (all_preds >= threshold).astype(np.uint8)
    target_binary = (all_targets >= threshold).astype(np.uint8)

    cm = confusion_matrix(target_binary.ravel(), pred_binary.ravel(), labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    
    # Masks (for additional checks, though not needed for the confusion matrix plot)
    tn_mask = (pred_binary == 0) & (target_binary == 0)
    fp_mask = (pred_binary == 1) & (target_binary == 0)
    fn_mask = (pred_binary == 0) & (target_binary == 1)
    tp_mask = (pred_binary == 1) & (target_binary == 1)
    
    counts = {'TN': tn_mask.sum(), 'FP': fp_mask.sum(), 'FN': fn_mask.sum(), 'TP': tp_mask.sum()}
    total_pixels = all_preds.size
    
    # Metrics (computed again, redundant but kept as per original code)
    accuracy = (tp + tn) / total_pixels * 100
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    log.info(f"Threshold: {threshold}")
    log.info(f"Accuracy: {accuracy:.2f}%")
    log.info(f"Precision: {precision:.4f}")
    log.info(f"Recall: {recall:.4f}")
    log.info(f"F1 Score: {f1_score:.4f}")

    # Compute percentages for the confusion matrix
    total_pixels_cm = cm.sum()  # Should match all_preds.size
    cm_percent = (cm / total_pixels_cm) * 100  # Convert counts to percentages

    # # Plot confusion matrix with percentages
    # plt.figure(figsize=(8, 6))
    # sns.heatmap(cm_percent, annot=True, fmt='.2f', cmap='Blues', 
    #             xticklabels=['Non-Flood', 'Flood'], yticklabels=['Non-Flood', 'Flood'])
    

    # Create a custom annotation array with % signs
    cm_percent_annot = np.array([[f"{val:.2f}% ({int((val/100)*total_pixels_cm)} pixels)" for val in row] for row in cm_percent])

    # Plot confusion matrix with percentages
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_percent, annot=cm_percent_annot, fmt='', cmap='Blues', 
                xticklabels=['Non-Flood', 'Flood'], yticklabels=['Non-Flood', 'Flood'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f"""Confusion Matrix (Threshold = {threshold})\nEvaluated on all predictions:{len(all_preds)} patches
              \nAccuracy:{accuracy:.2f} %, Precision: {precision:.2f}, Recall: {recall:.2f}, F1 Score: {f1_score:.2f}
                \nThreshold:{threshold}""")
    plt.tight_layout()


    confusion_matrix_plot_dir = output_dir + '/confusion_matrix_plots/normal_confusion_matrix'
    os.makedirs(confusion_matrix_plot_dir, exist_ok=True)
    plt.savefig(confusion_matrix_plot_dir + f'/cm_thres_{threshold}_normal.png')
    log.info(f"Saved batch evaluation loss plot to {confusion_matrix_plot_dir + f'/cm_thres_{threshold}_normal.png'}")

    # plt.show()
    plt.close()



def plot_confusion_matrix_all_classes_image_pixel_based_visualization(target, pred, input_image=None, threshold=0.5, title_prefix="", 
                                     block_size=8, grid_alpha=0.7, color_alpha=0.7, highlight_edges=False,
                                     number_of_patches = 25, patch_start_idx =0, patch_end_idx =5,  output_dir=None):
    """
    Plot confusion matrix visualization with larger pixel blocks and highlighted boundaries.
    Optimized for large images.
    
    Parameters:
    -----------
    pred : numpy.ndarray
        Prediction array
    target : numpy.ndarray
        Ground truth array
    input_image : numpy.ndarray, optional
        Background image to overlay results on
    threshold : float, default=0.5
        Threshold for binary classification
    output_dir : str, optional
        Directory to save the output image
    title_prefix : str, default=""
        Prefix for the title
    block_size : int, default=8
        Size of pixel blocks for downsampling (higher = faster but less detailed)
    grid_alpha : float, default=0.7
        Opacity of the grid lines
    color_alpha : float, default=0.7
        Opacity of the class colors
    highlight_edges : bool, default=False
        Whether to highlight edges between different regions
    """
    log = setup_log(__name__, project_config.MODEL_PIPELINE_LOG)

    start_time = time.time()
    log.info("Starting confusion matrix visualization...")
    log.info(f"Input shapes - Target: {target.shape}, Pred: {pred.shape}, Input image: {input_image.shape if input_image is not None else 'None'}")

    target = to_numpy(target)
    pred = to_numpy(pred)  # Convert to NumPy if needed
    input_image = to_numpy(input_image) if input_image is not None else target  # Convert to NumPy if needed
    
    # Default to using the target as the background if none provided
    # if input_image is None:
    #     input_image = target
    # else:
    #     input_image = to_numpy(input_image)  # Convert to NumPy if needed
        
    # Validate inputs
    if not isinstance(pred, np.ndarray) or not isinstance(target, np.ndarray) or (input_image is not None and not isinstance(input_image, np.ndarray)):
        log.error("Inputs must be numpy arrays.")
        raise ValueError("Inputs must be numpy arrays.")
    
    if pred.shape != target.shape or (input_image is not None and pred.shape != input_image.shape):
        log.error(f"Shape mismatch: pred {pred.shape}, target {target.shape}, input_image {input_image.shape}")
        raise ValueError(f"Shape mismatch: pred {pred.shape}, target {target.shape}, input_image {input_image.shape}")
    
    # Ensure 2D arrays
    if pred.ndim == 3 and pred.shape[0] == 1:
        pred = pred[0]
    elif pred.ndim != 2:
        log.error(f"Expected pred to be 2D or 3D with single channel, got shape {pred.shape}")
        raise ValueError(f"Expected pred to be 2D or 3D with single channel, got shape {pred.shape}")
    
    if target.ndim == 3 and target.shape[0] == 1:
        target = target[0]
    elif target.ndim != 2:
        log.error(f"Expected target to be 2D or 3D with single channel, got shape {target.shape}")
        raise ValueError(f"Expected target to be 2D or 3D with single channel, got shape {target.shape}")
    
    if input_image.ndim == 3 and input_image.shape[0] == 1:
        input_image = input_image[0]
    elif input_image.ndim != 2:
        log.error(f"Expected input_image to be 2D or 3D with single channel, got shape {input_image.shape}")
        raise ValueError(f"Expected input_image to be 2D or 3D with single channel, got shape {input_image.shape}")
    
    # Check for invalid values
    if np.any(np.isnan(pred)) or np.any(np.isnan(target)):
        log.error("NaN values detected in pred or target")
        raise ValueError("NaN values detected in pred or target")
    if np.any(np.isinf(pred)) or np.any(np.isinf(target)):
        log.error("Infinite values detected in pred or target")
        raise ValueError("Infinite values detected in pred or target")
    
    log.info(f"Input shapes - Pred: {pred.shape}, Target: {target.shape}, Input image: {input_image.shape}")
    log.info(f"Pred min/max: {pred.min():.4f}/{pred.max():.4f}, Target min/max: {target.min():.4f}/{target.max():.4f}")
    
    # Threshold
    pred_binary = (pred >= threshold).astype(np.uint8)
    target_binary = (target >= threshold).astype(np.uint8)
    
    # Confusion matrix
    cm = confusion_matrix(target_binary.ravel(), pred_binary.ravel(), labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    
    # Masks
    tn_mask = (pred_binary == 0) & (target_binary == 0)
    fp_mask = (pred_binary == 1) & (target_binary == 0)
    fn_mask = (pred_binary == 0) & (target_binary == 1)
    tp_mask = (pred_binary == 1) & (target_binary == 1)
    
    counts = {'TN': tn_mask.sum(), 'FP': fp_mask.sum(), 'FN': fn_mask.sum(), 'TP': tp_mask.sum()}
    total_pixels = pred.size
    
    # Metrics
    accuracy = (tp + tn) / total_pixels * 100
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    

    log.info(f"Threshold: {threshold}")
    log.info(f"Accuracy: {accuracy:.2f}%")
    log.info(f"Precision: {precision:.4f}")
    log.info(f"Recall: {recall:.4f}")
    log.info(f"F1 Score: {f1_score:.4f}")
    
    # Setup for plotting
    log.info("Pred image shape: %s", pred.shape)
    H, W = pred.shape
    fig, axes = plt.subplots(2, 2, figsize=(20, 20), dpi=100)
    axes = axes.ravel()
    
    class_colors = {
        'TN': 'blue',
        'FP': 'red',
        'FN': 'yellow',
        'TP': 'green'
    }
    
    cases = [
        ('True Negatives', tn_mask, 'TN'),
        ('False Positives', fp_mask, 'FP'),
        ('False Negatives', fn_mask, 'FN'),
        ('True Positives', tp_mask, 'TP')
    ]
    
    # Normalize input image to [0, 1] for consistent display
    input_image_display = input_image.astype(np.float32)
    if input_image_display.max() > 1.0:
        input_image_display /= 255.0
    
    # For edge highlighting
    if highlight_edges:
        # Create edge detection kernel
        edge_kernel = np.array([
            [1, 1, 1],
            [1, 0, 1],
            [1, 1, 1]
        ])

    plt.suptitle(f"""Image is reconstructed with {number_of_patches} and patch indices from {patch_start_idx} to {patch_end_idx}.\n
                 Original image shape: {(H,W)}.\n
                 For visualization the image is resized to: {(H // block_size, W // block_size)}.
                 The metrics (accuracy, precision, recall, F1 score) are computed using the full-resolution image having shape {(H,W)}
                \nAccuracy:{accuracy:.2f} %, Precision: {precision:.2f}, Recall: {recall:.2f}, F1 Score: {f1_score:.2f}
                \nThreshold:{threshold}""", fontsize=17, y=0.98)

    # Add "Actual" and "Predicted" labels for the entire 2x2 grid
    fig.text(0.5, 0.02, 'Predicted', ha='center', va='center', fontsize=16, fontweight='bold')
    fig.text(0.04, 0.5, 'Actual', ha='center', va='center', rotation='vertical', fontsize=16, fontweight='bold')

    # Process each case
    log.info("Processing visualization cases...")
    for idx, (title, mask, key) in enumerate(cases):
        ax = axes[idx]
        
        # Create a downsampled version for faster visualization if image is large
        if max(H, W) > 500:  # If the image is large
            # Calculate new dimensions, ensuring they don't go below 100px
            new_h = max(100, H // block_size)
            new_w = max(100, W // block_size)
            
            # Resize input image and mask using block-averaging
            log.info(f"Resizing image from {H}x{W} to {new_h}x{new_w} for {title}")
            input_small = resize(input_image_display, (new_h, new_w), 
                               anti_aliasing=True, preserve_range=True)
            
            # Resize mask using nearest neighbor to preserve binary nature
            mask_small = resize(mask.astype(float), (new_h, new_w), 
                              order=0, anti_aliasing=False, preserve_range=True)
            mask_small = mask_small > 0.5  # Convert back to binary
            
            log.info(f"Resized image from {H}x{W} to {new_h}x{new_w} for {title}")
            
            # Detect edges for highlighting if requested
            if highlight_edges:
                edge_mask = ndimage.convolve(mask_small.astype(float), edge_kernel)
                # Edge pixels are those that have neighbors of different class
                edge_mask = (edge_mask > 0) & (edge_mask < 8)
            
            # Create colored visualization
            overlay = np.zeros((new_h, new_w, 4))
            
            # Set colors for non-edge regions
            overlay[mask_small] = (*mcolors.to_rgba(class_colors[key])[:3], color_alpha)
            
            # Highlight edges if requested
            if highlight_edges:
                # Make edges darker and more opaque
                dark_color = mcolors.to_rgba(class_colors[key])
                # Convert to HSV to darken
                import colorsys
                h, s, v = colorsys.rgb_to_hsv(*dark_color[:3])
                dark_color = (*colorsys.hsv_to_rgb(h, s, v * 0.7), 1.0)  # Darker and fully opaque
                overlay[mask_small & edge_mask] = dark_color
            
            # Display
            ax.imshow(input_small, interpolation='nearest', alpha=0.6)
            ax.imshow(overlay, interpolation='nearest')
            
            # Add grid if desired - only show grid on downsized images
            if block_size > 1:
                # Calculate grid lines for blocks, draw every block_size pixels
                grid_x = np.arange(0, new_w, block_size)
                grid_y = np.arange(0, new_h, block_size)
                ax.set_xticks(grid_x - 0.5, minor=True)
                ax.set_yticks(grid_y - 0.5, minor=True)
                ax.grid(which='minor', color='black', linestyle='-', linewidth=0.5, alpha=grid_alpha)
        else:
            # For smaller images, use full resolution
            # Create RGBA visualization
            overlay = np.zeros((H, W, 4))
            overlay[mask] = (*mcolors.to_rgba(class_colors[key])[:3], color_alpha)
            
            # Highlight edges if requested
            if highlight_edges:
                edge_mask = ndimage.convolve(mask.astype(float), edge_kernel)
                edge_mask = (edge_mask > 0) & (edge_mask < 8)
                
                # Make edges darker
                dark_color = mcolors.to_rgba(class_colors[key])
                import colorsys
                h, s, v = colorsys.rgb_to_hsv(*dark_color[:3])
                dark_color = (*colorsys.hsv_to_rgb(h, s, v * 0.7), 1.0)
                overlay[mask & edge_mask] = dark_color
            
            # Display
            ax.imshow(input_image_display, interpolation='nearest', alpha=0.6)
            ax.imshow(overlay, interpolation='nearest')
            
            # Add grid
            ax.set_xticks(np.arange(0, W, block_size) - 0.5, minor=True)
            ax.set_yticks(np.arange(0, H, block_size) - 0.5, minor=True)
            ax.grid(which='minor', color='black', linestyle='-', linewidth=0.3, alpha=grid_alpha)
        
        # Turn off numerical axis ticks
        ax.tick_params(which='both', bottom=False, left=False, labelbottom=False, labelleft=False)
        
        # Add "Flood" and "Not Flood" labels for each subplot based on its position
        if idx == 0:  # TN (top-left): Actual Not Flood (0), Predicted Not Flood (0)
            # ax.set_xlabel('Flood')
            ax.set_ylabel('Not Flood', fontsize=16)
        elif idx == 1:  # FP (top-right): Actual Not Flood (0), Predicted Flood (1)
            pass
        elif idx == 2:  # FN (bottom-left): Actual Flood (1), Predicted Not Flood (0)
            ax.set_xlabel('Not Flood', fontsize=16)
            ax.set_ylabel('Flood', fontsize=16)
        elif idx == 3:  # TP (bottom-right): Actual Flood (1), Predicted Flood (1)
            ax.set_xlabel('Flood',fontsize=16)
        # The patch indice  are the ones which are returned by the evaluate_model function as all_denormalized_predictions and all_denormalized_targets
        # title_prefix = f"Reconstructed image with {number_of_patches} and patch indices from {patch_start_idx} to {patch_end_idx}.\n"

        # Add title and legend
        if(idx == 2 or idx == 3):
            ax.set_title(f"\n{title_prefix}{title} ({(counts[key]/total_pixels)*100:.2f}% pixels)", fontsize=16, pad=20)
        else:
            ax.set_title(f"{title_prefix}{title} ({(counts[key]/total_pixels)*100:.2f}% pixels)", fontsize=16, pad=20)
        patch = Patch(color=class_colors[key], label=f"{key}: {counts[key]} pixels")
        ax.legend(handles=[patch], loc='upper right', fontsize=14)
    
    # Adjust layout to accommodate outer labels
    # plt.tight_layout(rect=[0.06, 0.06, 1, 0.95])
    plt.tight_layout(rect=[0.06, 0.06, 1, 0.90])

    confusion_matrix_plot_dir = output_dir + '/confusion_matrix_plots/Image_based_Confusion_Matrix'
    os.makedirs(confusion_matrix_plot_dir, exist_ok=True)
    # plt.savefig(confusion_matrix_plot_dir + f'/cm_thres_{threshold}.png')
    plt.savefig(confusion_matrix_plot_dir + f'/cm_thres_{threshold}.png', dpi=100)
    log.info(f"Saved batch evaluation loss plot to {confusion_matrix_plot_dir + f'/cm_thres_{threshold}.png'}")

    # plt.show()    
    end_time = time.time()
    log.info(f"Visualization completed in {end_time - start_time:.2f} seconds")
    
    return {
        'masks': {'TP': tp_mask, 'TN': tn_mask, 'FP': fp_mask, 'FN': fn_mask},
        'counts': counts,
        'confusion_matrix': cm,
        'total_pixels': total_pixels,
        'metrics': {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score
        }
    }




def show_image(reconstructed_patches_prediction_image): # reconstructed_90_patches_prediction_image
    plt.imshow(reconstructed_patches_prediction_image)
    plt.title("Reconstructed Image")
    plt.axis('off')
    plt.show()


def reconstruct_image_from_patches(patches, grid_rows, grid_cols, patch_shape=None, patch_indices=None, start_idx=None, end_idx=None, output_dir = None):
# def reconstruct_image_from_patches(patches, grid_rows, grid_cols, patch_shape=None, start_idx=None, end_idx=None, output_dir = None):
    """
    Reconstruct a full image from a list of patches arranged in a grid, optionally selecting a middle range of patches.
    
    Args:
        patches (list or np.ndarray): List or array of patches, each a 2D array of shape (H_patch, W_patch).
        grid_rows (int): Number of rows in the patch grid (e.g., 5).
        grid_cols (int): Number of columns in the patch grid (e.g., 5).
        patch_shape (tuple, optional): Shape of each patch (H_patch, W_patch). If None, inferred from patches.
        patch_indices (list, optional): List of indices to select specific patches. Overrides start_idx, end_idx.
        start_idx (int, optional): Starting index for patch selection (inclusive).
        end_idx (int, optional): Ending index for patch selection (exclusive).
    
    Returns:
        np.ndarray: Reconstructed image, shape (grid_rows * H_patch, grid_cols * W_patch).
    
    Raises:
        ValueError: If patch count, shapes, or indices are inconsistent with grid dimensions.
    """
    log = setup_log(__name__, project_config.MODEL_PIPELINE_LOG)
    
    # Select patches based on indices or range
    if patch_indices is not None:
        if len(patch_indices) != grid_rows * grid_cols:
            log.error(f"Expected {grid_rows * grid_cols} patch indices, got {len(patch_indices)}")
            raise ValueError(f"Expected {grid_rows * grid_cols} patch indices, got {len(patch_indices)}")
        selected_patches = [patches[i] for i in patch_indices]
        log.info(f"Selected patches with indices: {patch_indices}")
    elif start_idx is not None and end_idx is not None:
        if end_idx - start_idx != grid_rows * grid_cols:
            log.error(f"Expected {grid_rows * grid_cols} patches from indices {start_idx}:{end_idx}, got {end_idx - start_idx}")
            raise ValueError(f"Expected {grid_rows * grid_cols} patches from indices {start_idx}:{end_idx}, got {end_idx - start_idx}")
        selected_patches = patches[start_idx:end_idx]
        log.info(f"Selected patches from indices {start_idx}:{end_idx}")
    else:
        if len(patches) != grid_rows * grid_cols:
            log.error(f"Expected {grid_rows * grid_cols} patches, got {len(patches)}")
            raise ValueError(f"Expected {grid_rows * grid_cols} patches, got {len(patches)}")
        selected_patches = patches
        log.info("Using all provided patches")
    
    # Get patch shape
    if patch_shape is None:
        if not selected_patches:
            log.error("Patch list is empty and patch_shape not provided")
            raise ValueError("Patch list is empty and patch_shape not provided")
        patch_shape = selected_patches[0].shape
        log.info(f"Inferred patch shape: {patch_shape}")
    
    if len(patch_shape) > 2:
        log.error(f"Expected 2D patches, got shape {patch_shape}")
        raise ValueError(f"Expected 2D patches, got shape {patch_shape}")
    
    H_patch, W_patch = patch_shape[-2:] if len(patch_shape) > 2 else patch_shape
    
    # Validate and preprocess patches
    selected_patches = [np.squeeze(patch) for patch in selected_patches]
    for i, patch in enumerate(selected_patches):
        if patch.shape != (H_patch, W_patch):
            log.error(f"Patch {i} has shape {patch.shape}, expected {(H_patch, W_patch)}")
            raise ValueError(f"Patch {i} has shape {patch.shape}, expected {(H_patch, W_patch)}")
    
    # Convert patches to NumPy array
    patches_array = np.array(selected_patches)  # Shape: (N, H_patch, W_patch)
    log.info(f"Patches array shape: {patches_array.shape}")
    
    # Reshape patches into grid
    try:
        patches_grid = patches_array.reshape(grid_rows, grid_cols, H_patch, W_patch)
        log.info(f"Reshaped patches into grid: {grid_rows}x{grid_cols}")
    except ValueError as e:
        log.error(f"Cannot reshape {len(selected_patches)} patches into {grid_rows}x{grid_cols} grid: {e}")
        raise ValueError(f"Cannot reshape {len(selected_patches)} patches into {grid_rows}x{grid_cols} grid: {e}")
    
    # Concatenate patches
    rows = []
    for i in range(grid_rows):
        row_patches = patches_grid[i]  # Shape: (grid_cols, H_patch, W_patch)
        row_image = np.concatenate(row_patches, axis=1)  # Shape: (H_patch, grid_cols * W_patch)
        rows.append(row_image)
    
    full_image = np.concatenate(rows, axis=0)  # Shape: (grid_rows * H_patch, grid_cols * W_patch)
    
    log.info(f"Reconstructed image shape: {full_image.shape}")
    

    # Prepare metadata
    title_text = (
        f'Reconstructed Image\n'
        f'Grid: ({grid_rows}x{grid_cols}), Patch: {H_patch}x{W_patch}, '
        f'Start idx: {start_idx}, End idx: {end_idx}'
    )

    # Create output directory
    reconstructed_image_dir = os.path.join(output_dir, 'reconstructed_image_from_patches')
    os.makedirs(reconstructed_image_dir, exist_ok=True)

    output_file = os.path.join(reconstructed_image_dir, f'reconstructed_image_{grid_rows}x{grid_cols}.png')

    # Plot and save with size, title, and color map
    fig, ax = plt.subplots(figsize=(10, 6), dpi=100)  # adjust size and resolution
    ax.imshow(full_image, cmap='Blues')              # use your desired colormap
    ax.set_title(title_text, fontsize=10)
    ax.axis('off')                                    # hide axes
    plt.tight_layout()
    fig.savefig(output_file, bbox_inches='tight', pad_inches=0.2)
    plt.close(fig)

    return full_image