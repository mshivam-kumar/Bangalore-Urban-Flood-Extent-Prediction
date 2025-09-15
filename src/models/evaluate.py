import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from src.models.loss import masked_bce_loss

def evaluate_model(model, test_loader, pad_val, nodata_dict, channel_names):
    """
    Evaluate model on test set and compute multiple metrics
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    test_loss = 0.0
    all_predictions = []
    all_targets = []
    all_masks = []
    
    # Metrics storage
    metrics = {
        'loss': 0.0,
        'accuracy': 0.0,
        'precision': 0.0,
        'recall': 0.0,
        'f1': 0.0,
        'iou': 0.0,
        'dice': 0.0
    }

    # criterion = FocalLoss(alpha=0.75, gamma=2.0, reduction='mean').to(device)

    pos_weight = 1.4
    criterion = None
    if pos_weight is not None:
        criterion = nn.BCEWithLogitsLoss(reduction="none", pos_weight=torch.tensor([pos_weight], device=device))
    else:
        criterion = nn.BCEWithLogitsLoss(reduction='none')


    
    with torch.no_grad():
        for features_dict, labels in tqdm(test_loader, desc="Evaluating"):
            # Select features
            stack_lst = []
            for channel_name in channel_names:
                if channel_name == "Rainfall": # we are handling rainfall separately below
                    continue
                stack_lst.append(features_dict[channel_name])
            
            selected_features = torch.stack(stack_lst, dim=1)
            
            static_features = selected_features
            rainfall = features_dict['Rainfall']

            x_static = static_features.to(device)
            x_rainfall = rainfall.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(x_static, x_rainfall)
            outputs = outputs.squeeze(1)
            
            # Calculate loss
            loss = masked_bce_loss(outputs, labels, x_static, pad_val, nodata_dict, channel_names, criterion)
            test_loss += loss.item()
            
            # Get predictions (sigmoid + threshold)
            probs = torch.sigmoid(outputs)
            predictions = (probs > 0.5).float()
            
            # Create valid mask (ignore padding and nodata)
            valid_mask = create_valid_mask(x_static, pad_val, nodata_dict, channel_names)
            
            # Store for batch metrics calculation
            all_predictions.append(predictions.cpu())
            all_targets.append(labels.cpu())
            all_masks.append(valid_mask.cpu())
    
    # Concatenate all batches
    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    all_masks = torch.cat(all_masks, dim=0)
    
    # Apply mask to ignore invalid pixels
    masked_predictions = all_predictions[all_masks]
    masked_targets = all_targets[all_masks]
    
    # Calculate metrics
    metrics['loss'] = test_loss / len(test_loader)
    metrics.update(calculate_metrics(masked_predictions, masked_targets))
    
    return metrics, all_predictions, all_targets, all_masks

def create_valid_mask(features, pad_val, nodata_dict, channel_names):
    """
    Create mask for valid pixels (ignore padding and nodata)
    """
    B, C, H, W = features.shape
    valid_mask = torch.ones((B, H, W), dtype=torch.bool, device=features.device)
    
    # Pad mask
    valid_mask &= ~(features == pad_val).any(dim=1)
    
    # NoData per channel
    for ch_idx, ch_name in enumerate(channel_names):
        nd_val = nodata_dict.get(ch_name, None)
        if nd_val is not None:
            valid_mask &= ~(features[:, ch_idx] == nd_val)
    
    return valid_mask

def calculate_metrics(predictions, targets):
    """
    Calculate various metrics for binary segmentation
    """
    # Ensure tensors are on CPU and flattened
    predictions = predictions.flatten()
    targets = targets.flatten()
    
    # Remove any remaining invalid values (should be handled by mask, but just in case)
    valid_mask = (targets >= 0) & (targets <= 1)
    predictions = predictions[valid_mask]
    targets = targets[valid_mask]
    
    if len(predictions) == 0:
        return {
            'accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0,
            'iou': 0.0,
            'dice': 0.0
        }
    
    # Convert to binary
    predictions_bin = (predictions > 0.5).float()
    targets_bin = (targets > 0.5).float()
    
    # Calculate confusion matrix components
    tp = ((predictions_bin == 1) & (targets_bin == 1)).sum().float()
    tn = ((predictions_bin == 0) & (targets_bin == 0)).sum().float()
    fp = ((predictions_bin == 1) & (targets_bin == 0)).sum().float()
    fn = ((predictions_bin == 0) & (targets_bin == 1)).sum().float()
    
    # Avoid division by zero
    eps = 1e-8
    
    # Calculate metrics
    accuracy = (tp + tn) / (tp + tn + fp + fn + eps)
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * (precision * recall) / (precision + recall + eps)
    iou = tp / (tp + fp + fn + eps)
    dice = 2 * tp / (2 * tp + fp + fn + eps)
    
    return {
        'accuracy': accuracy.item(),
        'precision': precision.item(),
        'recall': recall.item(),
        'f1': f1.item(),
        'iou': iou.item(),
        'dice': dice.item()
    }

def print_metrics(metrics):
    """
    Pretty print evaluation metrics
    """
    print("\n" + "="*50)
    print("EVALUATION METRICS")
    print("="*50)
    print(f"Loss:          {metrics['loss']:.6f}")
    print(f"Accuracy:      {metrics['accuracy']:.4f}")
    print(f"Precision:     {metrics['precision']:.4f}")
    print(f"Recall:        {metrics['recall']:.4f}")
    print(f"F1 Score:      {metrics['f1']:.4f}")
    print(f"IoU:           {metrics['iou']:.4f}")
    print(f"Dice Coefficient: {metrics['dice']:.4f}")
    print("="*50)


def save_results(metrics, predictions, targets, masks, save_dir=""):
    #     # Optional: Save results for further analysis
    results = {
        'metrics': metrics,
        'predictions': predictions.numpy(),
        'targets': targets.numpy(),
        'masks': masks.numpy()
    }
    
    
    # save_path = tag_save_dir + "/evaluation_results.pt"
    # save_path = save_dir / "evaluation_results.json"
    save_path = save_dir / "evaluation_results.pt"

    # Save if needed
    torch.save(results, save_path)
    print(f"Evaluated results saved to:{save_path}")

def plot_confusion_matrix_total_percentage(predictions, targets, mask, save_dir, cf_image_name=None):
    """
    Plot confusion matrix with percentages relative to total pixels
    """   
    # Apply mask
    masked_pred = predictions[mask].flatten()
    masked_target = targets[mask].flatten()
    
    # Convert to binary
    pred_bin = (masked_pred > 0.5).astype(int)
    target_bin = (masked_target > 0.5).astype(int)
    
    # Compute confusion matrix
    cm = confusion_matrix(target_bin, pred_bin)
    total_pixels = cm.sum()
    
    # Calculate percentages relative to total
    cm_percent = cm.astype('float') / total_pixels * 100
    cm_percent = np.round(cm_percent, 1)

    print(cm_percent)
    
    # Create annotation labels
    annot_labels = np.empty_like(cm).astype(str)
    n_rows, n_cols = cm.shape
    for i in range(n_rows):
        for j in range(n_cols):
            count = cm[i, j]
            percent = cm_percent[i, j]
            annot_labels[i, j] = f'{count}\n({percent}%)'
    
    # Create figure and plot
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=annot_labels, fmt='', cmap='Blues', 
                xticklabels=['No Flood', 'Flood'],
                yticklabels=['No Flood', 'Flood'],
                cbar_kws={'label': 'Number of Pixels'},
                ax=ax)
    
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title('Confusion Matrix (Count + Percentage of Total)')
    plt.tight_layout()
    
    if cf_image_name == None:
        cf_image_name="cf"
    # Save the figure if needed
    if save_dir is not None:
        save_path = save_dir / f"{cf_image_name}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to:{save_path}")
    
    # Display the plot
    plt.show()



def analyze_per_class_performance(predictions, targets, mask):
    """
    Analyze performance for flood vs non-flood classes separately
    """
    masked_pred = predictions[mask].flatten()
    masked_target = targets[mask].flatten()
    
    pred_bin = (masked_pred > 0.5).astype(int)
    target_bin = (masked_target > 0.5).astype(int)
    
    # Flood class performance
    flood_mask = target_bin == 1
    flood_accuracy = (pred_bin[flood_mask] == target_bin[flood_mask]).mean()
    
    # Non-flood class performance
    non_flood_mask = target_bin == 0
    non_flood_accuracy = (pred_bin[non_flood_mask] == target_bin[non_flood_mask]).mean()
    
    print(f"Flood class accuracy: {flood_accuracy:.4f}")
    print(f"Non-flood class accuracy: {non_flood_accuracy:.4f}")
    print(f"Number of flood pixels: {flood_mask.sum()}")
    print(f"Number of non-flood pixels: {non_flood_mask.sum()}")