import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc
import torch
import torch.nn as nn

def plot_training_metrics(training_metrics, validation_metrics, save_path=None):
    # Set style for better visualization
    plt.style.use('seaborn')
    
    # Create figure and subplots
    fig, axs = plt.subplots(2, 3, figsize=(15, 12))
    # fig.suptitle('Performance Metrics w.r.t RMSprop Optimizer', fontsize=18, y=0.95)
    
    # Flatten axs for easier iteration
    axs = axs.flatten()
    
    # Metrics to plot (excluding dice since it's similar to accuracy)
    metrics = ['loss', 'accuracy', 'precision', 'recall', 'fsc', 'iou']
    titles = ['Loss', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'IoU']
    
    # Plot each metric
    for idx, (metric, title) in enumerate(zip(metrics, titles)):
        # Plot training metric
        axs[idx].plot(training_metrics[metric], 
                     label=f'Training {title}', 
                     color='violet', 
                     linewidth=2)
        
        # Plot validation metric
        axs[idx].plot(validation_metrics[metric], 
                     label=f'Validation {title}', 
                     color='orange', 
                     linewidth=2)
        
        # Customize subplot
        axs[idx].set_title(f'{title}', pad=10)
        axs[idx].set_xlabel('Epoch')
        # axs[idx].set_ylabel(title)
        axs[idx].grid(True, linestyle='--', alpha=0.7)
        axs[idx].legend(loc='best')
        
        # Add horizontal and vertical grid
        axs[idx].grid(True, which='both', linestyle='--', alpha=0.6)
        
        # Set y-axis limits for metrics other than loss
        if metric != 'loss':
            axs[idx].set_ylim([0, 1.1])

    # Adjust layout to prevent overlap
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save plot if path is provided
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
    
    plt.show()

# Create a separate figure for Dice Score
def plot_dice_score(training_metrics, validation_metrics, save_path=None):
    plt.figure(figsize=(10, 6))
    plt.plot(training_metrics['dice'], label='Training Dice', color='violet', linewidth=2)
    plt.plot(validation_metrics['dice'], label='Validation Dice', color='orange', linewidth=2)
    
    # plt.title('Dice Score w.r.t RMSprop Optimizer', pad=20, fontsize=14)
    plt.xlabel('Epoch')
    # plt.ylabel('Dice Score')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='best')
    plt.ylim([0, 1.1])
    
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
    
    plt.show()

# Plot metrics
plot_training_metrics(training_metrics, validation_metrics, save_path='ISIC_performance_metrics_Adagr.png')

# Plot Dice score
plot_dice_score(training_metrics, validation_metrics, save_path='ISIC_dice_score_Adagr.png')

def save_metrics(training_metrics, validation_metrics, filename='ISIC_training_history_Adagr.csv'):
    # Create a DataFrame with all metrics
    data = {
        'epoch': list(range(1, len(training_metrics['loss']) + 1)),
        'train_loss': training_metrics['loss'],
        'train_accuracy': training_metrics['accuracy'],
        'train_precision': training_metrics['precision'],
        'train_recall': training_metrics['recall'],
        'train_fsc': training_metrics['fsc'],
        'train_dice': training_metrics['dice'],
        'train_iou': training_metrics['iou'],
        
        'val_loss': validation_metrics['loss'],
        'val_accuracy': validation_metrics['accuracy'],
        'val_precision': validation_metrics['precision'],
        'val_recall': validation_metrics['recall'],
        'val_fsc': validation_metrics['fsc'],
        'val_dice': validation_metrics['dice'],
        'val_iou': validation_metrics['iou']
    }
    
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)


def calculate_roc_curve(model, data_loader, device, threshold=0.5):
    model.eval()
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for images, masks in data_loader:
            images = images.to(device)
            masks = masks.to(device)
            
            outputs,_= model(images)
            probs = F.sigmoid(outputs).cpu().numpy()
            
            # Flatten and binarize
            all_probs.extend(probs.flatten())
            # Convert masks to binary values
            binary_masks = (masks.cpu().numpy() > threshold).astype(np.int32)
            all_labels.extend(binary_masks.flatten())
    
    # Convert to numpy arrays
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    
    # Ensure labels are binary
    all_labels = (all_labels > threshold).astype(np.int32)
    
    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)
    
    return fpr, tpr, roc_auc

def plot_roc_curve(fpr, tpr, roc_auc, save_path=None):
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    # plt.title('ROC Curve w.r.t RMSprop Optimizer')
    plt.legend(loc="lower right")
    plt.grid(True, linestyle='--', alpha=0.7)
    
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
    
    plt.show()

save_metrics(training_metrics, validation_metrics)
fpr, tpr, roc_auc = calculate_roc_curve(model, val_loader, device)  
plot_roc_curve(fpr, tpr, roc_auc, save_path='ISIC_roc_curve_Adagr.png')

def show_test_predictions(test_loader, model, device, num_images=5):
    torch.cuda.empty_cache()  # Clear GPU cache before starting
    model.eval()  # Set model to evaluation mode
    fig, axs = plt.subplots(3, num_images, figsize=(10, 8))
    
    # Get a batch of images that's large enough
    images, masks = next(iter(test_loader))
    batch_size = images.size(0)
    
    # Generate unique random indices
    selected_indices = random.sample(range(batch_size), num_images)
    
    with torch.no_grad():  # No need for gradients in evaluation
        for i, idx in enumerate(selected_indices):
            # Use the randomly selected index
            image = images[idx].to(device)
            mask = masks[idx].to(device)
            
            # Get the predicted mask from the model
            output,_ = model(image.unsqueeze(0))  # Add batch dimension (1, C, H, W)
            predicted_mask = torch.sigmoid(output).squeeze().cpu().numpy()  # Apply sigmoid and remove batch dim
            
            # Convert image and mask tensors to numpy arrays for visualization
            image_np = image.cpu().permute(1, 2, 0).numpy()  # Convert CHW to HWC
            mask_np = mask.cpu().squeeze().numpy()  # Convert mask to numpy
            
            # Plot the original image
            axs[0, i].imshow(image_np)
            axs[0, i].set_title(f"Image {i+1}")
            axs[0, i].axis('off')
            
            # Plot the ground truth mask
            axs[1, i].imshow(mask_np, cmap='gray')
            axs[1, i].set_title(f"Ground Truth {i+1}")
            axs[1, i].axis('off')
            
            # Plot the predicted mask
            axs[2, i].imshow(predicted_mask, cmap='gray')
            axs[2, i].set_title(f"Predicted Mask {i+1}")
            axs[2, i].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Save the figure
    fig.savefig('ISIC_predictions_Adagr.png', dpi=200)
    # Clean up
    plt.close(fig)

def visualize_feature_maps(feature_maps, ground_truths, save_path='ISIC_feature_maps_Adagr.png'):
    num_maps = len(feature_maps)
    fig, axs = plt.subplots(1, num_maps + 1, figsize=(15, 6))
    
    # Randomly select an image index from the batch
    batch_size = feature_maps[list(feature_maps.keys())[0]].shape[0]
    random_idx = torch.randint(0, batch_size, (1,)).item()
    
    for i, (name, feature_map) in enumerate(feature_maps.items()):
        # Feature map visualization (use first channel and normalize)
        feature_map_np = feature_map[random_idx, 0].detach().cpu().numpy()
        feature_map_np = (feature_map_np - feature_map_np.min()) / (feature_map_np.max() - feature_map_np.min())
        
        # Feature map subplot  
        axs[i].imshow(feature_map_np, cmap='viridis')
        if i == 4:
            axs[i].set_title('Output')
        else:
            axs[i].set_title(f'Decoder Stage: {i+1}')
        axs[i].axis('off')
    
    # Using the same random_idx for consistency
    ground_truth_np = ground_truths[random_idx].detach().cpu().numpy()
    ground_truth_np = ground_truth_np.squeeze(0)  # Squeeze the first dimension (channel)
    
    axs[-1].imshow(ground_truth_np, cmap='viridis')
    axs[-1].set_title('Ground Truth')
    axs[-1].axis('off')
    
    plt.tight_layout()
    
    # Add the random index to the save path to track different visualizations
    plt.savefig(save_path, dpi=200)
    plt.show()
  
#See the predictions
show_test_predictions(test_loader, model, device)
# Visualize feature maps for a test batch
test_batch_iter = iter(test_loader)
test_images, test_masks = next(test_batch_iter)
with torch.no_grad():
        test_images = test_images.to(device)
        outputs, feature_maps = model(test_images)
        visualize_feature_maps(feature_maps, test_masks)
