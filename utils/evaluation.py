from dec_feature_maps import visualize_feature_maps
from predictions.py import show_test_predictions
from plot_results import plot_training_metrics, plot_dice_score, plot_roc_curve
# 1. Plot metrics
plot_training_metrics(training_metrics, validation_metrics, save_path='ISIC_performance_metrics_RA.png')

# 2. Plot Dice score
plot_dice_score(training_metrics, validation_metrics, save_path='CVC_dice_score_AdamW.png')

# Save the metrices for future use
def save_metrics(training_metrics, validation_metrics, time, filename='ISIC_training_history_RA.csv'):
    # Create a DataFrame with all metrics
    data = {
        'epoch': list(range(1, len(training_metrics['loss']) + 1)),
        'train_loss': training_metrics['loss'],
        'train_accuracy': training_metrics['accuracy'],
        'train_precision': training_metrics['precision'],
        'train_recall': training_metrics['recall'],
        'train_dice': training_metrics['dice'],
        'train_iou': training_metrics['iou'],
        
        'val_loss': validation_metrics['loss'],
        'val_accuracy': validation_metrics['accuracy'],
        'val_precision': validation_metrics['precision'],
        'val_recall': validation_metrics['recall'],
        'val_dice': validation_metrics['dice'],
        'val_iou': validation_metrics['iou'],

        'training_time': round(time, 2)
    }
    
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)

# 3. Save metrics to CSV
#extract total time from metrics
tr_time = metrics['total_time_min']
save_metrics(training_metrics,  validation_metrics,  tr_time)

# 4. Calculate and plot ROC curve after training is complete:
fpr, tpr, roc_auc = calculate_roc_curve(model, val_loader, device)  
plot_roc_curve(fpr, tpr, roc_auc)

# 5. Show some test images with their predictions
show_test_predictions(test_loader, model, device)

# 6. Visualize feature maps for a test batch
test_batch_iter = iter(test_loader)
test_images, test_masks = next(test_batch_iter)
with torch.no_grad():
        test_images = test_images.to(device)
        outputs, feature_maps = model(test_images)
        visualize_feature_maps(feature_maps, test_masks)
