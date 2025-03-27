import torch
import torch.optim as optim
from model import MasetNet
from dataset import CVCDataset, ISICDataset, KVASDataset
from metrics import Performance_metrics

def train_model(train_loader, val_loader, scheduler, crieterion, model, optimizer):
    # Initialize lists for tracking metrics
training_metrics = {
    'loss': [], 'precision': [], 'accuracy': [], 
    'recall': [], 'fsc': [], 'dice': [], 'iou' : []
}
validation_metrics = {
    'loss': [], 'precision': [], 'accuracy': [], 
    'recall': [], 'fsc': [], 'dice': [], 'iou' : []
}

# Training loop
num_epochs = 60
start_time = time.time()

for epoch in range(num_epochs):
    # Training phase
    model.train()
    train_running = {
        'loss': 0.0, 'precision': 0.0, 'fsc': 0.0, 'accuracy': 0.0,
        'recall': 0.0, 'dice': 0.0, 'iou' : 0.0
    }
    
    for images, masks in train_loader:
        images = images.to(device)
        masks = masks.to(device)
        
        optimizer.zero_grad()
        outputs,_ = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        
        # Update running metrics
        train_running['loss'] += loss.item() * images.size(0)
        precision, recall, fsc, accuracy, dice, iou = Performance_metrics(outputs, masks)
        train_running['precision'] += precision
        train_running['recall'] += recall
        train_running['fsc'] += fsc
        train_running['accuracy'] += accuracy
        train_running['dice'] += dice
        train_running['iou'] += iou
        
    # Calculate epoch metrics for training
    epoch_metrics = {
        'loss': train_running['loss'] / len(train_loader.dataset),
        'precision': train_running['precision'] / len(train_loader),
        'recall': train_running['recall'] / len(train_loader),
        'fsc': train_running['fsc'] / len(train_loader),
        'accuracy': train_running['accuracy'] / len(train_loader),
        'dice': train_running['dice'] / len(train_loader),
        'iou': train_running['iou'] / len(train_loader)
    }
    
    # Store training metrics
    for key in training_metrics:
        training_metrics[key].append(epoch_metrics[key])
    
    print(f'Training Epoch {epoch+1}:')
    print(f'Accuracy: {epoch_metrics["accuracy"]:.3f}, Precision: {epoch_metrics["precision"]:.3f},'
          f' Recall: {epoch_metrics["recall"]:.3f}, F1 Score: {epoch_metrics["fsc"]:.3f},'
          f' Dice: {epoch_metrics["dice"]:.3f}, IoU: {epoch_metrics["iou"]:.3f}, Loss: {epoch_metrics["loss"]:.3f}')
    
    # Validation phase
    model.eval()
    val_running = {
        'loss': 0.0, 'precision': 0.0, 'fsc': 0.0, 'accuracy': 0.0,
        'recall': 0.0, 'dice': 0.0, 'iou' : 0.0
    }
    
    with torch.no_grad():
        for images, masks in val_loader:
            images = images.to(device)
            masks = masks.to(device)
            
            outputs,_ = model(images)
            loss = criterion(outputs, masks)
            
            # Update running metrics
            val_running['loss'] += loss.item() * images.size(0)
            precision, recall, fsc, accuracy, dice, iou = Performance_metrics(outputs, masks)
            val_running['precision'] += precision
            val_running['recall'] += recall
            val_running['fsc'] += fsc
            val_running['accuracy'] += accuracy
            val_running['dice'] += dice
            val_running['iou'] += iou
            
    # Calculate epoch metrics for validation
    val_epoch_metrics = {
        'loss': val_running['loss'] / len(val_loader.dataset),
        'precision': val_running['precision'] / len(val_loader),
        'recall': val_running['recall'] / len(val_loader),
        'fsc': val_running['fsc'] / len(val_loader),
        'accuracy': val_running['accuracy'] / len(val_loader),
        'dice': val_running['dice'] / len(val_loader),
        'iou': val_running['iou'] / len(val_loader)
    }
    
    # Store validation metrics
    for key in validation_metrics:
        validation_metrics[key].append(val_epoch_metrics[key])
    
    print(f'Validation Epoch {epoch+1}:')
    print(f'Accuracy: {val_epoch_metrics["accuracy"]:.3f}, Precision: {val_epoch_metrics["precision"]:.3f}, '
          f'Recall: {val_epoch_metrics["recall"]:.3f}, F1 Score: {val_epoch_metrics["fsc"]:.3f},'
          f' Dice: {val_epoch_metrics["dice"]:.3f}, IoU: {val_epoch_metrics["iou"]:.3f}, Loss: {val_epoch_metrics["loss"]:.3f}')
    
    # Update learning rate
    scheduler.step()
    if (epoch + 1) % 20 == 0:
        current_lr = optimizer.param_groups[0]['lr']
        print(f"At epoch {epoch+1}, Learning Rate: {current_lr}")

