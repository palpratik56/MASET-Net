import torch
import torch.optim as optim
from model import MasetNet
from dataset import CVCDataset, ISICDataset, KVASDataset
from metrics import Performance_metrics
from torch.amp import autocast, GradScaler
import time

def train_model(model, optimizer, scheduler, criterion, epochs, step_size):

    scaler = GradScaler(enabled=(device.type == "cuda"))
    
    training_metrics = {
        'loss': [], 'precision': [], 'accuracy': [],
        'recall': [], 'dice': [], 'iou': []
    }
    validation_metrics = {
        'loss': [], 'precision': [], 'accuracy': [],
        'recall': [], 'dice': [], 'iou': []
    }
    
    start_time = time.time()
    
    for epoch in range(epochs):
    
        # =========================
        # TRAINING
        # =========================
        model.train()
        torch.cuda.reset_peak_memory_stats(device)
    
        train_running = {
            'loss': 0.0, 'precision': 0.0, 'accuracy': 0.0,
            'recall': 0.0, 'dice': 0.0, 'iou': 0.0
        }
    
        for images, masks in train_loader:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
    
            optimizer.zero_grad(set_to_none=True)
    
            with autocast(device_type=device.type):
                outputs = model(images)
                loss = criterion(outputs, masks)
    
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
    
            train_running['loss'] += loss.item() * images.size(0)
    
            precision, recall, accuracy, dice, iou = Performance_metrics(outputs, masks)
            train_running['precision'] += precision
            train_running['recall'] += recall
            train_running['accuracy'] += accuracy
            train_running['dice'] +=  dice
            train_running['iou'] +=  iou
    
        # Peak VRAM (training)
        # train_peak_vram = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
    
        epoch_metrics = {
            'loss': train_running['loss'] / len(train_loader.dataset),
            'precision': train_running['precision'] / len(train_loader),
            'recall': train_running['recall'] / len(train_loader),
            'accuracy': train_running['accuracy'] / len(train_loader),
            'dice': train_running['dice'] / len(train_loader),
            'iou': train_running['iou'] / len(train_loader)
        }
    
        for key in training_metrics:
            training_metrics[key].append(epoch_metrics[key])

    
        print(f'\nEpoch {epoch+1} [TRAIN]')
        print(f'Acc: {epoch_metrics["accuracy"]:.3f}, '
              f'Prec: {epoch_metrics["precision"]:.3f}, '
              f'Rec: {epoch_metrics["recall"]:.3f}, '
              f'Dice: {epoch_metrics["dice"]:.3f}, '
              f'IoU: {epoch_metrics["iou"]:.3f}, '
              f'Loss: {epoch_metrics["loss"]:.3f}')
        # print(f'Peak VRAM (Train): {train_peak_vram:.2f} GB')
    
        # =========================
        # VALIDATION
        # =========================
        model.eval()
        torch.cuda.reset_peak_memory_stats(device)
    
        val_running = {
            'loss': 0.0, 'precision': 0.0, 'accuracy': 0.0,
            'recall': 0.0, 'dice': 0.0, 'iou': 0.0
        }
    
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device, non_blocking=True)
                masks = masks.to(device, non_blocking=True)
    
                with autocast(device_type=device.type):
                    outputs = model(images)
                    loss = criterion(outputs, masks)
    
                val_running['loss'] += loss.item() * images.size(0)
    
                precision, recall, accuracy, dice, iou = Performance_metrics(outputs, masks)
                val_running['precision'] += precision
                val_running['recall'] += recall
                val_running['accuracy'] += accuracy
                val_running['dice'] += dice
                val_running['iou'] += iou
    
        # Peak VRAM (validation)
        # val_peak_vram = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
    
        val_epoch_metrics = {
            'loss': val_running['loss'] / len(val_loader.dataset),
            'precision': val_running['precision'] / len(val_loader),
            'recall': val_running['recall'] / len(val_loader),
            'accuracy': val_running['accuracy'] / len(val_loader),
            'dice': val_running['dice'] / len(val_loader),
            'iou': val_running['iou'] / len(val_loader)
        }
    
        for key in validation_metrics:
            validation_metrics[key].append(val_epoch_metrics[key])
    
        print(f'Epoch {epoch+1} [VAL]')
        print(f'Acc: {val_epoch_metrics["accuracy"]:.3f}, '
              f'Prec: {val_epoch_metrics["precision"]:.3f}, '
              f'Rec: {val_epoch_metrics["recall"]:.3f}, '
              f'Dice: {val_epoch_metrics["dice"]:.3f}, '
              f'IoU: {val_epoch_metrics["iou"]:.3f}, '
              f'Loss: {val_epoch_metrics["loss"]:.3f}')
        # print(f'Peak VRAM (Val): {val_peak_vram:.2f} GB')
    
        scheduler.step()
    
        if (epoch + 1) % step_size == 0:
            print(f"Learning Rate: {optimizer.param_groups[0]['lr']}")
    
    training_time = (time.time() - start_time) / 60
    print(f'\nTotal training time: {training_time:.2f} minutes')

    return {'train': training_metrics, 
            'validation': validation_metrics, 'total_time_min': training_time}

