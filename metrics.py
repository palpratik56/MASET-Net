def Performance_metrics(preds, targets, threshold=0.5):
    preds = (preds > threshold).float()
    targets = (targets > threshold).float()

    # Flatten tensors
    preds = preds.view(-1)
    targets = targets.view(-1)

    # True Positives, False Positives, True Negatives, False Negatives
    TP = torch.sum(preds * targets)
    FP = torch.sum(preds * (1 - targets))
    TN = torch.sum((1 - preds) * (1 - targets))
    FN = torch.sum((1 - preds) * targets)

    # Precision, Recall, F1 Score
    precision = TP / (TP + FP + 1e-8)
    recall = TP / (TP + FN + 1e-8)
    fscore = (2 * precision * recall) / (precision + recall + 1e-8)
    #Accuracy
    acc = (TP + TN) / (TP + FP + FN +  TN + 1e-8)
    # Dice Coefficient
    dice = 2 * TP / (2 * TP + FP + FN + 1e-8)
    # IoU Score
    iou = TP / (TP + FP + FN + 1e-8)

    return precision.item(), recall.item(), fscore.item(), acc.item(), dice.item(), iou.item()
