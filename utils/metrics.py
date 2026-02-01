def Performance_metrics(preds, targets, threshold=0.5, eps=1e-7):
    """
    Computes Precision, Recall, Accuracy, Dice, IoU
    averaged over the batch (per-image).
    """

    # Convert logits → probabilities
    preds = torch.sigmoid(preds)

    # Binarize
    preds = (preds > threshold).float()
    targets = (targets > threshold).float()

    # Compute per-image statistics
    dims = (1, 2, 3)

    TP = (preds * targets).sum(dim=dims)
    FP = (preds * (1 - targets)).sum(dim=dims)
    TN = ((1 - preds) * (1 - targets)).sum(dim=dims)
    FN = ((1 - preds) * targets).sum(dim=dims)

    precision = (TP / (TP + FP + eps)).mean()
    recall = (TP / (TP + FN + eps)).mean()
    acc = ((TP + TN) / (TP + FP + FN + TN + eps)).mean()
    dice = (2 * TP / (2 * TP + FP + FN + eps)).mean()
    iou = (TP / (TP + FP + FN + eps)).mean()

    return precision.item(), recall.item(), acc.item(), dice.item(), iou.item()
