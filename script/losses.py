import torch
import torch.nn as nn
import torch.nn.functional as F


class ComplementCrossEntropyLoss(nn.Module):
    """
    Complement Cross-Entropy (CCE) loss.

    logits: (B, C)
    targets: (B,) int64 class indices in [0, C-1]

    This loss penalizes the probabilities of all incorrect classes.
    For each sample i and class j ≠ y_i:

        L_cce = - sum_{j != y} 1/(C-1) * log(1 - p_ij)

    where p = softmax(logits).
    """
    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, logits, targets):
        # logits: (B, C), targets: (B,)
        num_classes = logits.size(1)

        # (B, C)
        probs = F.softmax(logits, dim=1)

        # one-hot of true class: (B, C)
        y_onehot = F.one_hot(targets, num_classes=num_classes).float()

        # complement target: 1 for incorrect classes, 0 for true class, normalized
        complement_target = (1.0 - y_onehot) / (num_classes - 1.0)

        # complement probabilities: 1 - p_ij
        complement_prob = 1.0 - probs
        complement_prob = torch.clamp(complement_prob, min=1e-7)  # avoid log(0)

        log_complement_prob = torch.log(complement_prob)

        # per-sample loss: sum over classes
        loss_per_sample = -torch.sum(complement_target * log_complement_prob, dim=1)

        if self.reduction == "mean":
            return loss_per_sample.mean()
        elif self.reduction == "sum":
            return loss_per_sample.sum()
        else:
            return loss_per_sample
