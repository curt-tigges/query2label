# This implementation of CutMix was taken from the Github repo here:
# https://github.com/hysts/pytorch_cutmix
# The content of the above repo is usable via the standard MIT license.

import numpy as np
import torch
import torch.nn as nn


def cutmix(batch, alpha):
    """Applies random CutMix to images in batch

    Args:
        batch (Tensor): Images and labels
        alpha (float): Alpha value for CutMix algorithm

    Returns:
        tuple (Tensor, tuple): Shuffled images, and tuple containing targets,
        shuffled targets, and lambda (loss weighting value)
    """
    data, targets = batch

    indices = torch.randperm(data.size(0))
    shuffled_data = data[indices]
    shuffled_targets = targets[indices]

    lam = np.random.beta(alpha, alpha)

    image_h, image_w = data.shape[2:]
    cx = np.random.uniform(0, image_w)
    cy = np.random.uniform(0, image_h)
    w = image_w * np.sqrt(1 - lam)
    h = image_h * np.sqrt(1 - lam)
    x0 = int(np.round(max(cx - w / 2, 0)))
    x1 = int(np.round(min(cx + w / 2, image_w)))
    y0 = int(np.round(max(cy - h / 2, 0)))
    y1 = int(np.round(min(cy + h / 2, image_h)))

    data[:, :, y0:y1, x0:x1] = shuffled_data[:, :, y0:y1, x0:x1]
    targets = (targets, shuffled_targets, lam)

    return data, targets


class CutMixCollator:
    """Custom Collator for dataloader"""

    def __init__(self, alpha):
        self.alpha = alpha

    def __call__(self, batch):
        batch = torch.utils.data.dataloader.default_collate(batch)
        batch = cutmix(batch, self.alpha)
        return batch


class CutMixCriterion:
    """Applies criterion in a weighted fashion based on image shuffling"""

    def __init__(self, criterion):
        """Creates loss function

        Args:
            criterion (torch.nn loss object): Should be a binary loss class
        """
        self.criterion = criterion

    def __call__(self, preds, targets):
        """Applies loss function

        Args:
            preds (Tensor): Vector of prediction logits
            targets (tuple of Tensors): Targets and shuffled targets

        Returns:
            float: calculated loss
        """
        targets1, targets2, lam = targets
        return lam * self.criterion(preds, targets1) + (1 - lam) * self.criterion(
            preds, targets2
        )
