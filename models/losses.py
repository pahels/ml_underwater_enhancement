import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F


class PerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = models.vgg16(pretrained=True).features[:16]
        for p in vgg.parameters():
            p.requires_grad = False
        self.vgg = vgg.eval()

    def forward(self, pred, target):
        return F.l1_loss(self.vgg(pred), self.vgg(target))


def color_consistency_loss(pred):
    # Mean per channel
    mean = pred.mean(dim=[2, 3])
    r, g, b = mean[:, 0], mean[:, 1], mean[:, 2]
    return torch.mean((r - g) ** 2 + (r - b) ** 2 + (g - b) ** 2)
