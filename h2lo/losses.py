import torch
import torch.nn as nn
import torch.nn.functional as F


class GaussianGradientLoss3D(nn.Module):
    """3D gradient-consistency loss.

    Volume gradients are taken by convolution with the first derivatives of a 3D Gaussian
    (dG/dx, dG/dy, dG/dz); the loss is the summed MSE between predicted and target gradients.
    """

    def __init__(self, sigma=1.0, kernel_size=5):
        super().__init__()
        self.kernel_size = kernel_size
        half = (kernel_size - 1) / 2
        r = torch.linspace(-half, half, kernel_size)
        gz, gy, gx = torch.meshgrid(r, r, r, indexing='ij')
        g = torch.exp(-(gx ** 2 + gy ** 2 + gz ** 2) / (2 * sigma ** 2)) / (2 * torch.pi * sigma ** 2) ** 1.5
        shape = (1, 1, kernel_size, kernel_size, kernel_size)
        self.register_buffer('gx', (-gx / sigma ** 2 * g).view(shape))
        self.register_buffer('gy', (-gy / sigma ** 2 * g).view(shape))
        self.register_buffer('gz', (-gz / sigma ** 2 * g).view(shape))

    def forward(self, pred, target):
        """pred, target: [B, C, D, H, W] (sub-)volumes."""
        c = pred.shape[1]
        pad = self.kernel_size // 2
        loss = 0.
        for k in (self.gx, self.gy, self.gz):
            k = k.expand(c, 1, -1, -1, -1)
            loss = loss + F.mse_loss(F.conv3d(pred, k, groups=c, padding=pad),
                                     F.conv3d(target, k, groups=c, padding=pad))
        return loss
