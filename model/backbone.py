import torch
import torch.nn as nn
import torch.nn.functional as F

class SpatioTemporalVideoEncoder(nn.Module):
    """
    3D Spatio-Temporal Residual-Style CNN.
    Input Tensor: (Batch, Channels=3, Frames=16, Height=112, Width=112)
    """
    def __init__(self, latent_dim=128):
        super().__init__()
        self.conv1 = nn.Conv3d(3, 32, kernel_size=(3, 5, 5), stride=(1, 2, 2), padding=(1, 2, 2))
        self.bn1 = nn.BatchNorm3d(32)
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.conv2 = nn.Conv3d(32, 64, kernel_size=(3, 3, 3), stride=1, padding=1)
        self.bn2 = nn.BatchNorm3d(64)
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.adaptive_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.proj = nn.Linear(64, latent_dim)

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.adaptive_pool(x)
        x = torch.flatten(x, 1)
        return F.normalize(self.proj(x), p=2, dim=-1)


class AudioSpectrogramEncoder(nn.Module):
    """
    Stereo Mel-Spectrogram 2D-CNN.
    Input Tensor: (Batch, Channels=2, Freq_Bands=128, Time_Frames=64)
    """
    def __init__(self, latent_dim=128):
        super().__init__()
        self.conv1 = nn.Conv2d(2, 32, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.proj = nn.Linear(64, latent_dim)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.adaptive_pool(x)
        x = torch.flatten(x, 1)
        return F.normalize(self.proj(x), p=2, dim=-1)


class SAVeDuality(nn.Module):
    """
    Self-Supervised Cross-Modal Duality Network.
    Computes metric cosine alignment between audio physics and video dynamics.
    """
    def __init__(self, latent_dim=128):
        super().__init__()
        self.v_enc = SpatioTemporalVideoEncoder(latent_dim)
        self.a_enc = AudioSpectrogramEncoder(latent_dim)

    def forward(self, v_tensor, a_tensor):
        z_v = self.v_enc(v_tensor)
        z_a = self.a_enc(a_tensor)
        return z_v, z_a

    @torch.no_grad()
    def get_alignment_index(self, v_tensor, a_tensor):
        z_v = self.v_enc(v_tensor)
        z_a = self.a_enc(a_tensor)
        # Cosine similarity on unit hypersphere
        similarity = torch.sum(z_v * z_a, dim=-1).item()
        return float(similarity)