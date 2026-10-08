"""H2LO: a DeepONet-style neural operator for HF -> LF MRI synthesis.

    G(u)(x) = sum_k b_k(u, x) * t_k(x) + beta

- Branch net: a lightweight 3D CNN encodes the HF volume u into a dense feature field;
  the coefficients b_k(u, x) are read out at the query coordinate x by trilinear sampling.
- Trunk net: a SIREN maps the normalized coordinate x in [-1, 1]^3 to basis values t_k(x)
  (a ReLU MLP trunk is provided for the "w/o SIREN" ablation).
- Output: their dot product plus a learnable global bias.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def normalize_volume(volume: np.ndarray) -> np.ndarray:
    """Max-normalize a volume to [0, 1]."""
    volume = volume.astype(np.float32)
    m = volume.max()
    if m > 0:
        volume = volume / m
    return np.clip(volume, 0, 1)


def make_coord_grid(shape, flatten=True):
    """Voxel-centre coordinates in [-1, 1] for a volume of shape (H, W, D)."""
    axes = [torch.linspace(-1, 1, n) if n > 1 else torch.zeros(1) for n in shape]
    coords = torch.stack(torch.meshgrid(*axes, indexing='ij'), dim=-1)  # [H, W, D, 3]
    return coords.view(-1, 3) if flatten else coords


# ----------------------------------------------------------------------------
# Branch net
# ----------------------------------------------------------------------------
class LightEncoder(nn.Module):
    """Lightweight 3D CNN branch encoder (five 3x3x3 convolutions)."""

    def __init__(self, feature_dim=128, base_channels=32):
        super().__init__()
        c = base_channels
        self.net = nn.Sequential(
            nn.Conv3d(1, c, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv3d(c, c, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv3d(c, 2 * c, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv3d(2 * c, 2 * c, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv3d(2 * c, feature_dim, 3, padding=1),
        )

    def forward(self, x):
        return self.net(x)


# ----------------------------------------------------------------------------
# Trunk nets
# ----------------------------------------------------------------------------
class SineLayer(nn.Module):
    """sin(omega_0 * (W x + b)) with the SIREN initialization (Sitzmann et al., NeurIPS 2020)."""

    def __init__(self, in_features, out_features, is_first=False, omega_0=30.):
        super().__init__()
        self.omega_0 = omega_0
        self.linear = nn.Linear(in_features, out_features)
        with torch.no_grad():
            bound = 1 / in_features if is_first else np.sqrt(6 / in_features) / omega_0
            self.linear.weight.uniform_(-bound, bound)

    def forward(self, x):
        return torch.sin(self.omega_0 * self.linear(x))


class SirenTrunkNet(nn.Module):
    """SIREN trunk: coordinates [..., 3] -> basis values [..., P]; the last layer is linear."""

    def __init__(self, in_dim=3, out_dim=128, depth=4, width=256, first_omega_0=30., hidden_omega_0=30.):
        super().__init__()
        layers = [SineLayer(in_dim, width, is_first=True, omega_0=first_omega_0)]
        for _ in range(depth - 2):
            layers.append(SineLayer(width, width, omega_0=hidden_omega_0))
        final = nn.Linear(width, out_dim)
        with torch.no_grad():
            bound = np.sqrt(6 / width) / hidden_omega_0
            final.weight.uniform_(-bound, bound)
        layers.append(final)
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class MLPTrunkNet(nn.Module):
    """ReLU MLP trunk (the "w/o SIREN" ablation); the last layer is linear."""

    def __init__(self, in_dim=3, out_dim=128, depth=4, width=256):
        super().__init__()
        layers = []
        for i in range(depth):
            layers.append(nn.Linear(in_dim if i == 0 else width, out_dim if i == depth - 1 else width))
            if i < depth - 1:
                layers.append(nn.ReLU(inplace=True))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ----------------------------------------------------------------------------
# H2LO operator
# ----------------------------------------------------------------------------
class H2LO(nn.Module):
    def __init__(self, feature_dim=128, encoder_base_channels=32, trunk_type='siren',
                 decoder_depth=4, decoder_width=256, omega=30.):
        super().__init__()
        self.encoder = LightEncoder(feature_dim=feature_dim, base_channels=encoder_base_channels)
        if trunk_type == 'siren':
            self.trunk = SirenTrunkNet(3, feature_dim, decoder_depth, decoder_width, omega, omega)
        elif trunk_type == 'mlp':
            self.trunk = MLPTrunkNet(3, feature_dim, decoder_depth, decoder_width)
        else:
            raise ValueError(f"unknown trunk_type: {trunk_type}")
        self.bias = nn.Parameter(torch.zeros(1))

    @staticmethod
    def sample(feat_map, coord):
        """Branch coefficients at coord [B, K, 3] (ordered as the volume axes) -> [B, K, P]."""
        grid = coord.flip(-1).unsqueeze(1).unsqueeze(1)  # grid_sample expects (z, y, x)
        b = F.grid_sample(feat_map, grid, mode='bilinear', align_corners=True, padding_mode='border')
        return b[:, :, 0, 0, :].permute(0, 2, 1)

    def query(self, feat_map, coord):
        """LF intensity at coord [B, K, 3] given the branch feature map -> [B, K, 1]."""
        return (self.sample(feat_map, coord) * self.trunk(coord)).sum(-1, keepdim=True) + self.bias

    def forward(self, hf_vol, coord):
        """hf_vol [B, 1, H, W, D], coord [B, K, 3] in [-1, 1] -> [B, K, 1]."""
        return self.query(self.encoder(hf_vol), coord)


@torch.no_grad()
def predict_full_volume(model, hf_vol, chunk_size=100000, use_amp=True):
    """Predict the whole LF volume on the HF grid. hf_vol: [1, 1, H, W, D] -> [H, W, D]."""
    model.eval()
    shape = tuple(hf_vol.shape[2:])
    amp = use_amp and hf_vol.is_cuda
    coord_all = make_coord_grid(shape).to(hf_vol.device)
    with torch.amp.autocast('cuda', enabled=amp):
        feat_map = model.encoder(hf_vol)
    out = []
    for s in range(0, coord_all.shape[0], chunk_size):
        with torch.amp.autocast('cuda', enabled=amp):
            pred = model.query(feat_map, coord_all[s:s + chunk_size].unsqueeze(0))
        out.append(pred.squeeze(0).squeeze(-1).float())
    return torch.cat(out).view(*shape)
