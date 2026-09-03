import torch
import torch.optim as optim

from src.encoder.pointnet import DynamicLocalPoolPointnet
from src.dynamic_planes_conv_onet.models.decoder import DynamicLocalDecoder
from src.common import ChangeBasis


def make_encoder(c_dim=32, plane_resolution=32, hidden_dim=32):
    return DynamicLocalPoolPointnet(
        c_dim=c_dim, hidden_dim=hidden_dim, plane_resolution=plane_resolution,
        n_channels=3, unet=False, pos_encoding=False,
    )


def make_decoder(c_dim=32):
    return DynamicLocalDecoder(
        dim=3, z_dim=0, c_dim=c_dim, hidden_size=32,
        padding=0.1, pos_encoding=False,
    )


def dummy_opt(module):
    params = [p for p in module.parameters() if p.requires_grad]
    return optim.Adam(params, lr=1e-3)


def axis_bases(batch_size=2, device='cpu'):
    n = torch.eye(3, device=device).unsqueeze(0).expand(batch_size, -1, -1).contiguous()
    return ChangeBasis(n, device=device)
