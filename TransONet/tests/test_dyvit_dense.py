import unittest

import torch

from src.models.dyvit import VisionTransformerDiffPruning
from tests.helpers import dummy_opt, make_encoder


def _vit(C, R):
    return VisionTransformerDiffPruning(
        img_size=R, patch_size=2, embed_dim=C, depth=2, in_chans=C, num_classes=0,
        num_heads=8, mlp_ratio=4, qkv_bias=True, pruning_loc=[], token_ratio=[1.0],
        distill=False, drop_path_rate=0.0)


class TestDyViTDense(unittest.TestCase):
    def test_out_shape(self):
        vit = _vit(32, 32)
        vit.eval()
        with torch.no_grad():
            y = vit(torch.randn(2, 32, 32, 32))
        self.assertEqual(tuple(y.shape), (2, 32, 32, 32))
        self.assertTrue(torch.isfinite(y).all())

    def test_c16_and_r128(self):
        vit = _vit(16, 32)
        vit.eval()
        with torch.no_grad():
            y = vit(torch.randn(2, 16, 32, 32))
        self.assertEqual(tuple(y.shape), (2, 16, 32, 32))
        vit = _vit(32, 128)
        vit.eval()
        with torch.no_grad():
            y = vit(torch.randn(1, 32, 128, 128))
        self.assertEqual(tuple(y.shape), (1, 32, 128, 128))

    def test_train_backward(self):
        vit = _vit(16, 32)
        x = torch.randn(2, 16, 32, 32, requires_grad=True)
        vit.train()
        y = vit(x)
        y = y[0] if isinstance(y, tuple) else y
        y.sum().backward()
        vit.eval()
        with torch.no_grad():
            y2 = vit(x.detach())
        self.assertEqual(y.shape, y2.shape)
        self.assertTrue(torch.isfinite(x.grad).all())
        for p in vit.parameters():
            if p.requires_grad and p.grad is not None:
                self.assertTrue(torch.isfinite(p.grad).all())


class TestEncoderDense(unittest.TestCase):
    def test_shapes(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        enc = make_encoder(c_dim=16, plane_resolution=32, hidden_dim=32).to(device)
        p = torch.rand(2, 64, 3, device=device) - 0.5
        enc.eval()
        with torch.no_grad():
            fea = enc(p, dummy_opt(enc))
        self.assertEqual(tuple(fea['planes'].shape), (2, 3, 16, 32, 32))
        self.assertEqual(tuple(fea['c_mat'].shape), (2, 3, 4, 3))
        self.assertTrue(torch.isfinite(fea['planes']).all())

        enc = make_encoder(c_dim=32, plane_resolution=128, hidden_dim=32).to(device)
        p = torch.rand(2, 32, 3, device=device) - 0.5
        enc.eval()
        with torch.no_grad():
            fea = enc(p, dummy_opt(enc))
        self.assertEqual(tuple(fea['planes'].shape), (2, 3, 32, 128, 128))
        self.assertEqual(tuple(fea['c_mat'].shape), (2, 3, 4, 3))
        self.assertTrue(torch.isfinite(fea['planes']).all())

    def test_train_backward(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        enc = make_encoder(c_dim=16, plane_resolution=32, hidden_dim=32).to(device)
        enc.train()
        p = torch.rand(2, 32, 3, device=device) - 0.5
        fea = enc(p, dummy_opt(enc))
        fea['planes'].sum().backward()
        self.assertEqual(tuple(fea['planes'].shape), (2, 3, 16, 32, 32))
        for par in enc.parameters():
            if par.requires_grad and par.grad is not None:
                self.assertTrue(torch.isfinite(par.grad).all())


if __name__ == '__main__':
    unittest.main()
