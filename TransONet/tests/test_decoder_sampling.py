import unittest

import torch

from tests.helpers import axis_bases, make_decoder


class TestDecoderSampling(unittest.TestCase):
    def test_forward_not_using_summed_basis(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        B, L, C, R, Q = 2, 3, 32, 32, 9
        dec = make_decoder(c_dim=C).to(device)
        p = torch.rand(B, Q, 3, device=device) - 0.5
        maps = torch.randn(B, L, C, R, R, device=device)
        bases = axis_bases(B, device=device)
        z = torch.empty(B, 0, device=device)
        out = dec(p, z, {'planes': maps, 'c_mat': bases})
        out_sum = dec(p, z, {'planes': maps[:, 0], 'c_mat': bases.sum(dim=1)})
        self.assertEqual(tuple(out.shape), (B, Q))
        self.assertFalse(torch.allclose(out, out_sum, atol=1e-4))


if __name__ == '__main__':
    unittest.main()
