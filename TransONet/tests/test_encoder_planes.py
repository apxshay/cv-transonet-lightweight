import unittest

import torch

from tests.helpers import dummy_opt, make_encoder


class TestEncoderPlanes(unittest.TestCase):
    def _encode(self):
        enc = make_encoder()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        enc.to(device)
        p = torch.rand(2, 64, 3, device=device) - 0.5
        enc.eval()
        with torch.no_grad():
            fea = enc(p, dummy_opt(enc))
        return fea, p

    def test_three_maps_and_bases(self):
        fea, p = self._encode()
        self.assertEqual(tuple(fea['c_mat'].shape), (p.size(0), 3, 4, 3))
        self.assertEqual(tuple(fea['planes'].shape), (p.size(0), 3, 32, 32, 32))
        for i in range(3):
            for j in range(i + 1, 3):
                self.assertFalse(torch.allclose(fea['planes'][:, i], fea['planes'][:, j], atol=1e-5))
                self.assertFalse(torch.allclose(fea['c_mat'][:, i], fea['c_mat'][:, j], atol=1e-5))

    def test_bases_not_summed(self):
        fea, _ = self._encode()
        self.assertEqual(fea['c_mat'].dim(), 4)
        self.assertEqual(fea['c_mat'].shape[1], 3)

    def test_output_device(self):
        fea, p = self._encode()
        self.assertEqual(fea['planes'].device, p.device)
        self.assertEqual(fea['c_mat'].device, p.device)


if __name__ == '__main__':
    unittest.main()
