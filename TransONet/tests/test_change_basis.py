import unittest

import torch

from src.common import ChangeBasis


class TestChangeBasis(unittest.TestCase):
    def test_shape_and_finite(self):
        n = torch.eye(3).unsqueeze(0).expand(4, -1, -1).contiguous()
        c_mat = ChangeBasis(n)
        self.assertEqual(tuple(c_mat.shape), (4, 3, 4, 3))
        self.assertTrue(torch.isfinite(c_mat).all())

    def test_near_zero_normal_is_finite(self):
        n = torch.randn(2, 3, 3) * 1e-12
        c_mat = ChangeBasis(n)
        self.assertTrue(torch.isfinite(c_mat).all())

    def test_stays_on_cpu(self):
        n = torch.eye(3).unsqueeze(0).expand(2, -1, -1).contiguous()
        self.assertEqual(ChangeBasis(n).device.type, 'cpu')


if __name__ == '__main__':
    unittest.main()
