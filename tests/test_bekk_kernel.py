import unittest

import torch
import torch.nn.functional as F

from bekk_kernel import BEKKLSTM


class BEKKKernelTests(unittest.TestCase):
    def test_convex_mixture_alpha_can_be_returned(self):
        model = BEKKLSTM(
            input_size=2,
            n_assets=2,
            hidden_size=4,
            modulation="convex_mixture",
        )

        sigma, chol, params = model(torch.zeros(2, 3, 2), return_alpha=True)

        self.assertEqual(sigma.shape, (2, 2, 2))
        self.assertEqual(chol.shape, (2, 2, 2))
        self.assertEqual(params["alpha"].shape, (2, 3))
        expected_alpha = torch.sigmoid(torch.tensor(-6.0))
        self.assertTrue(torch.allclose(params["alpha"], torch.full_like(params["alpha"], expected_alpha)))

    def test_convex_mixture_nn_branch_is_returned_on_standardized_scale(self):
        return_std = torch.tensor([0.02, 0.03])
        bekk_scale = 100.0
        jitter = 1e-8
        model = BEKKLSTM(
            input_size=2,
            n_assets=2,
            hidden_size=4,
            modulation="convex_mixture",
            return_std=return_std,
            bekk_scale=bekk_scale,
            jitter=jitter,
        )

        with torch.no_grad():
            model.cell.alpha_head.bias.fill_(100.0)

        sigma, _ = model(torch.zeros(1, 1, 2))

        nn_diag = (F.softplus(torch.zeros(2)) + jitter).square()
        scale = return_std * bekk_scale
        expected_diag = nn_diag + jitter / scale.square()

        actual_diag = torch.diagonal(sigma, dim1=-2, dim2=-1).squeeze(0)
        self.assertTrue(torch.allclose(actual_diag, expected_diag, atol=1e-7, rtol=0.0))


if __name__ == "__main__":
    unittest.main()
