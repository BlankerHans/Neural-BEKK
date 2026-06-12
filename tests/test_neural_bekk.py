import unittest

import torch

from neural_bekk import NeuralBekk
from vech import vech, unvech


class NeuralBekkTests(unittest.TestCase):
    def test_vech_unvech_roundtrip_matches_lower_triangular_order(self):
        L = torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [2.0, 3.0, 0.0],
                [4.0, 5.0, 6.0],
            ]
        )

        self.assertTrue(torch.equal(unvech(vech(L), d=3), L))

    def test_initial_parameters_use_g_for_garch_and_b_for_asymmetry(self):
        model = NeuralBekk(
            n_assets=2,
            input_size=2,
            asym=True,
            Sigma0=torch.eye(2),
            jitter=1e-8,
            init_a=0.05,
            init_g=0.90,
            init_b=0.05,
        )

        _, _, params = model(torch.zeros(3, 4, 2), return_params=True)

        self.assertTrue(torch.allclose(params["a"], torch.full_like(params["a"], 0.05), atol=1e-6))
        self.assertTrue(torch.allclose(params["g"], torch.full_like(params["g"], 0.90), atol=1e-6))
        self.assertTrue(torch.allclose(params["b"], torch.full_like(params["b"], 0.05), atol=1e-6))

        self.assertTrue(torch.allclose(torch.diagonal(params["G"], dim1=-2, dim2=-1), params["g"]))
        self.assertTrue(torch.allclose(torch.diagonal(params["B"], dim1=-2, dim2=-1), params["b"]))

    def test_uppercase_C_sigma0_initialization_flag_is_supported(self):
        model = NeuralBekk(
            n_assets=2,
            input_size=2,
            Sigma0=torch.eye(2),
            init_C_from_sigma0=True,
        )

        sigma, chol = model(torch.zeros(1, 2, 2))

        self.assertEqual(sigma.shape, (1, 2, 2))
        self.assertEqual(chol.shape, (1, 2, 2))

    def test_initial_persistence_is_bounded_by_tau(self):
        tau_max = 0.995
        model = NeuralBekk(
            n_assets=2,
            input_size=2,
            asym=True,
            Sigma0=torch.eye(2),
            tau_max=tau_max,
            init_a=0.05,
            init_g=0.90,
            init_b=0.05,
        )

        _, _, params = model(torch.zeros(1, 2, 2), return_params=True)
        persistence = params["a"].square() + params["g"].square() + params["b"].square()

        self.assertTrue(torch.all(persistence < tau_max))
        self.assertTrue(torch.allclose(persistence, params["tau"], atol=1e-6))

    def test_c_initialization_targets_sigma0_intercept(self):
        jitter = 1e-8
        sigma0 = torch.tensor(
            [
                [1.0, 0.2],
                [0.2, 2.0],
            ],
            dtype=torch.float32,
        )
        init_a = 0.05
        init_g = 0.90
        init_b = 0.05

        model = NeuralBekk(
            n_assets=2,
            input_size=2,
            asym=True,
            Sigma0=sigma0,
            jitter=jitter,
            init_a=init_a,
            init_g=init_g,
            init_b=init_b,
        )

        C = model._make_C_from_raw(model.C_raw.unsqueeze(0)).squeeze(0)
        intercept_factor = 1.0 - init_g**2 - init_a**2 - 0.5 * init_b**2
        expected = intercept_factor * sigma0 + jitter * torch.eye(2)

        self.assertTrue(torch.allclose(C @ C.T, expected, atol=1e-6))

    def test_full_initial_parameters_use_full_identity_matrices(self):
        model = NeuralBekk(
            n_assets=3,
            input_size=3,
            bekk_type="full",
            asym=True,
            Sigma0=torch.eye(3),
            jitter=1e-8,
            init_a=0.05,
            init_g=0.90,
            init_b=0.05,
        )

        _, _, params = model(torch.zeros(2, 4, 3), return_params=True)
        I = torch.eye(3).view(1, 1, 3, 3)

        self.assertTrue(torch.allclose(params["A"], 0.05 * I, atol=1e-6))
        self.assertTrue(torch.allclose(params["G"], 0.90 * I, atol=1e-6))
        self.assertTrue(torch.allclose(params["B"], 0.05 * I, atol=1e-6))
        self.assertIsNone(params["a"])
        self.assertIsNone(params["g"])
        self.assertIsNone(params["b"])

    def test_full_operator_norm_persistence_is_bounded_by_tau(self):
        model = NeuralBekk(
            n_assets=3,
            input_size=3,
            bekk_type="full",
            asym=True,
            Sigma0=torch.eye(3),
            tau_max=0.995,
        )
        raw = torch.randn(7, model.param_head[-1].out_features) * 5.0

        _, A, G, B, tau, _, _, _ = model._split_full_params(raw, return_meta=True)
        persistence = (
            torch.linalg.matrix_norm(A, ord=2, dim=(-2, -1)).square()
            + torch.linalg.matrix_norm(G, ord=2, dim=(-2, -1)).square()
            + torch.linalg.matrix_norm(B, ord=2, dim=(-2, -1)).square()
        )

        self.assertTrue(torch.all(persistence <= tau.squeeze(-1) + 1e-5))

    def test_full_forward_returns_cholesky_and_full_parameter_sequences(self):
        model = NeuralBekk(
            n_assets=3,
            input_size=5,
            bekk_type="full",
            asym=True,
            Sigma0=torch.eye(3),
            return_std=torch.ones(3),
        )
        sigma, chol, params = model(torch.randn(4, 6, 5), return_params=True)

        self.assertEqual(sigma.shape, (4, 3, 3))
        self.assertEqual(chol.shape, (4, 3, 3))
        self.assertEqual(params["A"].shape, (4, 6, 3, 3))
        self.assertEqual(params["G"].shape, (4, 6, 3, 3))
        self.assertEqual(params["B"].shape, (4, 6, 3, 3))
        self.assertEqual(params["tau"].shape, (4, 6, 1))
        self.assertEqual(params["weights"].shape, (4, 6, 3))
        self.assertEqual(params["operator_norms"].shape, (4, 6, 3))
        self.assertTrue(torch.isfinite(sigma).all())


if __name__ == "__main__":
    unittest.main()
