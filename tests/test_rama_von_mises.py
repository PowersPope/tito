import importlib.util
import unittest

import numpy as np
import torch

from tito.models.rama import (
    ResidueVonMisesMixture,
    class_conditioned_rama_mmd,
)


def load_fitter():
    spec = importlib.util.spec_from_file_location(
        "fit_rama_von_mises",
        "scripts/fit_rama_von_mises.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class RamaVonMisesTest(unittest.TestCase):
    def test_rama_mmd_is_zero_for_identical_distributions(self):
        phi = torch.tensor([-1.2, -0.9, 1.9, 2.2])
        psi = torch.tensor([-0.8, -0.5, 2.4, 2.7])
        rama_class = torch.tensor([0, 0, 1, 1])
        loss = class_conditioned_rama_mmd(
            phi, psi, phi, psi, rama_class
        )
        self.assertAlmostEqual(float(loss), 0.0, places=6)

    def test_rama_mmd_detects_shift_and_has_gradients(self):
        target_phi = torch.tensor([-1.2, -0.9, 1.9, 2.2])
        target_psi = torch.tensor([-0.8, -0.5, 2.4, 2.7])
        pred_phi = (target_phi + 0.9).clone().requires_grad_(True)
        pred_psi = (target_psi - 0.7).clone().requires_grad_(True)
        rama_class = torch.tensor([0, 0, 1, 1])
        loss = class_conditioned_rama_mmd(
            pred_phi,
            pred_psi,
            target_phi,
            target_psi,
            rama_class,
        )
        loss.backward()
        self.assertGreater(float(loss.detach()), 0.01)
        self.assertTrue(torch.isfinite(pred_phi.grad).all())
        self.assertTrue(torch.isfinite(pred_psi.grad).all())
        self.assertGreater(float(pred_phi.grad.abs().sum()), 0.0)
        self.assertGreater(float(pred_psi.grad.abs().sum()), 0.0)

    def test_em_recovers_two_modes(self):
        fitter = load_fitter()
        rng = np.random.default_rng(7)
        component = rng.integers(0, 2, size=4000)
        phi = np.where(
            component == 0,
            rng.vonmises(-1.1, 20.0, size=4000),
            rng.vonmises(2.2, 15.0, size=4000),
        )
        psi = np.where(
            component == 0,
            rng.vonmises(-0.7, 18.0, size=4000),
            rng.vonmises(2.6, 12.0, size=4000),
        )
        fit = fitter.fit_mixture(phi, psi, components=2, iterations=100, seed=3)
        self.assertTrue(np.all(fit["kappa_phi"] > 1.0))
        self.assertTrue(np.all(fit["kappa_psi"] > 1.0))
        self.assertAlmostEqual(float(fit["weights"].sum()), 1.0, places=5)

    def test_cartesian_nll_has_finite_gradients(self):
        prior = ResidueVonMisesMixture(
            log_weights=[[0.0]],
            mu_phi=[[-1.0]],
            mu_psi=[[1.0]],
            kappa_phi=[[5.0]],
            kappa_psi=[[4.0]],
            active_class=[True],
        )
        torch.manual_seed(4)
        x = torch.randn(8, 3, requires_grad=True)
        phi_index = torch.tensor([[0], [1], [2], [3]])
        psi_index = torch.tensor([[4], [5], [6], [7]])
        nll, count = prior.nll_from_coordinates(
            x,
            phi_index,
            psi_index,
            torch.tensor([0]),
        )
        nll.backward()
        self.assertEqual(count.item(), 1)
        self.assertTrue(torch.isfinite(nll))
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertGreater(float(x.grad.abs().sum()), 0.0)


if __name__ == "__main__":
    unittest.main()