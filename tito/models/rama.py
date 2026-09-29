"""Residue-conditioned mixtures of product von Mises distributions."""

import math
from pathlib import Path

import numpy as np
import torch
from torch import nn

from tito.models.loss_terms import torsion_sin_cos


def rama_angles_from_coordinates(x, rama_phi_index, rama_psi_index):
    """Return paired phi/psi angles in radians for Ramachandran residues."""
    phi_sin, phi_cos = torsion_sin_cos(x, rama_phi_index)
    psi_sin, psi_cos = torsion_sin_cos(x, rama_psi_index)
    return torch.atan2(phi_sin, phi_cos), torch.atan2(psi_sin, psi_cos)


def circular_rama_features(phi, psi):
    """Embed the phi/psi torus in R^4 without an angle-wrap discontinuity."""
    return torch.stack(
        [torch.sin(phi), torch.cos(phi), torch.sin(psi), torch.cos(psi)],
        dim=-1,
    )


def _multiscale_rbf(x, y, bandwidths):
    squared_distance = torch.cdist(x, y).pow(2)
    kernels = [
        torch.exp(-squared_distance / (2.0 * float(bandwidth) ** 2))
        for bandwidth in bandwidths
    ]
    return torch.stack(kernels, dim=0).mean(dim=0)


def class_conditioned_rama_mmd(
    pred_phi,
    pred_psi,
    target_phi,
    target_psi,
    rama_class,
    bandwidths=(0.25, 0.5, 1.0, 2.0),
):
    """Differentiable distribution loss for p(phi, psi | residue class).

    This is the biased squared maximum mean discrepancy (MMD), averaged over
    residue classes in proportion to their target counts. The circular feature
    embedding makes the comparison continuous across -pi/pi.
    """
    if rama_class.numel() == 0:
        return pred_phi.new_zeros(())

    pred = circular_rama_features(pred_phi, pred_psi)
    target = circular_rama_features(target_phi, target_psi)
    total = pred.new_zeros(())
    total_count = 0

    for class_id in torch.unique(rama_class):
        mask = rama_class == class_id
        count = int(mask.sum().item())
        if count == 0:
            continue
        pred_class = pred[mask]
        target_class = target[mask]
        mmd = (
            _multiscale_rbf(pred_class, pred_class, bandwidths).mean()
            + _multiscale_rbf(target_class, target_class, bandwidths).mean()
            - 2.0 * _multiscale_rbf(pred_class, target_class, bandwidths).mean()
        )
        total = total + count * mmd
        total_count += count

    return total / max(total_count, 1)


def _log_i0(kappa):
    """Stable log(I0(kappa)) for non-negative concentrations."""
    return torch.log(torch.special.i0e(kappa)) + torch.abs(kappa)


class ResidueVonMisesMixture(nn.Module):
    """Fixed MD-fitted p(phi, psi | residue_class).

    Each component is a product of two von Mises densities.  Phi and psi are
    coupled through their shared mixture component, which is enough to model
    distinct alpha, beta and other Ramachandran basins without allowing an
    arbitrary Cartesian product of independently selected phi/psi modes.

    Buffers are non-persistent because the prior is an external fitted asset,
    not a learned model parameter.  Re-attach it from the same .npz when
    resuming training.
    """

    def __init__(
        self,
        log_weights,
        mu_phi,
        mu_psi,
        kappa_phi,
        kappa_psi,
        active_class,
    ):
        super().__init__()
        tensors = {
            "log_weights": log_weights,
            "mu_phi": mu_phi,
            "mu_psi": mu_psi,
            "kappa_phi": kappa_phi,
            "kappa_psi": kappa_psi,
            "active_class": active_class,
        }
        for name, value in tensors.items():
            tensor = torch.as_tensor(value)
            if name == "active_class":
                tensor = tensor.to(torch.bool)
            else:
                tensor = tensor.to(torch.float32)
            self.register_buffer(name, tensor, persistent=False)

        if self.log_weights.ndim != 2:
            raise ValueError("Mixture parameters must have shape [classes, components]")
        expected_shape = self.log_weights.shape
        for name in ("mu_phi", "mu_psi", "kappa_phi", "kappa_psi"):
            if getattr(self, name).shape != expected_shape:
                raise ValueError(f"{name} does not match {expected_shape}")
        if self.active_class.shape != (expected_shape[0],):
            raise ValueError("active_class must have shape [classes]")

    @classmethod
    def from_npz(cls, path):
        with np.load(Path(path), allow_pickle=False) as data:
            return cls(
                log_weights=data["log_weights"],
                mu_phi=data["mu_phi"],
                mu_psi=data["mu_psi"],
                kappa_phi=data["kappa_phi"],
                kappa_psi=data["kappa_psi"],
                active_class=data["active_class"],
            )

    def log_prob(self, phi, psi, class_id):
        class_id = class_id.to(torch.long)
        if class_id.numel() == 0:
            return phi.new_empty((0,)), class_id.new_empty((0,), dtype=torch.bool)
        if class_id.min() < 0 or class_id.max() >= self.log_weights.shape[0]:
            raise ValueError("rama_class is outside the fitted prior's class range")

        weights = self.log_weights[class_id]
        mu_phi = self.mu_phi[class_id]
        mu_psi = self.mu_psi[class_id]
        kappa_phi = self.kappa_phi[class_id]
        kappa_psi = self.kappa_psi[class_id]

        log_phi = (
            kappa_phi * torch.cos(phi.unsqueeze(-1) - mu_phi)
            - math.log(2.0 * math.pi)
            - _log_i0(kappa_phi)
        )
        log_psi = (
            kappa_psi * torch.cos(psi.unsqueeze(-1) - mu_psi)
            - math.log(2.0 * math.pi)
            - _log_i0(kappa_psi)
        )
        log_prob = torch.logsumexp(weights + log_phi + log_psi, dim=-1)
        return log_prob, self.active_class[class_id]

    def nll_from_coordinates(
        self,
        x,
        rama_phi_index,
        rama_psi_index,
        rama_class,
    ):
        phi, psi = rama_angles_from_coordinates(
            x, rama_phi_index, rama_psi_index
        )

        log_prob, valid = self.log_prob(phi, psi, rama_class)
        if valid.any():
            return -log_prob[valid].mean(), valid.sum()
        return x.new_zeros(()), x.new_zeros((), dtype=torch.long)