"""Squared-row-norm sampling shared by CPU and CUDA."""

import torch


def row_weights(A):
    weights = A.square().sum(dim=1)
    total = weights.sum()
    if not torch.isfinite(weights).all() or not torch.isfinite(total) or total <= 0:
        raise ValueError("Squared row norms must be finite with positive total weight")
    return weights, weights / total


def sample_rows(probabilities, num_steps, generator=None):
    if generator is not None:
        if not isinstance(generator, torch.Generator):
            raise TypeError("generator must be a torch.Generator")
        if generator.device != probabilities.device:
            raise ValueError("generator must match the sampling device")
    return torch.multinomial(
        probabilities, num_steps, replacement=True, generator=generator
    )
