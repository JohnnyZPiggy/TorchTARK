"""Single-row, single-chain, unregularized TARK."""

import torch

from .sampling import row_weights, sample_rows


@torch.no_grad()
def tark(A, b, num_steps=1000, burn_in=None, x0=None, generator=None,
         row_indices=None):
    """Average post-update iterates x_(B+1), ..., x_T using O(d) accumulation.

    Inputs must be dense finite float32/float64 tensors on the same device.
    This numerical solver does not support autograd and never mutates inputs.
    Default burn-in is T//2. Optional integer row_indices (length T, on A.device)
    replay a fixed row sequence and are mutually exclusive with generator.
    Seeded sampling is reproducible on the same device, not across device types.
    """
    if type(num_steps) is not int or num_steps <= 0:
        raise ValueError("num_steps must be a positive integer")
    if burn_in is None:
        burn_in = num_steps // 2
    if type(burn_in) is not int or not 0 <= burn_in < num_steps:
        raise ValueError("burn_in must be an integer in [0, num_steps)")
    if not isinstance(A, torch.Tensor) or A.layout != torch.strided:
        raise TypeError("A must be a dense tensor")
    if A.ndim != 2 or min(A.shape) < 1:
        raise ValueError("A must have shape (n, d) with n,d >= 1")
    if A.dtype not in (torch.float32, torch.float64):
        raise TypeError("A must have dtype float32 or float64")
    for name, value, shape in (("A", A, A.shape), ("b", b, (A.shape[0],)),
                               ("x0", x0, (A.shape[1],))):
        if value is None and name == "x0":
            continue
        if not isinstance(value, torch.Tensor) or value.layout != torch.strided:
            raise TypeError(f"{name} must be a dense tensor")
        if value.shape != shape:
            raise ValueError(f"{name} must have shape {shape}")
        if value.dtype != A.dtype or value.device != A.device:
            raise ValueError(f"{name} must match A dtype and device")
        if not torch.isfinite(value).all():
            raise ValueError(f"{name} must contain only finite values")
    weights, probabilities = row_weights(A)
    if row_indices is None:
        indices = sample_rows(probabilities, num_steps, generator)
    else:
        if generator is not None:
            raise ValueError("row_indices and generator are mutually exclusive")
        if not isinstance(row_indices, torch.Tensor):
            raise TypeError("row_indices must be an integer tensor")
        if row_indices.layout != torch.strided or row_indices.dtype not in (
                torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
            raise TypeError("row_indices must be a dense integer tensor")
        if row_indices.shape != (num_steps,) or row_indices.device != A.device:
            raise ValueError("row_indices must have length num_steps and match A device")
        if ((row_indices < 0) | (row_indices >= A.shape[0])).any():
            raise ValueError("row_indices out of bounds")
        indices = row_indices.to(dtype=torch.long)
        if (weights[indices] == 0).any():
            raise ValueError("row_indices cannot select a zero-norm row")
    x = A.new_zeros(A.shape[1]) if x0 is None else x0.clone()
    tail = torch.zeros_like(x)
    for t, i in enumerate(indices):
        row = A[i]
        x.add_(row * ((b[i] - row @ x) / weights[i]))
        if t >= burn_in:
            tail.add_(x)
    result = tail / (num_steps - burn_in)
    if not torch.isfinite(result).all():
        raise RuntimeError("TARK produced nonfinite output; check input scaling")
    return result
