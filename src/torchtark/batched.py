"""Independent TARK chains vectorized across the chain axis."""

import torch

from .sampling import row_weights


@torch.no_grad()
def tark_batched(A, b, row_indices, burn_in=None, x0=None):
    """Replay K row sequences [K,T], returning post-update tail means [K,d].

    Dense finite float32/64 inputs must share A's dtype/device; integer row
    indices must be on A.device. x0 defaults to zeros [K,d]. No inputs are
    mutated and no autograd graph or iterate history is retained. Default
    burn-in is T//2; B=0 includes all T updates and B=T-1 returns final states.
    """
    if not isinstance(A, torch.Tensor) or A.layout != torch.strided:
        raise TypeError("A must be a dense tensor")
    if A.ndim != 2 or min(A.shape) < 1:
        raise ValueError("A must have shape (n,d) with n,d >= 1")
    if A.dtype not in (torch.float32, torch.float64):
        raise TypeError("A must have dtype float32 or float64")
    if not isinstance(row_indices, torch.Tensor) or row_indices.layout != torch.strided:
        raise TypeError("row_indices must be a dense integer tensor")
    if row_indices.dtype not in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
        raise TypeError("row_indices must have integer dtype")
    if row_indices.ndim != 2 or min(row_indices.shape) < 1:
        raise ValueError("row_indices must have shape (K,T) with K,T >= 1")
    if row_indices.device != A.device:
        raise ValueError("row_indices must match A device")
    K, T = row_indices.shape
    if burn_in is None:
        burn_in = T // 2
    if type(burn_in) is not int or not 0 <= burn_in < T:
        raise ValueError("burn_in must be an integer in [0,T)")
    for name, value, shape in (("A", A, A.shape), ("b", b, (A.shape[0],)),
                               ("x0", x0, (K, A.shape[1]))):
        if name == "x0" and value is None:
            continue
        if not isinstance(value, torch.Tensor) or value.layout != torch.strided:
            raise TypeError(f"{name} must be a dense tensor")
        if value.shape != shape:
            raise ValueError(f"{name} must have shape {shape}")
        if value.dtype != A.dtype or value.device != A.device:
            raise ValueError(f"{name} must match A dtype and device")
        if not torch.isfinite(value).all():
            raise ValueError(f"{name} must contain only finite values")
    weights, _ = row_weights(A)
    if ((row_indices < 0) | (row_indices >= A.shape[0])).any():
        raise ValueError("row_indices out of bounds")
    indices = row_indices.to(dtype=torch.long)
    if (weights[indices] == 0).any():
        raise ValueError("row_indices cannot select zero-norm rows")
    X = A.new_zeros((K, A.shape[1])) if x0 is None else x0.clone()
    tail = torch.zeros_like(X)
    for t in range(T):
        i = indices[:, t]
        rows = A[i]
        residual = b[i] - (rows * X).sum(dim=1)
        X.add_((residual / weights[i])[:, None] * rows)
        if t >= burn_in:
            tail.add_(X)
    result = tail / (T - burn_in)
    if not torch.isfinite(result).all():
        raise RuntimeError("Batched TARK produced nonfinite output; check input scaling")
    return result
