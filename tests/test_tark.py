import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from torchtark import tark
from torchtark.sampling import row_weights, sample_rows


@pytest.mark.parametrize("device", ["cpu", pytest.param(
    "cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable"))])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_worked_examples(device, dtype):
    A = torch.tensor([[3., 4.]], device=device, dtype=dtype)
    b = A.new_tensor([10.])
    x0 = A.new_zeros(2, requires_grad=True)
    rows = torch.tensor([0], device=device)
    copies = [v.clone() for v in (A, b, x0, rows)]
    result = tark(A, b, 1, 0, x0=x0, row_indices=rows)
    torch.testing.assert_close(result, A.new_tensor([1.2, 1.6]))
    assert result.device == A.device and result.dtype == dtype
    assert not result.requires_grad
    for original, copy in zip((A, b, x0, rows), copies):
        assert torch.equal(original, copy)
    A = torch.eye(2, dtype=dtype, device=device)
    b = A.new_tensor([4., -1.])
    rows = torch.tensor([0, 1, 0, 1], device=device)
    x0 = A.new_zeros(2)
    for burn, expected in [(0, [4., -.75]), (1, [4., -1.]),
                           (3, [4., -1.]), (None, [4., -1.])]:
        torch.testing.assert_close(tark(A, b, 4, burn, x0=x0, row_indices=rows),
                                   A.new_tensor(expected))
    assert torch.equal(x0, A.new_zeros(2))
    A = torch.ones((2, 1), device=device, dtype=dtype)
    b = A.new_tensor([0., 2.])
    torch.testing.assert_close(tark(A, b, 4, 0, row_indices=rows), A.new_tensor([1.]))
    torch.testing.assert_close(tark(A, b, 4, 3, row_indices=rows), A.new_tensor([2.]))


@pytest.mark.parametrize("device", ["cpu", pytest.param(
    "cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable"))])
def test_sampling(device):
    A = torch.tensor([[0., 0.], [1., 0.], [0., 2.]], device=device)
    weights, probabilities = row_weights(A)
    torch.testing.assert_close(weights, A.new_tensor([0., 1., 4.]))
    torch.testing.assert_close(probabilities, A.new_tensor([0., .2, .8]))
    first = sample_rows(probabilities, 10000, torch.Generator(device=device).manual_seed(7))
    second = sample_rows(probabilities, 10000, torch.Generator(device=device).manual_seed(7))
    assert torch.equal(first, second)
    assert not (first == 0).any()
    assert abs((first == 1).float().mean().item() - .2) < .025
    b = A.new_tensor([0., 1., 2.])
    x = tark(A, b, 20, generator=torch.Generator(device=device).manual_seed(9))
    y = tark(A, b, 20, generator=torch.Generator(device=device).manual_seed(9))
    assert torch.equal(x, y)


@pytest.mark.parametrize("kwargs", [
    {"num_steps": 0}, {"num_steps": -1}, {"num_steps": True}, {"num_steps": 1.5},
    {"burn_in": -1}, {"burn_in": 4}, {"burn_in": True}, {"burn_in": 1.5},
    {"x0": torch.zeros(3)}, {"x0": torch.zeros(2, dtype=torch.float64)},
    {"x0": torch.tensor([float("nan"), 0.])},
    {"row_indices": torch.tensor([0, 0, 0, 2])},
    {"row_indices": torch.tensor([0, 0, 0, -1])},
    {"row_indices": torch.tensor([0., 0., 0., 0.])},
    {"row_indices": torch.tensor([True, False, True, False])},
    {"row_indices": torch.zeros((4, 1), dtype=torch.long)},
    {"row_indices": [0, 0, 0, 0]},
    {"row_indices": torch.zeros(4, dtype=torch.long), "generator": torch.Generator()},
    {"generator": 1},
])
def test_invalid_options(kwargs):
    options = dict(num_steps=4)
    options.update(kwargs)
    with pytest.raises((ValueError, TypeError)):
        tark(torch.eye(2), torch.ones(2), **options)


@pytest.mark.parametrize("A,b", [
    (torch.ones(2), torch.ones(2)), (torch.ones(0, 2), torch.ones(0)),
    (torch.ones(2, 0), torch.ones(2)), (torch.eye(2), torch.ones(2, 1)),
    (torch.eye(2), torch.ones(2, dtype=torch.float64)),
    (torch.eye(2, dtype=torch.int64), torch.ones(2, dtype=torch.int64)),
    (torch.eye(2, dtype=torch.float16), torch.ones(2, dtype=torch.float16)),
    (torch.eye(2, dtype=torch.complex64), torch.ones(2, dtype=torch.complex64)),
    (torch.zeros(2, 2), torch.ones(2)),
    (torch.full((2, 2), float("inf")), torch.ones(2)),
    (torch.full((2, 2), float("nan")), torch.ones(2)),
    (torch.eye(2), torch.tensor([0., float("inf")])),
    (torch.eye(2), torch.tensor([0., float("nan")])),
    (torch.full((2, 2), 1e30), torch.ones(2)),
    (torch.eye(2).to_sparse(), torch.ones(2)),
    ([[1.]], torch.ones(1)), (torch.eye(2), [1., 2.]),
])
def test_invalid_inputs(A, b):
    with pytest.raises((ValueError, TypeError)):
        tark(A, b, num_steps=4)


def test_forced_zero_row():
    with pytest.raises(ValueError, match="zero-norm"):
        tark(torch.tensor([[0.], [1.]]), torch.ones(2), 1,
             row_indices=torch.tensor([0]))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_device_mismatch():
    A = torch.eye(2, device="cuda")
    b = torch.ones(2, device="cuda")
    for kwargs in ({"x0": torch.zeros(2)}, {"row_indices": torch.zeros(4, dtype=torch.long)},
                   {"generator": torch.Generator()}):
        with pytest.raises(ValueError):
            tark(A, b, 4, **kwargs)
    with pytest.raises(ValueError):
        tark(A, b.cpu(), 4)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_correctness_cli(device, tmp_path):
    process = subprocess.run(
        [sys.executable, "-m", "experiments.tark_correctness", "--device", device,
         "--output-dir", str(tmp_path)], cwd=Path(__file__).resolve().parents[1],
        capture_output=True, text=True)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    if device == "cuda" and not torch.cuda.is_available():
        assert process.returncode == 1
        assert manifest["status"] == "failure"
        assert manifest["actual_tensor_device"] is None
        assert "no CPU fallback" in manifest["error"]
    else:
        assert process.returncode == 0, process.stdout + process.stderr
        assert manifest["status"] == "success"
        assert manifest["actual_tensor_device"].startswith(device)
        assert all(check["passed"] for check in manifest["checks"])
