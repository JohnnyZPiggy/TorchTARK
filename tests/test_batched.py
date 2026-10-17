import json

import pytest
import torch

from torchtark import tark, tark_batched
from experiments import tark_batched_correctness as cli


DEVICES = ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA unavailable"))]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("burn", [0, 1, 3, None])
def test_scalar_equivalence(device, dtype, burn):
    for matrix, rhs in [([[2., 1.], [1., 3.]], [3., 4.]),
                        ([[1., 0.], [0., 1.], [1., 1.]], [1., 2., 5.])]:
        A = torch.tensor(matrix, device=device, dtype=dtype)
        b = A.new_tensor(rhs)
        rows = torch.tensor([[0, 1, 0, 1], [1, 1, 0, 0], [0, 0, 1, 1]], device=device)
        if A.shape[0] == 3:
            rows[2, 2:] = 2
        x0 = A.new_tensor([[1., -2.], [3., 1.], [-1., 4.]], requires_grad=True)
        originals = [v.clone() for v in (A, b, rows, x0)]
        for K in (1, 3):
            actual = tark_batched(A, b, rows[:K], burn, x0[:K])
            expected = torch.stack([tark(A, b, 4, burn, x0[k], row_indices=rows[k])
                                    for k in range(K)])
            torch.testing.assert_close(actual, expected)
            assert actual.shape == (K, 2) and actual.device == A.device
            assert actual.dtype == dtype and not actual.requires_grad
        for value, original in zip((A, b, rows, x0), originals):
            assert torch.equal(value, original)


@pytest.mark.parametrize("burn,expected", [(0, [[4., -.75], [2., -1.]]),
    (3, [[4., -1.], [4., -1.]]), (None, [[4., -1.], [4., -1.]])])
def test_tail_worked(burn, expected):
    torch.testing.assert_close(tark_batched(torch.eye(2), torch.tensor([4., -1.]),
        torch.tensor([[0, 1, 0, 1], [1, 1, 0, 0]]), burn), torch.tensor(expected))


@pytest.mark.parametrize("changes", [
    {"A": torch.ones(2)}, {"A": torch.ones(0, 2)}, {"A": torch.ones(2, 0)},
    {"A": [[1.]]}, {"A": torch.eye(2).to_sparse()},
    {"A": torch.eye(2, dtype=torch.int64)}, {"A": torch.eye(2, dtype=torch.float16)},
    {"A": torch.eye(2, dtype=torch.complex64)}, {"A": torch.zeros(2, 2)},
    {"A": torch.full((2, 2), float("inf"))}, {"A": torch.full((2, 2), float("nan"))},
    {"A": torch.full((2, 2), 1e30)}, {"b": [1., 2.]}, {"b": torch.ones(2, 1)},
    {"b": torch.ones(2, dtype=torch.float64)}, {"b": torch.tensor([float("nan"), 1.])},
    {"x0": torch.ones(2)}, {"x0": torch.ones(1, 2, dtype=torch.float64)},
    {"x0": torch.full((1, 2), float("inf"))}, {"x0": [[1., 2.]]},
    {"row_indices": [[0, 1]]}, {"row_indices": torch.tensor([0, 1])},
    {"row_indices": torch.empty(0, 2, dtype=torch.long)},
    {"row_indices": torch.empty(1, 0, dtype=torch.long)},
    {"row_indices": torch.tensor([[0., 1.]])}, {"row_indices": torch.tensor([[True]])},
    {"row_indices": torch.tensor([[-1, 0]])}, {"row_indices": torch.tensor([[0, 2]])},
    {"row_indices": torch.tensor([[0, 1]]).to_sparse()},
    {"burn_in": -1}, {"burn_in": 2}, {"burn_in": True}, {"burn_in": 0.5},
])
def test_invalid_contract(changes):
    kwargs = dict(A=torch.eye(2), b=torch.ones(2), row_indices=torch.tensor([[0, 1]]))
    kwargs.update(changes)
    with pytest.raises((TypeError, ValueError)):
        tark_batched(**kwargs)


def test_zero_row_and_nonfinite_output():
    with pytest.raises(ValueError, match="zero-norm"):
        tark_batched(torch.tensor([[0.], [1.]]), torch.ones(2), torch.tensor([[0]]))
    with pytest.raises(RuntimeError, match="nonfinite"):
        tark_batched(torch.ones(1, 1), torch.tensor([3e38]), torch.tensor([[0, 0]]), 0)


@pytest.mark.parametrize("index_dtype", [torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64])
def test_single_step(index_dtype):
    result = tark_batched(torch.tensor([[3., 4.]]), torch.tensor([10.]),
                          torch.zeros((2, 1), dtype=index_dtype))
    torch.testing.assert_close(result, torch.tensor([[1.2, 1.6], [1.2, 1.6]]))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_device_mismatch():
    A, b, rows = torch.eye(2, device="cuda"), torch.ones(2, device="cuda"), torch.tensor([[0, 1]], device="cuda")
    for changes in ({"b": b.cpu()}, {"row_indices": rows.cpu()}, {"x0": torch.zeros(1, 2)}):
        kwargs = dict(A=A, b=b, row_indices=rows)
        kwargs.update(changes)
        with pytest.raises(ValueError):
            tark_batched(**kwargs)


@pytest.mark.parametrize("device", DEVICES)
def test_cli(tmp_path, device):
    assert cli.main(["--device", device, "--output-dir", str(tmp_path)]) == 0
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["status"] == "success"
    assert len(manifest["comparisons"]) == 4
    assert all(c["passed"] for c in manifest["comparisons"])
    assert manifest["actual_tensor_device"].startswith(device)


def test_cli_no_cuda(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert cli.main(["--device", "cuda", "--output-dir", str(tmp_path)]) == 1
    m = json.loads((tmp_path / "manifest.json").read_text())
    assert m["status"] == "failure" and "no CPU fallback" in m["error"]
