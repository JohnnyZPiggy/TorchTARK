import json
import math

import pytest
import torch

from experiments import tark_batched_benchmark as bench


def run(tmp_path, *extra):
    code = bench.main(["--device", "cpu", "--output-dir", str(tmp_path),
                       "--n-rows", "4", "--n-cols", "2", "--num-steps", "3",
                       "--chain-counts", "1,2,4", "--warmup", "0", "--repeats", "2", *extra])
    return code, json.loads((tmp_path / "manifest.json").read_text())


def test_cpu_schema_and_replay(tmp_path):
    code, m = run(tmp_path)
    assert code == m["exit_status"] == 0 and m["status"] == "success"
    assert m["actual_tensor_device"] == "cpu" and m["dtype"] == "torch.float32"
    assert m["source_commit"] is None or len(m["source_commit"]) == 40
    assert m["timestamp_utc"] and m["random_policy"] and m["caveats"]
    rng = torch.Generator().manual_seed(7)
    A, b = torch.randn(4, 2, generator=rng), torch.randn(4, generator=rng)
    w = A.square().sum(1)
    rows = torch.multinomial(w/w.sum(), 12, replacement=True, generator=rng).reshape(4, 3)
    from torchtark import tark
    for entry, K in zip(m["measurements"], (1, 2, 4)):
        assert entry["chains"] == K and len(entry["elapsed_ms"]) == 2
        assert all(math.isfinite(t) and t > 0 for t in entry["elapsed_ms"])
        assert entry["min_ms"] == min(entry["elapsed_ms"])
        assert entry["max_ms"] == max(entry["elapsed_ms"])
        assert entry["median_ms"] == pytest.approx(sum(entry["elapsed_ms"])/2)
        assert entry["chains_per_second"] == pytest.approx(K*1000/entry["median_ms"])
        assert entry["chain_steps_per_second"] == pytest.approx(3*entry["chains_per_second"])
        expected = torch.stack([tark(A, b, 3, row_indices=r) for r in rows[:K]])
        assert entry["output_checksum"] == pytest.approx(expected.double().sum().item(), abs=1e-5)
        assert entry["output_finite"]


def test_cuda_unavailable(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    code, m = run(tmp_path, "--device", "cuda")
    assert code == 1 and m["status"] == "failure" and not m["measurements"]
    assert "no CPU fallback" in m["error"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda(tmp_path):
    code, m = run(tmp_path, "--device", "cuda")
    assert code == 0 and m["actual_tensor_device"].startswith("cuda:") and m["gpu_name"]


def test_sync_order(monkeypatch):
    from experiments import tark_benchmark as timing
    events, ticks = [], iter([1., 1.1])
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: events.append("sync"))
    def clock():
        events.append("clock")
        return next(ticks)
    monkeypatch.setattr(timing.time, "perf_counter", clock)
    bench.timed_call(lambda: events.append("solver"), torch.device("cuda"))
    assert events == ["sync", "clock", "solver", "sync", "clock"]


@pytest.mark.parametrize("flag,value", [
    ("--chain-counts", ""), ("--chain-counts", "0"), ("--chain-counts", "-1,2"),
    ("--chain-counts", "1,1"), ("--chain-counts", "1.5"), ("--chain-counts", "1,"),
    ("--chain-counts", "99999999"), ("--n-rows", "0"), ("--n-cols", "0"),
    ("--n-rows", "10000001"), ("--n-cols", "10000001"), ("--num-steps", "1000001"),
    ("--warmup", "-1"), ("--repeats", "0"), ("--repeats", "101"), ("--seed", "-1"),
    ("--n-rows", "nan")])
def test_invalid(tmp_path, flag, value):
    # Equals form permits a negative comma-separated argument through argparse.
    code, m = run(tmp_path, f"{flag}={value}")
    assert code == 1 and m["status"] == "failure" and m["error"]


@pytest.mark.parametrize("kind", ["nan", "shape", "dtype", "repeat", "partial"])
def test_bad_output(tmp_path, monkeypatch, kind):
    count = 0
    def solver(A, b, rows, **kwargs):
        nonlocal count
        count += 1
        if kind == "partial" and rows.shape[0] > 1:
            raise RuntimeError("later K failed")
        shape = (rows.shape[0], 3 if kind == "shape" else 2)
        return torch.full(shape, float("nan") if kind == "nan" else
                          float(count) if kind == "repeat" else 0.,
                          dtype=torch.float64 if kind == "dtype" else torch.float32)
    monkeypatch.setattr(bench, "tark_batched", solver)
    code, m = run(tmp_path)
    assert code == 1 and m["status"] == "failure"
    if kind == "partial":
        assert len(m["measurements"]) == 1


@pytest.mark.parametrize("duration", [0., float("nan"), float("inf"), -1.])
def test_invalid_duration(tmp_path, monkeypatch, duration):
    monkeypatch.setattr(bench, "timed_call", lambda call, device: (call(), duration))
    assert run(tmp_path)[0] == 1


def test_warmup_and_minimal(tmp_path):
    assert run(tmp_path, "--chain-counts", "1", "--n-rows", "1", "--n-cols", "1",
               "--num-steps", "1", "--warmup", "1")[0] == 0
