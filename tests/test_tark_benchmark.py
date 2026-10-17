import json
import math

import pytest
import torch

from experiments import tark_benchmark as bench


def run(tmp_path, *extra):
    code = bench.main(["--device", "cpu", "--output-dir", str(tmp_path),
                       "--n-rows", "4", "--n-cols", "2", "--num-steps", "3",
                       "--warmup", "1", "--repeats", "2", *extra])
    return code, json.loads((tmp_path / "manifest.json").read_text())


def test_cpu(tmp_path):
    code, m = run(tmp_path)
    assert code == m["exit_status"] == 0
    assert m["status"] == "success" and m["actual_tensor_device"] == "cpu"
    assert m["dtype"] == "torch.float32" and m["output_finite"]
    assert len(m["elapsed_ms"]) == 2
    assert all(math.isfinite(t) and t >= 0 for t in m["elapsed_ms"])
    assert m["min_ms"] <= m["median_ms"] <= m["max_ms"]
    assert m["source_commit"] is None or len(m["source_commit"]) == 40
    assert m["timestamp_utc"] and m["parameters"]["seed"] == 7
    _, repeat = run(tmp_path)
    assert m["output_checksum"] == repeat["output_checksum"]


def test_cuda_unavailable(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    code, m = run(tmp_path, "--device", "cuda")
    assert code == 1 and m["status"] == "failure"
    assert "no CPU fallback" in m["error"] and m["elapsed_ms"] == []


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda(tmp_path):
    code, m = run(tmp_path, "--device", "cuda")
    assert code == 0 and m["actual_tensor_device"].startswith("cuda:")
    assert m["gpu_name"]


def test_timing_order(monkeypatch):
    events = []
    ticks = iter([1., 1.25])
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: events.append("sync"))
    def clock():
        events.append("clock")
        return next(ticks)
    monkeypatch.setattr(bench.time, "perf_counter", clock)
    def call():
        events.append("solver")
        return 42
    assert bench.timed_call(call, torch.device("cuda")) == (42, 250.)
    assert events == ["sync", "clock", "solver", "sync", "clock"]


@pytest.mark.parametrize("flag,value", [("--n-rows", "0"), ("--n-cols", "-1"),
    ("--num-steps", "0"), ("--warmup", "-1"), ("--repeats", "0"),
    ("--seed", "-1"), ("--n-rows", "1000000000"), ("--repeats", "101"),
    ("--n-cols", "nan"), ("--num-steps", "1.5")])
def test_bad_args(tmp_path, flag, value):
    code, m = run(tmp_path, flag, value)
    assert code == 1 and m["status"] == "failure" and m["error"]
    assert m["elapsed_ms"] == []


@pytest.mark.parametrize("failure", ["nan", "shape", "dtype", "repeat", "timeout"])
def test_bad_solver(tmp_path, monkeypatch, failure):
    calls = 0
    def solver(A, b, **kwargs):
        nonlocal calls
        calls += 1
        if failure == "timeout":
            raise TimeoutError("test timeout")
        if failure == "nan":
            return A.new_full((2,), float("nan"))
        if failure == "shape":
            return A.new_zeros(3)
        if failure == "dtype":
            return A.new_zeros(2, dtype=torch.float64)
        return A.new_full((2,), float(calls))
    monkeypatch.setattr(bench, "tark", solver)
    code, m = run(tmp_path)
    assert code == 1 and m["status"] == "failure" and m["error"]


def test_zero_warmup(tmp_path):
    assert run(tmp_path, "--warmup", "0", "--n-rows", "1", "--n-cols", "1")[0] == 0
