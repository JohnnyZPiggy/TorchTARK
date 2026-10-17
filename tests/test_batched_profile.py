import json
import math
from types import SimpleNamespace

import pytest
import torch

from experiments import tark_batched_profile as cli


def run(tmp_path, *extra):
    code = cli.main(["--device", "cpu", "--output-dir", str(tmp_path),
        "--n-rows", "4", "--n-cols", "2", "--num-steps", "4",
        "--chain-counts", "1,2", "--warmup", "0", *extra])
    return code, json.loads((tmp_path / "manifest.json").read_text())


def test_cpu(tmp_path):
    code, m = run(tmp_path)
    assert code == m["exit_status"] == 0 and m["status"] == "success"
    assert m["actual_tensor_device"] == "cpu" and m["operator_time_units"] == "microseconds"
    assert m["source_commit"] is None or len(m["source_commit"]) == 40
    assert m["python_version"] and m["pytorch_version"] and m["timestamp_utc"]
    assert [p["chains"] for p in m["profiles"]] == [1, 2]
    _, again = run(tmp_path)
    for p, q in zip(m["profiles"], again["profiles"]):
        assert p["output_shape"] == [p["chains"], 2] and p["output_finite"]
        assert p["output_checksum"] == q["output_checksum"]
        assert p["top_operators"] and len(p["top_operators"]) <= 30
        assert p["gpu_kernel_events"]["status"] == "unavailable"
        for row in p["top_operators"]:
            assert row["key"] and row["count"] >= 1
            assert all(v is None or math.isfinite(v) and v >= 0
                       for key, v in row.items() if key in cli.TIME_FIELDS)
        assert any(row["key"] == "aten::index" for row in p["watched_operators"])


def test_no_cuda(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    code, m = run(tmp_path, "--device", "cuda")
    assert code == 1 and m["status"] == "failure" and "no CPU fallback" in m["error"]
    assert not m["profiles"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda(tmp_path):
    code, m = run(tmp_path, "--device", "cuda")
    assert code == 0 and m["actual_tensor_device"].startswith("cuda:") and m["gpu_name"]


@pytest.mark.parametrize("flag,value", [("--n-rows", "0"), ("--n-cols", "-1"),
    ("--num-steps", "0"), ("--num-steps", "101"), ("--warmup", "-1"),
    ("--warmup", "11"), ("--seed", "-1"), ("--chain-counts", ""),
    ("--chain-counts", "1,1"), ("--chain-counts", "0"), ("--chain-counts", "513"),
    ("--chain-counts", "1,2,3,4,5"), ("--chain-counts", "1.5"),
    ("--n-rows", "1000001"), ("--n-cols", "1000001"), ("--n-cols", "nan")])
def test_invalid(tmp_path, flag, value):
    code, m = run(tmp_path, f"{flag}={value}")
    assert code == 1 and m["status"] == "failure" and m["error"]


def test_unsupported_activity(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.profiler, "supported_activities", lambda: set())
    assert run(tmp_path)[0] == 1


def test_profiler_failure(tmp_path, monkeypatch):
    def fail(**kwargs):
        raise RuntimeError("profiler unsupported in test")
    monkeypatch.setattr(torch.profiler, "profile", fail)
    code, m = run(tmp_path)
    assert code == 1 and "unsupported" in m["error"] and not m["profiles"]


def test_nonfinite_output(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "tark_batched", lambda A, b, rows:
                        A.new_full((rows.shape[0], A.shape[1]), float("nan")))
    code, m = run(tmp_path)
    assert code == 1 and m["status"] == "failure" and not m["profiles"]


def test_summary_missing_fields():
    fake = SimpleNamespace(key_averages=lambda: [SimpleNamespace(
        key="aten::index", count=2, self_cpu_time_total=1., cpu_time_total=2.)])
    report = cli.summarize(fake)
    assert report["missing_api_fields"] == ["device_time_total", "self_device_time_total"]
    assert report["totals"]["self_device_time_total"] is None
    fake.key_averages = lambda: []
    with pytest.raises(RuntimeError, match="unsupported or empty"):
        cli.summarize(fake)
    fake.key_averages = lambda: [SimpleNamespace(key="test", count=1,
        self_cpu_time_total=float("nan"), cpu_time_total=1.)]
    with pytest.raises(RuntimeError, match="Invalid profiler time"):
        cli.summarize(fake)


def test_profiler_scope(tmp_path, monkeypatch):
    events = []
    class FakeProfile:
        def __init__(self, **kwargs):
            assert kwargs["record_shapes"] is kwargs["profile_memory"] is kwargs["with_stack"] is False
        def __enter__(self):
            events.append("enter")
            return self
        def __exit__(self, *args):
            events.append("exit")
        def key_averages(self):
            return [SimpleNamespace(key="test", count=1, self_cpu_time_total=1., cpu_time_total=1.)]
    def solver(A, b, rows):
        events.append("solver")
        return A.new_zeros((rows.shape[0], A.shape[1]))
    monkeypatch.setattr(torch.profiler, "profile", FakeProfile)
    monkeypatch.setattr(cli, "tark_batched", solver)
    assert run(tmp_path, "--warmup", "1")[0] == 0
    assert events == ["solver", "enter", "solver", "exit"] * 2
