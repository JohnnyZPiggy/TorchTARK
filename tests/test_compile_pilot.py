import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from experiments import tark_batched_compile_pilot as pilot


def run(tmp_path, *extra):
    code = pilot.main(["--device", "cpu", "--output-dir", str(tmp_path),
        "--n-rows", "4", "--n-cols", "2", "--num-steps", "2",
        "--chain-counts", "1", "--warmup", "0", "--repeats", "1", *extra])
    return code, json.loads((tmp_path/"manifest.json").read_text())


@pytest.mark.parametrize("extra", [[], ["--dtype","float64","--burn-in","0","--nonzero-x0"],
                                      ["--dtype","float64","--burn-in","1","--nonzero-x0"]])
def test_real_cpu_cli(tmp_path, extra):
    process = subprocess.run([sys.executable, "-m", "experiments.tark_batched_compile_pilot",
        "--device", "cpu", "--output-dir", str(tmp_path), "--backend", "aot_eager",
        "--n-rows", "4", "--n-cols", "2", "--num-steps", "2", "--chain-counts", "1,2",
        "--warmup", "1", "--repeats", "2", *extra],
        cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, timeout=180)
    assert process.returncode == 0, process.stdout+process.stderr
    m = json.loads((tmp_path/"manifest.json").read_text())
    assert m["status"] == "feasibility_report_complete"
    for case in m["cases"]:
        strict, regular = case["probes"]
        assert strict["status"] in ("unsupported_fullgraph", "captured_graph_execution")
        assert regular["status"] == "partial_graph_execution"
        assert regular["graphs"] and any(g["executions"]>0 for g in regular["graphs"])
        assert regular["comparison"]["inputs_unchanged"]
        assert regular["comparison"]["max_abs_error"] <= 1e-5
        assert regular["cold_start_ms"] > 0 and regular["steady_new_graphs"] == 0
        assert len(regular["eager_elapsed_ms"]) == len(regular["compiled_elapsed_ms"]) == 2


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("burn", [0, 3, None])
def test_comparison_boundaries(dtype, burn):
    from torchtark import tark_batched
    A, b, rows = torch.eye(2,dtype=dtype), torch.tensor([4.,-1.],dtype=dtype), torch.tensor([[0,1,0,1]])
    x0 = torch.tensor([[1.,2.]],dtype=dtype)
    inputs = [A,b,rows,x0]
    result = tark_batched(A,b,rows,burn,x0)
    assert pilot.compare(result,result.clone(),inputs,[v.clone() for v in inputs])["inputs_unchanged"]
    with pytest.raises(RuntimeError, match="numerical"):
        pilot.compare(result+1,result,inputs,[v.clone() for v in inputs])


@pytest.mark.parametrize("flag,value", [("--num-steps","11"),("--num-steps","0"),
    ("--n-rows","0"),("--n-cols","65"),("--warmup","4"),("--repeats","0"),
    ("--seed","-1"),("--chain-counts","1,1"),("--chain-counts","513"),
    ("--burn-in","2"),("--chain-counts","nan")])
def test_invalid(tmp_path,flag,value):
    assert run(tmp_path,f"{flag}={value}")[0] == 1


def test_unavailable_compile(tmp_path,monkeypatch):
    monkeypatch.setattr(torch,"compile",None)
    code,m = run(tmp_path)
    assert code==1 and "unavailable" in m["error"]


def test_backend_error(tmp_path,monkeypatch):
    def fail(*args,**kwargs):
        raise RuntimeError("test backend failed")
    monkeypatch.setattr(torch,"compile",fail)
    code,m = run(tmp_path)
    assert code==1 and "backend failed" in m["error"]


def test_eager_only_not_success(tmp_path,monkeypatch):
    monkeypatch.setattr(torch,"compile",lambda fn,**kwargs: fn)
    code,m = run(tmp_path,"--route","regular")
    assert code==1 and "No captured backend graph" in m["error"]


def test_no_cuda(tmp_path,monkeypatch):
    monkeypatch.setattr(torch.cuda,"is_available",lambda:False)
    code,m = run(tmp_path,"--device","cuda")
    assert code==1 and "no CPU fallback" in m["error"]


@pytest.mark.skipif(not torch.cuda.is_available(),reason="CUDA unavailable")
def test_cuda(tmp_path):
    assert run(tmp_path,"--device","cuda","--backend","aot_eager")[0]==0
