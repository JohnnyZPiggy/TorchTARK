import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from experiments.smoke import main


def run_cli(device, output_dir):
    return subprocess.run(
        [sys.executable, "-m", "experiments.smoke", "--device", device,
         "--output-dir", str(output_dir)],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True, text=True,
    )


def test_cpu_cli(tmp_path):
    output = tmp_path / "nested" / "run"
    process = run_cli("cpu", output)
    assert process.returncode == 0, process.stdout + process.stderr
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest == {
        "selected_device": "cpu", "actual_tensor_device": "cpu",
        "pytorch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(), "seed": 0,
        "status": "success", "result": [17., 39.],
    }
    assert "success" in process.stdout


def test_cuda_unavailable(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert main(["--device", "cuda", "--output-dir", str(tmp_path)]) == 1
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["status"] == "failure"
    assert manifest["actual_tensor_device"] is None
    assert "no CPU fallback" in capsys.readouterr().out


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_cli(tmp_path):
    process = run_cli("cuda", tmp_path)
    assert process.returncode == 0, process.stdout + process.stderr
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["status"] == "success"
    assert manifest["actual_tensor_device"].startswith("cuda:")
    assert manifest["result"] == [17., 39.]


def test_invalid_device(tmp_path):
    process = run_cli("invalid", tmp_path)
    assert process.returncode != 0
    assert not (tmp_path / "manifest.json").exists()
