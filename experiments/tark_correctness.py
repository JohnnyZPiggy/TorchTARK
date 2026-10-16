"""Run small deterministic checks through the public TARK API."""

import argparse
import json
from pathlib import Path
import subprocess

import torch

from torchtark import tark


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[1],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    manifest = dict(selected_device=args.device, actual_tensor_device=None,
                    commit=commit, pytorch_version=torch.__version__,
                    cuda_available=torch.cuda.is_available(), dtype="torch.float64",
                    status="failure", checks=[])
    try:
        if args.device == "cuda" and not manifest["cuda_available"]:
            raise RuntimeError("CUDA requested but unavailable; no CPU fallback")
        cases = [
            ("consistent", [[1., 0.], [0., 1.]], [4., -1.], [4., -.75]),
            ("inconsistent", [[1.], [1.]], [0., 2.], [1.]),
        ]
        for name, matrix, rhs, expected in cases:
            A = torch.tensor(matrix, dtype=torch.float64, device=args.device)
            b = torch.tensor(rhs, dtype=A.dtype, device=A.device)
            rows = torch.tensor([0, 1, 0, 1], device=A.device)
            result = tark(A, b, num_steps=4, burn_in=0, row_indices=rows)
            manifest["actual_tensor_device"] = str(result.device)
            passed = torch.allclose(result, result.new_tensor(expected))
            manifest["checks"].append(dict(name=name, expected=expected,
                                           observed=result.cpu().tolist(), passed=passed))
            if not passed or result.device != A.device:
                raise RuntimeError(f"{name} correctness/device check failed")
        manifest["status"] = "success"
    except (RuntimeError, ValueError) as error:
        manifest["error"] = str(error)
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"{manifest['status']}: requested={args.device}, "
          f"actual={manifest['actual_tensor_device']}")
    if "error" in manifest:
        print(manifest["error"])
    return 0 if manifest["status"] == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
