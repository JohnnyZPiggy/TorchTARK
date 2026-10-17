"""Compare public batched TARK with scalar chain replays."""

import argparse
import json
from pathlib import Path
import subprocess

import torch

from torchtark import tark, tark_batched


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
    manifest = dict(status="failure", selected_device=args.device,
                    actual_tensor_device=None, source_commit=commit,
                    pytorch_version=torch.__version__, dtype="torch.float64",
                    cuda_available=torch.cuda.is_available(), comparisons=[])
    try:
        if args.device == "cuda" and not manifest["cuda_available"]:
            raise RuntimeError("CUDA requested but unavailable; no CPU fallback")
        cases = [
            ("consistent", [[1., 0.], [0., 1.]], [4., -1.]),
            ("inconsistent", [[1.], [1.]], [0., 2.]),
        ]
        for name, matrix, rhs in cases:
            A = torch.tensor(matrix, dtype=torch.float64, device=args.device)
            b = A.new_tensor(rhs)
            rows = torch.tensor([[0, 1, 0, 1], [1, 1, 0, 0], [0, 0, 0, 1]], device=A.device)
            for K in (1, 3):
                result = tark_batched(A, b, rows[:K], burn_in=0)
                expected = torch.stack([tark(A, b, 4, 0, row_indices=r) for r in rows[:K]])
                # First chain has a closed-form worked answer as well as the scalar oracle.
                worked = A.new_tensor([4., -.75] if name == "consistent" else [1.])
                passed = (torch.allclose(result, expected) and
                          torch.allclose(result[0], worked) and
                          result.device == A.device and bool(torch.isfinite(result).all()))
                manifest["actual_tensor_device"] = str(result.device)
                manifest["comparisons"].append(dict(
                    case=name, chains=K, burn_in=0, expected=expected.cpu().tolist(),
                    observed=result.cpu().tolist(),
                    max_abs_error=(result - expected).abs().max().item(), passed=passed))
                if not passed:
                    raise RuntimeError(f"{name}, K={K}: comparison failed")
        manifest["status"] = "success"
    except (ValueError, RuntimeError) as error:
        manifest["error"] = str(error)
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"{manifest['status']}: requested={args.device}, actual={manifest['actual_tensor_device']}")
    if "error" in manifest:
        print(manifest["error"])
    return 0 if manifest["status"] == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
