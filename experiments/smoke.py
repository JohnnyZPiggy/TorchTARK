"""Verify PyTorch device execution without implementing TARK."""

import argparse
import json
from pathlib import Path

import torch


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "selected_device": args.device,
        "actual_tensor_device": None,
        "pytorch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "seed": 0,
        "status": "failure",
    }
    try:
        if args.device == "cuda" and not manifest["cuda_available"]:
            raise RuntimeError("CUDA requested but CUDA is unavailable; no CPU fallback")
        torch.manual_seed(manifest["seed"])
        matrix = torch.tensor([[1., 2.], [3., 4.]], device=args.device)
        vector = torch.tensor([5., 6.], device=args.device)
        manifest["actual_tensor_device"] = str(matrix.device)
        result = (matrix @ vector).cpu().tolist()
        if result != [17., 39.]:
            raise RuntimeError(f"Numerical check failed: {result}")
        manifest.update(status="success", result=result)
    except (RuntimeError, AssertionError) as error:
        manifest["error"] = str(error)
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"{manifest['status']}: requested={args.device}, "
          f"actual={manifest['actual_tensor_device']}")
    if "error" in manifest:
        print(manifest["error"])
    return 0 if manifest["status"] == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
