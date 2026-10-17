"""Measure the public eager TARK fixed-row replay path."""

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import statistics
import subprocess
import time

import torch

from torchtark import tark


def timed_call(call, device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    result = call()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = (time.perf_counter() - start) * 1000
    if not math.isfinite(elapsed) or elapsed < 0:
        raise RuntimeError("Invalid elapsed duration")
    return result, elapsed


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    for name, default in (("n-rows", 1024), ("n-cols", 64), ("num-steps", 500),
                          ("warmup", 2), ("repeats", 5), ("seed", 7)):
        parser.add_argument(f"--{name}", default=str(default))
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[1],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    manifest = dict(
        status="failure", exit_status=1, selected_device=args.device,
        actual_tensor_device=None, cuda_available=torch.cuda.is_available(),
        gpu_name=None, pytorch_version=torch.__version__, source_commit=commit,
        timestamp_utc=datetime.now(timezone.utc).isoformat(), dtype="torch.float32",
        parameters={key: getattr(args, key) for key in
                    ("n_rows", "n_cols", "num_steps", "warmup", "repeats", "seed")},
        elapsed_ms=[], median_ms=None, min_ms=None, max_ms=None,
        timing_method="perf_counter wall clock; CUDA synchronized before/after each call",
        timed_region="public tark call including validation and default zero x0; burn_in=T//2",
        excluded="input/row generation, transfers, output checks, JSON and stdout",
        generation="CPU float32 randn A then b; one seeded CPU generator; CPU norm-squared multinomial replacement row sequence; transfer before timing",
    )
    try:
        p = {key: int(value) for key, value in manifest["parameters"].items()}
        manifest["parameters"] = p
        if any(p[key] < 1 for key in ("n_rows", "n_cols", "num_steps", "repeats")) or p["warmup"] < 0:
            raise ValueError("Dimensions, steps and repeats must be positive; warmup >= 0")
        if not 0 <= p["seed"] < 2**63:
            raise ValueError("seed must be in [0, 2**63)")
        # Bound allocations and loop work for this small baseline, not large-scale benchmarking.
        if (p["n_rows"] * p["n_cols"] > 10_000_000 or p["n_rows"] > 1_000_000
                or p["num_steps"] > 100_000 or p["warmup"] + p["repeats"] > 100
                or p["num_steps"] * (p["warmup"] + p["repeats"]) * p["n_cols"] > 100_000_000):
            raise ValueError("Requested allocation/work exceeds small baseline resource limits")
        if args.device == "cuda" and not manifest["cuda_available"]:
            raise RuntimeError("CUDA requested but unavailable; no CPU fallback")
        device = torch.device(args.device)
        if device.type == "cuda":
            manifest["gpu_name"] = torch.cuda.get_device_name(device)
        generator = torch.Generator(device="cpu").manual_seed(p["seed"])
        A = torch.randn(p["n_rows"], p["n_cols"], generator=generator, dtype=torch.float32, device="cpu")
        b = torch.randn(p["n_rows"], generator=generator, dtype=torch.float32, device="cpu")
        weights = A.square().sum(dim=1)
        rows = torch.multinomial(weights / weights.sum(), p["num_steps"],
                                 replacement=True, generator=generator)
        A, b, rows = A.to(device), b.to(device), rows.to(device)
        reference = None

        def call():
            return tark(A, b, num_steps=p["num_steps"], row_indices=rows)

        def check(result):
            nonlocal reference
            if (result.shape != (p["n_cols"],) or result.device != A.device
                    or result.dtype != torch.float32 or not torch.isfinite(result).all()):
                raise RuntimeError("Invalid solver output shape/device/dtype/finiteness")
            if reference is None:
                reference = result.clone()
            elif not torch.allclose(result, reference, rtol=1e-5, atol=1e-6):
                raise RuntimeError("Repeated solver outputs disagree")

        for _ in range(p["warmup"]):
            check(call())
        for _ in range(p["repeats"]):
            result, elapsed = timed_call(call, device)
            check(result)
            manifest["elapsed_ms"].append(elapsed)
        checksum = result.double().sum().item()
        if not math.isfinite(checksum):
            raise RuntimeError("Nonfinite output checksum")
        durations = manifest["elapsed_ms"]
        manifest.update(status="success", exit_status=0,
                        actual_tensor_device=str(result.device), output_finite=True,
                        output_checksum=checksum, median_ms=statistics.median(durations),
                        min_ms=min(durations), max_ms=max(durations))
    except (ValueError, RuntimeError, TypeError, OverflowError, TimeoutError) as error:
        manifest["error"] = str(error)
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"{manifest['status']}: device={args.device}, median_ms={manifest['median_ms']}")
    if "error" in manifest:
        print(manifest["error"])
    return manifest["exit_status"]


if __name__ == "__main__":
    raise SystemExit(main())
