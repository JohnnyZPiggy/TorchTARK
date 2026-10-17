"""Measure batched TARK latency and aggregate independent-chain throughput."""

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import statistics
import subprocess

import torch

from torchtark import tark_batched
from experiments.tark_benchmark import timed_call


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--chain-counts", default="1,8,32,128,512")
    for flag, default in (("n-rows", 1024), ("n-cols", 64), ("num-steps", 500),
                          ("warmup", 2), ("repeats", 5), ("seed", 7)):
        parser.add_argument(f"--{flag}", default=str(default))
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[1],
            capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    m = dict(status="failure", exit_status=1, selected_device=args.device,
             actual_tensor_device=None, cuda_available=torch.cuda.is_available(),
             gpu_name=None, pytorch_version=torch.__version__, source_commit=commit,
             timestamp_utc=datetime.now(timezone.utc).isoformat(), dtype="torch.float32",
             parameters={key: getattr(args, key) for key in
                         ("n_rows", "n_cols", "num_steps", "warmup", "repeats", "seed", "chain_counts")},
             measurements=[],
             random_policy="One seeded CPU generator: float32 randn A then b, then norm-squared replacement multinomial max(K)*T rows reshaped [max(K),T]; each K uses prefix; transfers before timing",
             timing_method="perf_counter wall clock; CUDA synchronize before/after every timed call; warmups unrecorded",
             timed_region="public tark_batched including validation, default zero x0 and burn_in=T//2",
             excluded="generation, stochastic sampling, transfers, output checks, JSON and stdout",
             caveats="Batch latency and aggregate chain throughput, not single-chain speedup. K=1 uses batched API, not comparable to scalar tark benchmark. Compare CPU/GPU only with matched inputs in same runtime.")
    try:
        p = {key: int(value) for key, value in m["parameters"].items() if key != "chain_counts"}
        counts = [int(value) for value in args.chain_counts.split(",")]
        if not counts or any(k < 1 for k in counts) or len(set(counts)) != len(counts):
            raise ValueError("chain-counts must contain distinct positive integers")
        p["chain_counts"] = counts
        m["parameters"] = p
        if (any(p[key] < 1 for key in ("n_rows", "n_cols", "num_steps", "repeats"))
                or p["warmup"] < 0 or not 0 <= p["seed"] < 2**63):
            raise ValueError("Positive sizes/steps/repeats, warmup >= 0 and seed in [0,2**63) required")
        Kmax, T, d = max(counts), p["num_steps"], p["n_cols"]
        calls = p["warmup"] + p["repeats"]
        # Bound input/state allocation and aggregate work, including warmups.
        if (len(counts) > 32 or p["n_rows"] > 1_000_000 or p["n_rows"] * d > 10_000_000
                or Kmax * T > 2_000_000 or Kmax * d > 2_000_000
                or T > 100_000 or calls > 100
                or sum(counts) * T * calls > 5_000_000
                or sum(counts) * T * calls * d > 500_000_000):
            raise ValueError("Requested allocation/work exceeds batched baseline limits")
        if args.device == "cuda" and not m["cuda_available"]:
            raise RuntimeError("CUDA requested but unavailable; no CPU fallback")
        device = torch.device(args.device)
        if device.type == "cuda":
            m["gpu_name"] = torch.cuda.get_device_name(device)
        rng = torch.Generator(device="cpu").manual_seed(p["seed"])
        A = torch.randn(p["n_rows"], d, generator=rng, device="cpu", dtype=torch.float32)
        b = torch.randn(p["n_rows"], generator=rng, device="cpu", dtype=torch.float32)
        weights = A.square().sum(dim=1)
        rows = torch.multinomial(weights / weights.sum(), Kmax * T,
                                 replacement=True, generator=rng).reshape(Kmax, T)
        A, b, rows = A.to(device), b.to(device), rows.to(device)
        for K in counts:
            reference = None
            durations = []

            def call():
                return tark_batched(A, b, rows[:K], burn_in=None, x0=None)

            def check(result):
                nonlocal reference
                if (result.shape != (K, d) or result.dtype != torch.float32
                        or result.device != A.device or not torch.isfinite(result).all()):
                    raise RuntimeError(f"K={K}: invalid output shape/dtype/device/finiteness")
                if reference is None:
                    reference = result.clone()
                elif not torch.allclose(result, reference, rtol=1e-5, atol=1e-6):
                    raise RuntimeError(f"K={K}: repeated outputs disagree")

            for _ in range(p["warmup"]):
                check(call())
            for _ in range(p["repeats"]):
                result, elapsed = timed_call(call, device)
                if not math.isfinite(elapsed) or elapsed <= 0:
                    raise RuntimeError("Elapsed milliseconds must be finite and positive")
                check(result)
                durations.append(elapsed)
            median = statistics.median(durations)
            checksum = result.double().sum().item()
            cps = K * 1000 / median
            steps = cps * T
            if not all(math.isfinite(v) for v in (checksum, cps, steps)):
                raise RuntimeError("Nonfinite checksum or throughput")
            m["actual_tensor_device"] = str(result.device)
            m["measurements"].append(dict(
                chains=K, elapsed_ms=durations, median_ms=median,
                min_ms=min(durations), max_ms=max(durations),
                chains_per_second=cps, chain_steps_per_second=steps,
                output_finite=True, output_checksum=checksum))
        m.update(status="success", exit_status=0)
    except (ValueError, RuntimeError, TypeError, OverflowError, TimeoutError) as error:
        m["error"] = str(error)
    (args.output_dir / "manifest.json").write_text(
        json.dumps(m, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"{m['status']}: device={args.device}, completed_K={[v['chains'] for v in m['measurements']]}")
    if "error" in m:
        print(m["error"])
    return m["exit_status"]


if __name__ == "__main__":
    raise SystemExit(main())
